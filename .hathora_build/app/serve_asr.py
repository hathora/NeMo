import logging
import os
import signal
import struct
import subprocess
import sys
import tempfile
import wave
from typing import List, Optional, Union

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
os.environ["TQDM_DISABLE"] = "0"

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import nemo.collections.asr as nemo_asr
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def signal_handler(sig, frame):
    logger.info(f"Received signal {sig}. Shutting down gracefully...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://(hathora-voice|hathora-voice-.*|hathora-voice-.*-hathora)\.vercel\.app|https://models\.hathora\.dev",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "ETag",
        "X-Hathora-Process-Id",
        "X-Hathora-Region",
        "X-Hathora-Request-Duration",
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "https://localhost",
        "https://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Hathora-Process-Id",
        "X-Hathora-Region",
        "X-Hathora-Request-Duration",
    ],
)

MODEL_ID = os.getenv("MODEL_ID", "nvidia/parakeet-tdt-0.6b-v3")


class ModelManager:
    """Singleton manager for the ASR model."""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self):
        """Get or load the ASR model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self):
        """Load the ASR model with progress tracking."""
        import time
        
        logger.info(f"Loading model: {MODEL_ID}")
        logger.info("Fetching model from Hugging Face...")
        
        start_time = time.time()
        
        try:
            model = nemo_asr.models.ASRModel.from_pretrained(MODEL_ID)
            download_time = time.time() - start_time
            logger.info(f"Model loaded from cache/download complete (took {download_time:.2f}s)")
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise
        
        if torch.cuda.is_available():
            logger.info(f"CUDA available - GPU: {torch.cuda.get_device_name(0)}")
            gpu_start = time.time()
            model = model.cuda()
            logger.info(f"Model moved to GPU (took {time.time() - gpu_start:.2f}s)")
        else:
            logger.warning("CUDA not available - using CPU (inference will be VERY slow)")
        
        model.eval()
        self._model = model
        total_time = time.time() - start_time
        logger.info(f"Model ready for inference (total startup: {total_time:.2f}s)")


model_manager = ModelManager()


def extract_texts(transcribe_result) -> List[str]:
    """Extract text from transcription results, handling various return types."""
    if not transcribe_result:
        return [""]
    
    if isinstance(transcribe_result, tuple):
        transcribe_result = transcribe_result[0]
    
    if isinstance(transcribe_result, list) and transcribe_result:
        first_item = transcribe_result[0]
        
        # Handle Hypothesis objects
        if hasattr(first_item, 'text'):
            return [item.text if hasattr(item, 'text') else str(item) for item in transcribe_result]
        
        # Handle nested lists
        if isinstance(first_item, list):
            return [first_item[0] if first_item else "" for first_item in transcribe_result]
        
        # Handle plain strings
        if isinstance(first_item, str):
            return transcribe_result
    
    return [str(transcribe_result[0]) if transcribe_result else ""]


@app.on_event("startup")
def startup_event():
    logger.info("=== NeMo ASR API Starting ===")
    logger.info(f"Model ID: {MODEL_ID}")
    logger.info(f"Port: {os.getenv('PORT', '8080')}")
    logger.info("Pre-loading model on startup...")
    model_manager.get_model()
    logger.info("=== Startup complete - ready to accept requests ===")


@app.get("/v1/health")
def healthcheck():
    return {"status": "ok"}


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> str:
    """Convert raw PCM data to WAV file."""
    output_fd, output_path = tempfile.mkstemp(suffix=".wav")
    os.close(output_fd)
    
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    
    return output_path


def extract_audio_segment(input_path: str, start_time: Optional[float], end_time: Optional[float]) -> str:
    """Extract audio segment using ffmpeg if start_time or end_time is specified."""
    if start_time is None and end_time is None:
        return input_path
    
    output_fd, output_path = tempfile.mkstemp(suffix=".wav")
    os.close(output_fd)
    
    cmd = ["ffmpeg", "-y", "-i", input_path]
    
    if start_time is not None:
        cmd.extend(["-ss", str(start_time)])
    
    if end_time is not None:
        if start_time is not None:
            duration = end_time - start_time
            cmd.extend(["-t", str(duration)])
        else:
            cmd.extend(["-to", str(end_time)])
    
    cmd.extend(["-ac", "1", "-ar", "16000", output_path])
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        os.remove(output_path)
        raise RuntimeError(f"Failed to extract audio segment: {e.stderr.decode()}")


@app.post("/v1/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    channel_selector: str = Query("average"),
    start_time: Optional[float] = Query(None, description="Start time in seconds"),
    end_time: Optional[float] = Query(None, description="End time in seconds"),
    is_pcm: bool = Query(False, description="Set to true if uploading raw PCM data"),
    sample_rate: int = Query(16000, description="Sample rate in Hz (for PCM only)"),
    channels: int = Query(1, description="Number of audio channels (for PCM only)"),
    sample_width: int = Query(2, description="Sample width in bytes (for PCM only)"),
):
    """Transcribe audio. Supports WAV, MP3, FLAC, OGG, or raw PCM.
    
    For PCM: set is_pcm=true and specify sample_rate, channels, and sample_width.
    """
    logger.info(f"Received transcription request - file: {file.filename}, is_pcm: {is_pcm}")
    if is_pcm:
        logger.info(f"PCM parameters: sample_rate={sample_rate}, channels={channels}, sample_width={sample_width}")
    
    model = model_manager.get_model()
    data = await file.read()
    logger.info(f"Audio data size: {len(data)} bytes")
    
    if is_pcm:
        logger.info("Converting PCM to WAV...")
        tmp_path = pcm_to_wav(data, sample_rate, channels, sample_width)
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        
        logger.info("Normalizing audio to standard PCM WAV format...")
        normalized_fd, normalized_path = tempfile.mkstemp(suffix=".wav")
        os.close(normalized_fd)
        
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", tmp_path,
                    "-acodec", "pcm_s16le",  # 16-bit PCM
                    "-ar", "16000",           # 16kHz sample rate
                    "-ac", "1",               # mono
                    normalized_path
                ],
                check=True,
                capture_output=True
            )
            os.remove(tmp_path)
            tmp_path = normalized_path
            logger.info("Audio normalization complete")
        except subprocess.CalledProcessError as e:
            os.remove(normalized_path)
            logger.error(f"Audio normalization failed: {e.stderr.decode()}")
            raise RuntimeError(f"Failed to normalize audio: {e.stderr.decode()}")

    segmented_path = None
    try:
        if start_time is not None or end_time is not None:
            logger.info(f"Extracting audio segment: {start_time}s to {end_time}s")
            segmented_path = extract_audio_segment(tmp_path, start_time, end_time)
            audio_path = segmented_path
        else:
            audio_path = tmp_path
        
        cs: Union[str, int]
        cs = channel_selector
        if isinstance(cs, str) and cs.isdigit():
            cs = int(cs)

        logger.info(f"Starting transcription with channel_selector={cs}")
        results = model.transcribe(
            [audio_path],
            batch_size=1,
            return_hypotheses=False,
            verbose=False,
            num_workers=0,
            channel_selector=cs,
        )
        texts = extract_texts(results)
        transcript = texts[0] if texts else ""
        logger.info(f"Transcription complete: {len(transcript)} characters")
        return {"text": transcript}
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        raise
    finally:
        try:
            os.remove(tmp_path)
            if segmented_path and os.path.exists(segmented_path):
                os.remove(segmented_path)
        except Exception:
            pass


