import asyncio
import json
import logging
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass, asdict
from typing import List, Optional, Union

import numpy as np

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
os.environ["TQDM_DISABLE"] = "0"

from fastapi import FastAPI, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import nemo.collections.asr as nemo_asr
from omegaconf import open_dict
import torch

# ============================================================================
# CUDA/PyTorch Optimization Flags
# ============================================================================
# Enable cuDNN autotuner to find the best algorithm for the hardware
torch.backends.cudnn.benchmark = True

# Enable TF32 for faster matmul on Ampere+ GPUs (H100, A100, RTX 30xx+)
# TF32 uses 19-bit precision which is faster than FP32 with minimal accuracy loss
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Disable debug/profiling features for production
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)

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

MODEL_ID = os.getenv("MODEL_ID")
if not MODEL_ID:
    raise RuntimeError("MODEL_ID environment variable must be set to a valid NeMo ASR model.")
EOU_TOKEN = "<EOU>"
EOB_TOKEN = "<EOB>"

# Streaming ASR configuration
SAMPLE_RATE = 16000
CHUNK_SIZE_IN_SECS = 0.08  # 80ms chunks for FastConformer
ATT_CONTEXT_SIZE = [70, 1]  # Left and right attention context
LOG_MEL_ZERO = -16.635  # Log-Mel spectrogram value for zero signals


@dataclass
class ASRResult:
    """Result from streaming ASR transcription."""
    text: str
    is_final: bool
    eou_prob: Optional[float] = None
    eob_prob: Optional[float] = None
    eou_latency: Optional[float] = None
    eob_latency: Optional[float] = None
    processing_time: Optional[float] = None

    def to_dict(self):
        return asdict(self)


class AudioBufferer:
    """Simple audio buffer for accumulating samples."""
    
    def __init__(self, sample_rate: int, buffer_size_in_secs: float, device: str = "cuda"):
        self.buffer_size = int(buffer_size_in_secs * sample_rate)
        self.device = device
        self.sample_buffer = torch.zeros(self.buffer_size, dtype=torch.float32, device=device)
    
    def reset(self) -> None:
        self.sample_buffer.zero_()
    
    def update(self, audio: np.ndarray) -> None:
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio).to(self.device)
        audio_size = audio.shape[0]
        if audio_size > self.buffer_size:
            raise ValueError(f"Frame size ({audio_size}) exceeds buffer size ({self.buffer_size})")
        shift = audio_size
        self.sample_buffer[:-shift] = self.sample_buffer[shift:].clone()
        self.sample_buffer[-shift:] = audio.clone()
    
    def get_buffer(self) -> torch.Tensor:
        return self.sample_buffer.clone()
    
    def is_buffer_empty(self) -> bool:
        return self.sample_buffer.sum() == 0


class CacheFeatureBufferer:
    """Feature buffer with mel-spectrogram preprocessing for streaming ASR."""
    
    def __init__(
        self,
        sample_rate: int,
        buffer_size_in_secs: float,
        chunk_size_in_secs: float,
        preprocessor_cfg,
        device: str,
        fill_value: float = LOG_MEL_ZERO,
    ):
        if buffer_size_in_secs < chunk_size_in_secs:
            raise ValueError(
                f"Buffer size ({buffer_size_in_secs}s) should be no less than chunk size ({chunk_size_in_secs}s)"
            )
        self.sample_rate = sample_rate
        self.buffer_size_in_secs = buffer_size_in_secs
        self.chunk_size_in_secs = chunk_size_in_secs
        self.device = device
        
        if hasattr(preprocessor_cfg, 'log') and preprocessor_cfg.log:
            self.ZERO_LEVEL_SPEC_DB_VAL = LOG_MEL_ZERO
        else:
            self.ZERO_LEVEL_SPEC_DB_VAL = fill_value
        
        self.n_feat = preprocessor_cfg.features
        self.timestep_duration = preprocessor_cfg.window_stride
        self.n_chunk_look_back = int(self.timestep_duration * self.sample_rate)
        self.chunk_size = int(self.chunk_size_in_secs * self.sample_rate)
        self.sample_buffer = AudioBufferer(sample_rate, buffer_size_in_secs, device)
        self.feature_buffer_len = int(buffer_size_in_secs / self.timestep_duration)
        self.feature_chunk_len = int(chunk_size_in_secs / self.timestep_duration)
        self.feature_buffer = torch.full(
            [self.n_feat, self.feature_buffer_len],
            self.ZERO_LEVEL_SPEC_DB_VAL,
            dtype=torch.float32,
            device=self.device,
        )
        self.preprocessor = nemo_asr.models.ASRModel.from_config_dict(preprocessor_cfg)
        self.preprocessor.to(self.device)
    
    def is_buffer_empty(self) -> bool:
        return self.sample_buffer.is_buffer_empty()
    
    def reset(self) -> None:
        self.sample_buffer.reset()
        self.feature_buffer.fill_(self.ZERO_LEVEL_SPEC_DB_VAL)
    
    def _update_feature_buffer(self, feat_chunk: torch.Tensor) -> None:
        self.feature_buffer[:, :-self.feature_chunk_len] = self.feature_buffer[:, self.feature_chunk_len:].clone()
        self.feature_buffer[:, -self.feature_chunk_len:] = feat_chunk.clone()
    
    def preprocess(self, audio_signal: torch.Tensor) -> torch.Tensor:
        audio_signal = audio_signal.unsqueeze_(0).to(self.device)
        audio_signal_len = torch.tensor([audio_signal.shape[1]], device=self.device)
        features, _ = self.preprocessor(input_signal=audio_signal, length=audio_signal_len)
        features = features.squeeze()
        return features
    
    def update(self, audio: np.ndarray) -> None:
        self.sample_buffer.update(audio)
        if math.isclose(self.buffer_size_in_secs, self.chunk_size_in_secs):
            samples = self.sample_buffer.sample_buffer.clone()
        else:
            samples = self.sample_buffer.sample_buffer[-(self.n_chunk_look_back + self.chunk_size):]
        features = self.preprocess(samples)
        if (diff := features.shape[1] - self.feature_chunk_len - 1) > 0:
            features = features[:, :-diff]
        self._update_feature_buffer(features[:, -self.feature_chunk_len:])
    
    def get_feature_buffer(self) -> torch.Tensor:
        return self.feature_buffer.clone()


class StreamingASRManager:
    """Manages streaming ASR with cache-aware encoding and EOU detection."""
    
    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
        self.eou_string = EOU_TOKEN
        self.eob_string = EOB_TOKEN
        self.att_context_size = ATT_CONTEXT_SIZE
        self.chunk_size_in_secs = CHUNK_SIZE_IN_SECS
        
        # Get tokenizer and blank ID
        self.tokenizer = self.model.tokenizer
        self.blank_id = len(self.tokenizer.vocab)
        
        # Determine decoder type
        if isinstance(self.model, nemo_asr.models.EncDecCTCModel):
            self.decoder_type = "ctc"
        elif isinstance(self.model, nemo_asr.models.EncDecRNNTModel):
            self.decoder_type = "rnnt"
        else:
            self.decoder_type = "rnnt"  # Default to RNNT for hybrid models
        
        # Configure model for streaming
        self._configure_streaming()
        
        # Initialize buffers and caches
        self._init_buffer()
        self._reset_cache()
        self._previous_hypotheses = self._get_blank_hypothesis()
        self._last_transcript_timestamp = time.time()
        
        logger.info(f"StreamingASRManager initialized with decoder_type={self.decoder_type}")
    
    def _configure_streaming(self):
        """Configure model for cache-aware streaming."""
        # Set attention context size
        if hasattr(self.model.encoder, "set_default_att_context_size"):
            self.model.encoder.set_default_att_context_size(att_context_size=self.att_context_size)
        
        # Configure decoding strategy
        decoding_cfg = self.model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "greedy"
            decoding_cfg.compute_timestamps = False
            decoding_cfg.preserve_alignments = True
            if hasattr(self.model, 'joint'):
                decoding_cfg.greedy.max_symbols = 10
                decoding_cfg.fused_batch_size = -1
        self.model.change_decoding_strategy(decoding_cfg)
        
        # Get streaming parameters from model
        window_stride_in_secs = self.model.cfg.preprocessor.window_stride
        model_stride = self.model.cfg.encoder.subsampling_factor
        
        self.model_chunk_size = self.model.encoder.streaming_cfg.chunk_size
        if isinstance(self.model_chunk_size, list):
            self.model_chunk_size = self.model_chunk_size[1]
        
        self.pre_encode_cache_size = self.model.encoder.streaming_cfg.pre_encode_cache_size
        if isinstance(self.pre_encode_cache_size, list):
            self.pre_encode_cache_size = self.pre_encode_cache_size[1]
        
        self.pre_encode_cache_size_in_secs = self.pre_encode_cache_size * window_stride_in_secs
        self.tokens_per_frame = math.ceil(np.trunc(self.chunk_size_in_secs / window_stride_in_secs) / model_stride)
        
        # Setup encoder streaming params
        self.model.encoder.setup_streaming_params(
            chunk_size=self.model_chunk_size // model_stride,
            shift_size=self.tokens_per_frame
        )
        
        model_chunk_size_in_secs = self.model_chunk_size * window_stride_in_secs
        self.buffer_size_in_secs = self.pre_encode_cache_size_in_secs + model_chunk_size_in_secs
    
    def _init_buffer(self):
        """Initialize the audio/feature buffer."""
        self._audio_buffer = CacheFeatureBufferer(
            sample_rate=SAMPLE_RATE,
            buffer_size_in_secs=self.buffer_size_in_secs,
            chunk_size_in_secs=self.chunk_size_in_secs,
            preprocessor_cfg=self.model.cfg.preprocessor,
            device=self.device,
        )
    
    def _reset_cache(self):
        """Reset encoder cache."""
        (
            self._cache_last_channel,
            self._cache_last_time,
            self._cache_last_channel_len,
        ) = self.model.encoder.get_initial_cache_state(1)  # batch size 1
    
    def _get_blank_hypothesis(self):
        """Get a blank hypothesis for initialization."""
        from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
        blank_hypothesis = Hypothesis(score=0.0, y_sequence=[], dec_state=None, timestamp=[], last_token=None)
        return [blank_hypothesis]
    
    @property
    def drop_extra_pre_encoded(self):
        return self.model.encoder.streaming_cfg.drop_extra_pre_encoded
    
    def get_text_from_tokens(self, tokens: List[int]) -> str:
        """Convert token IDs to text."""
        sep = "\u2581"  # SentencePiece separator
        tokens = [int(t) for t in tokens if t != self.blank_id]
        if tokens:
            pieces = self.tokenizer.ids_to_tokens(tokens)
            text = "".join([p.replace(sep, ' ') if p.startswith(sep) else p for p in pieces])
        else:
            text = ""
        return text
    
    def _get_best_hypothesis(self, encoded, encoded_len, partial_hypotheses=None):
        """Get best hypothesis from decoder."""
        if self.decoder_type == "ctc":
            best_hyp = self.model.decoding.ctc_decoder_predictions_tensor(
                encoded, encoded_len, return_hypotheses=True,
            )
        elif self.decoder_type == "rnnt":
            best_hyp = self.model.decoding.rnnt_decoder_predictions_tensor(
                encoded, encoded_len, return_hypotheses=True, partial_hypotheses=partial_hypotheses
            )
        else:
            raise ValueError(f"Decoder type {self.decoder_type} not supported.")
        return best_hyp
    
    def _get_tokens_and_probs_from_alignments(self, alignments):
        """Extract tokens and their probabilities from alignments."""
        tokens = []
        probs = []
        
        if self.decoder_type == "ctc":
            all_logits = alignments[0]
            all_tokens = alignments[1]
            for i in range(len(all_tokens)):
                token_id = int(all_tokens[i])
                if token_id != self.blank_id:
                    tokens.append(token_id)
                    logits = all_logits[i]
                    probs_i = torch.softmax(logits, dim=-1)[token_id].item()
                    probs.append(probs_i)
        elif self.decoder_type == "rnnt":
            for t in range(len(alignments)):
                for u in range(len(alignments[t])):
                    logits, token_id = alignments[t][u]
                    token_id = int(token_id)
                    if token_id != self.blank_id:
                        tokens.append(token_id)
                        probs_i = torch.softmax(logits, dim=-1)[token_id].item()
                        probs.append(probs_i)
        
        return tokens, probs
    
    def get_eou_probability(self, tokens: List[int], probs: List[float], eou_string: str) -> Optional[float]:
        """Get the probability of the EOU token."""
        try:
            text_tokens = self.tokenizer.ids_to_tokens(tokens)
            if eou_string in text_tokens:
                eou_index = text_tokens.index(eou_string)
                return probs[eou_index]
        except (ValueError, IndexError):
            pass
        return None
    
    def transcribe(self, audio: bytes) -> ASRResult:
        """Process audio chunk and return transcription result."""
        start_time = time.time()
        
        # Convert bytes to numpy array (16-bit PCM)
        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        self._audio_buffer.update(audio_array)
        
        features = self._audio_buffer.get_feature_buffer()
        feature_lengths = torch.tensor([features.shape[1]], device=self.device)
        features = features.unsqueeze(0)  # Add batch dimension
        
        # Use AMP (Automatic Mixed Precision) for faster inference on supported GPUs
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device == "cuda"):
            (
                encoded,
                encoded_len,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
            ) = self.model.encoder.cache_aware_stream_step(
                processed_signal=features,
                processed_signal_length=feature_lengths,
                cache_last_channel=self._cache_last_channel,
                cache_last_time=self._cache_last_time,
                cache_last_channel_len=self._cache_last_channel_len,
                keep_all_outputs=False,
                drop_extra_pre_encoded=self.drop_extra_pre_encoded,
            )
            
            best_hyp = self._get_best_hypothesis(encoded, encoded_len, partial_hypotheses=self._previous_hypotheses)
            
            self._previous_hypotheses = best_hyp
            self._cache_last_channel = cache_last_channel
            self._cache_last_time = cache_last_time
            self._cache_last_channel_len = cache_last_channel_len
        
        tokens, probs = self._get_tokens_and_probs_from_alignments(best_hyp[0].alignments)
        text = self.get_text_from_tokens(tokens)
        
        is_final = False
        eou_latency = None
        eob_latency = None
        eou_prob = None
        eob_prob = None
        current_timestamp = time.time()
        
        # Check for EOU/EOB tokens
        if self.eou_string in text or self.eob_string in text:
            is_final = True
            
            if self.eou_string in text:
                eou_latency = (
                    current_timestamp - self._last_transcript_timestamp
                    if text.strip() == self.eou_string else 0.0
                )
                eou_prob = self.get_eou_probability(tokens, probs, self.eou_string)
            
            if self.eob_string in text:
                eob_latency = (
                    current_timestamp - self._last_transcript_timestamp
                    if text.strip() == self.eob_string else 0.0
                )
                eob_prob = self.get_eou_probability(tokens, probs, self.eob_string)
            
            # Reset state after final transcription
            self.reset_state()
        
        if text.strip():
            self._last_transcript_timestamp = current_timestamp
        
        processing_time = time.time() - start_time
        
        return ASRResult(
            text=text,
            is_final=is_final,
            eou_latency=eou_latency,
            eob_latency=eob_latency,
            eou_prob=eou_prob,
            eob_prob=eob_prob,
            processing_time=processing_time,
        )
    
    def reset_state(self):
        """Reset all state for a new utterance."""
        self._audio_buffer.reset()
        self._reset_cache()
        self._previous_hypotheses = self._get_blank_hypothesis()
        self._last_transcript_timestamp = time.time()


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
        import sys
        
        # Use print with flush for immediate visibility in container logs
        print(f"[MODEL] Starting to load: {MODEL_ID}", flush=True)
        print(f"[MODEL] Fetching from Hugging Face (this may take a few minutes on first run)...", flush=True)
        
        start_time = time.time()
        
        try:
            # Log progress every 30 seconds during download
            model = nemo_asr.models.ASRModel.from_pretrained(MODEL_ID)
            download_time = time.time() - start_time
            print(f"[MODEL] Download/cache complete (took {download_time:.2f}s)", flush=True)
        except Exception as e:
            print(f"[MODEL] ERROR: Failed to load model: {e}", flush=True)
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[MODEL] CUDA available - GPU: {gpu_name}", flush=True)
            gpu_start = time.time()
            model = model.cuda()
            print(f"[MODEL] Model moved to GPU (took {time.time() - gpu_start:.2f}s)", flush=True)
        else:
            print("[MODEL] WARNING: CUDA not available - using CPU (will be VERY slow)", flush=True)
        
        model.eval()
        self._model = model
        total_time = time.time() - start_time
        print(f"[MODEL] Ready for inference (total startup: {total_time:.2f}s)", flush=True)
        sys.stdout.flush()


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


def sanitize_transcript(text: str) -> str:
    """Remove special tokens like <EOU> and <EOB> that are emitted by streaming models."""
    if not text:
        return text
    cleaned = text.replace(EOU_TOKEN, "").replace(EOB_TOKEN, "")
    return cleaned.strip()


@app.on_event("startup")
def startup_event():
    print("=" * 60, flush=True)
    print("=== NeMo ASR API Starting ===", flush=True)
    print(f"Model ID: {MODEL_ID}", flush=True)
    print(f"Port: {os.getenv('PORT', '8080')}", flush=True)
    print("=" * 60, flush=True)
    print("Pre-loading model on startup...", flush=True)
    model_manager.get_model()
    print("=" * 60, flush=True)
    print("=== Startup complete - ready to accept requests ===", flush=True)
    print("=" * 60, flush=True)


@app.get("/v1/health")
def healthcheck():
    return {"status": "ok"}


# Store active streaming sessions
streaming_sessions: dict[str, StreamingASRManager] = {}


def is_streaming_model() -> bool:
    """Check if the loaded model supports streaming with EOU detection."""
    model = model_manager.get_model()
    # Check if model has streaming capabilities
    if not hasattr(model, 'encoder') or not hasattr(model.encoder, 'streaming_cfg'):
        return False
    if not hasattr(model.encoder, 'cache_aware_stream_step'):
        return False
    return True


def get_or_create_streaming_session(session_id: str) -> StreamingASRManager:
    """Get existing streaming session or create a new one."""
    if session_id not in streaming_sessions:
        model = model_manager.get_model()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        streaming_sessions[session_id] = StreamingASRManager(model, device)
        logger.info(f"Created new streaming session: {session_id}")
    return streaming_sessions[session_id]


def remove_streaming_session(session_id: str):
    """Remove a streaming session."""
    if session_id in streaming_sessions:
        del streaming_sessions[session_id]
        logger.info(f"Removed streaming session: {session_id}")


@app.websocket("/v1/stream")
async def stream_transcribe(websocket: WebSocket):
    """WebSocket endpoint for streaming ASR with EOU detection.
    
    Protocol:
    - Client sends binary audio frames (16-bit PCM, 16kHz, mono)
    - Server responds with JSON messages containing transcription results
    
    Message format (server -> client):
    {
        "type": "transcript",
        "text": "transcribed text",
        "is_final": false,
        "eou_detected": false,
        "eou_prob": null,
        "processing_time": 0.05
    }
    
    Control messages (client -> server):
    - {"type": "reset"} - Reset the streaming state for a new utterance
    - {"type": "config", "sample_rate": 16000} - Configure stream parameters
    """
    await websocket.accept()
    
    # Generate unique session ID
    session_id = f"ws_{id(websocket)}_{time.time()}"
    logger.info(f"WebSocket connected: {session_id}")
    
    # Check if model supports streaming
    if not is_streaming_model():
        await websocket.send_json({
            "type": "error",
            "message": f"Model {MODEL_ID} does not support streaming. Use /v1/transcribe for batch transcription."
        })
        await websocket.close()
        return
    
    streaming_manager = None
    
    try:
        streaming_manager = get_or_create_streaming_session(session_id)
        
        # Send ready message
        await websocket.send_json({
            "type": "ready",
            "session_id": session_id,
            "model": MODEL_ID,
            "sample_rate": SAMPLE_RATE,
            "chunk_size_ms": int(CHUNK_SIZE_IN_SECS * 1000),
        })
        
        while True:
            # Receive message (can be binary audio or JSON control message)
            message = await websocket.receive()
            
            if "bytes" in message:
                # Binary audio data
                audio_data = message["bytes"]
                
                if len(audio_data) == 0:
                    continue
                
                try:
                    result = streaming_manager.transcribe(audio_data)
                    
                    # Clean the text for response
                    clean_text = sanitize_transcript(result.text)
                    
                    response = {
                        "type": "transcript",
                        "text": clean_text,
                        "is_final": result.is_final,
                        "eou_detected": EOU_TOKEN in result.text if result.text else False,
                        "eob_detected": EOB_TOKEN in result.text if result.text else False,
                        "eou_prob": result.eou_prob,
                        "eob_prob": result.eob_prob,
                        "eou_latency": result.eou_latency,
                        "eob_latency": result.eob_latency,
                        "processing_time": result.processing_time,
                    }
                    
                    await websocket.send_json(response)
                    
                    # Log EOU events
                    if result.is_final:
                        logger.info(f"[{session_id}] EOU detected: '{clean_text}' (prob: {result.eou_prob})")
                    
                except Exception as e:
                    logger.error(f"[{session_id}] Transcription error: {e}", exc_info=True)
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
            
            elif "text" in message:
                # JSON control message
                try:
                    control = json.loads(message["text"])
                    msg_type = control.get("type")
                    
                    if msg_type == "reset":
                        # Reset state for new utterance
                        streaming_manager.reset_state()
                        await websocket.send_json({
                            "type": "reset_ack",
                            "message": "State reset successfully"
                        })
                        logger.info(f"[{session_id}] State reset by client request")
                    
                    elif msg_type == "ping":
                        await websocket.send_json({"type": "pong"})
                    
                    elif msg_type == "close":
                        logger.info(f"[{session_id}] Close requested by client")
                        break
                    
                    else:
                        logger.warning(f"[{session_id}] Unknown control message type: {msg_type}")
                
                except json.JSONDecodeError:
                    logger.warning(f"[{session_id}] Invalid JSON control message")
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        remove_streaming_session(session_id)
        try:
            await websocket.close()
        except Exception:
            pass


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
        
        cs: Union[str, int, None]
        cs = channel_selector
        if isinstance(cs, str) and cs.isdigit():
            cs = int(cs)

        # Check if model is a streaming model (has streaming_cfg) - these use Lhotse
        # and don't support channel_selector in the same way
        is_streaming_model = hasattr(model, 'encoder') and hasattr(model.encoder, 'streaming_cfg')
        
        logger.info(f"Starting transcription with channel_selector={cs}, is_streaming={is_streaming_model}")
        
        transcribe_kwargs = {
            "audio": [audio_path],
            "batch_size": 1,
            "return_hypotheses": False,
            "verbose": False,
            "num_workers": 0,
        }
        
        # Only pass channel_selector for non-streaming models
        if not is_streaming_model:
            transcribe_kwargs["channel_selector"] = cs
        
        results = model.transcribe(**transcribe_kwargs)
        texts = extract_texts(results)
        transcript = sanitize_transcript(texts[0] if texts else "")
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


