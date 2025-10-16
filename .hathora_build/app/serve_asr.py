import os
import subprocess
import tempfile
from typing import List, Optional, Union

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import nemo.collections.asr as nemo_asr
import torch


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://(hathora-voice|hathora-voice-.*|hathora-voice-.*-hathora)\.vercel\.app",
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
_model = None


def load_model():
    global _model
    if _model is None:
        model = nemo_asr.models.ASRModel.from_pretrained(MODEL_ID)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        _model = model


def extract_texts(transcribe_result: Union[List[str], List[List[str]]]) -> List[str]:
    if not transcribe_result:
        return [""]
    first_item = transcribe_result[0]
    if isinstance(first_item, list):
        return [first_item[0] if first_item else ""]
    return transcribe_result


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/v1/health")
def healthcheck():
    return {"status": "ok"}


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
):
    load_model()
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    segmented_path = None
    try:
        if start_time is not None or end_time is not None:
            segmented_path = extract_audio_segment(tmp_path, start_time, end_time)
            audio_path = segmented_path
        else:
            audio_path = tmp_path
        
        cs: Union[str, int]
        cs = channel_selector
        if isinstance(cs, str) and cs.isdigit():
            cs = int(cs)

        results = _model.transcribe(
            [audio_path],
            batch_size=1,
            return_hypotheses=False,
            verbose=False,
            num_workers=0,
            channel_selector=cs,
        )
        texts = extract_texts(results)
        return {"text": texts[0] if texts else ""}
    finally:
        try:
            os.remove(tmp_path)
            if segmented_path and os.path.exists(segmented_path):
                os.remove(segmented_path)
        except Exception:
            pass


