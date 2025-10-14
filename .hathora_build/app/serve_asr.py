import os
import tempfile
from typing import List, Union

from fastapi import FastAPI, File, UploadFile
import nemo.collections.asr as nemo_asr
import torch


app = FastAPI()

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


@app.post("/v1/transcribe")
async def transcribe(file: UploadFile = File(...), channel_selector: str = "average"):
    load_model()
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        cs: Union[str, int]
        cs = channel_selector
        if isinstance(cs, str) and cs.isdigit():
            cs = int(cs)

        results = _model.transcribe(
            [tmp_path],
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
        except Exception:
            pass


