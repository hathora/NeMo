# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os
import tempfile
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
from pytriton.decorators import batch
from pytriton.model_config import Tensor

import nemo.collections.asr as nemo_asr
from nemo.deploy import ITritonDeployable
from nemo.deploy.utils import cast_output


class ASRDeploy(ITritonDeployable):
    """Triton-compatible deployable for NeMo ASR models.

    Inputs:
      - audio: bytes (WAV/FLAC/OGG/MP3). Batch of audio files as bytes.
      - sample_rate: optional int per item. If provided, will be used when raw PCM data is supplied.

    Outputs:
      - text: bytes. Transcriptions per input.
    """

    def __init__(
        self,
        hf_model_id: Optional[str] = None,
        nemo_checkpoint_path: Optional[str] = None,
        device: str = "cuda",
    ) -> None:
        if not hf_model_id and not nemo_checkpoint_path:
            raise ValueError("Either hf_model_id or nemo_checkpoint_path must be provided")

        if hf_model_id:
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=hf_model_id)
        else:
            self.model = nemo_asr.models.ASRModel.restore_from(restore_path=nemo_checkpoint_path)

        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

    @property
    def get_triton_input(self):
        return (
            Tensor(name="audio", shape=(-1,), dtype=bytes),
            Tensor(name="sample_rate", shape=(-1,), dtype=np.int_, optional=True),
        )

    @property
    def get_triton_output(self):
        return (
            Tensor(name="text", shape=(-1,), dtype=bytes),
        )

    @staticmethod
    def _write_audio_to_tmp(data: bytes, sample_rate: Optional[int]) -> str:
        """Write audio bytes to a temporary WAV file and return the path."""
        # Try to decode with soundfile; if it fails and sample_rate provided, assume raw PCM float32 mono
        try:
            buf = io.BytesIO(data)
            audio, sr = sf.read(buf, dtype="float32", always_2d=False)
        except Exception:
            if sample_rate is None:
                raise
            # Interpret as raw float32 mono
            audio = np.frombuffer(data, dtype=np.float32)
            sr = int(sample_rate)

        # Ensure mono
        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            # convert to mono by averaging channels
            audio = audio.mean(axis=1)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        sf.write(tmp_path, audio, sr)
        return tmp_path

    @batch
    def triton_infer_fn(self, **inputs: np.ndarray):
        audio_list: List[bytes] = [bytes(b) for b in inputs["audio"].tolist()]
        sr_list: Optional[np.ndarray] = inputs.get("sample_rate", None)
        if sr_list is not None:
            sr_list = sr_list.reshape(-1)

        tmp_files: List[str] = []
        try:
            for i, audio_bytes in enumerate(audio_list):
                sr = int(sr_list[i]) if sr_list is not None and len(sr_list) > i else None
                tmp_files.append(self._write_audio_to_tmp(audio_bytes, sr))

            transcripts = self.model.transcribe(tmp_files)
            # transcribe can return List[str] or List[List[str]] depending on model; flatten if needed
            if len(transcripts) > 0 and isinstance(transcripts[0], list):
                transcripts = [t[0] if t else "" for t in transcripts]

            return {"text": cast_output(transcripts, np.bytes_)}
        finally:
            for p in tmp_files:
                try:
                    os.remove(p)
                except Exception:
                    pass


