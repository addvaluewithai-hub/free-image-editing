from __future__ import annotations

from io import BytesIO
from pathlib import Path

import modal

APP_NAME = "qwen3-tts-06b-test"
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
MODEL_ROOT = Path("/models")
MODEL_PATH = MODEL_ROOT / "Qwen3-TTS-12Hz-0.6B-CustomVoice"
HF_CACHE = MODEL_ROOT / "hf-cache"

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("qwen3-tts-models", create_if_missing=True)


download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface_hub[hf_xet]>=0.34")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1"})
)

# Keep Qwen3-TTS isolated from the existing VoiceTut/VoxCPM2 deployments.
# Start with PyTorch SDPA for a reliable baseline; FlashAttention can be benchmarked separately later.
tts_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11"
    )
    .entrypoint([])
    .apt_install("git", "build-essential", "ffmpeg", "libsndfile1", "ninja-build")
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "HF_HOME": str(HF_CACHE),
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .run_commands("python -m pip install --upgrade pip setuptools wheel")
    .run_commands(
        "pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 "
        "--extra-index-url https://download.pytorch.org/whl/cu128"
    )
    .run_commands(
        "pip install qwen-tts==0.1.1 soundfile==0.13.1 "
        "huggingface_hub[hf_xet]>=0.34"
    )
)


@app.function(
    image=download_image,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={str(MODEL_ROOT): model_volume},
    timeout=60 * 60,
    cpu=4,
    memory=8192,
)
def download_model() -> str:
    import os
    from huggingface_hub import snapshot_download

    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_PATH),
        token=os.environ.get("HF_TOKEN"),
    )
    model_volume.commit()
    return str(MODEL_PATH)


@app.cls(
    image=tts_image,
    gpu="L4",
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={str(MODEL_ROOT): model_volume},
    scaledown_window=30,
    max_containers=1,
    timeout=15 * 60,
    memory=16384,
)
class Qwen3TTS:
    @modal.enter()
    def load(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        if not MODEL_PATH.exists():
            raise RuntimeError("Qwen3-TTS weights are missing. Run download_model first.")

        torch.set_float32_matmul_precision("high")
        self.model = Qwen3TTSModel.from_pretrained(
            str(MODEL_PATH),
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        print("QWEN3_TTS_LOADED model=Qwen3-TTS-12Hz-0.6B-CustomVoice device=cuda:0")

    def _to_wav_bytes(self, waveform, sample_rate: int) -> bytes:
        import soundfile as sf

        out = BytesIO()
        sf.write(out, waveform, sample_rate, format="WAV", subtype="PCM_16")
        return out.getvalue()

    @modal.method()
    def synthesize(
        self,
        text: str,
        speaker: str = "Ryan",
        instruct: str | None = None,
        language: str = "English",
    ) -> bytes:
        text = text.strip()
        if not text:
            raise ValueError("text cannot be empty")
        if len(text) > 6000:
            raise ValueError("text is too long; maximum is 6000 characters")

        wavs, sample_rate = self.model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
        )
        return self._to_wav_bytes(wavs[0], int(sample_rate))
