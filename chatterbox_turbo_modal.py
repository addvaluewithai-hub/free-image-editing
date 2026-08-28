from __future__ import annotations

from io import BytesIO
from pathlib import Path

import modal

APP_NAME = "chatterbox-turbo-test"
MODEL_ID = "ResembleAI/chatterbox-turbo"
MODEL_ROOT = Path("/models")
MODEL_PATH = MODEL_ROOT / "chatterbox-turbo"
HF_CACHE = MODEL_ROOT / "hf-cache"
CHATTERBOX_COMMIT = "5de7a54aa4e5e2baadb0182dde554908b48b85c2"

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("chatterbox-turbo-models", create_if_missing=True)


download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface_hub[hf_xet]>=0.34")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1"})
)

# Chatterbox currently pins torch/torchaudio 2.6.0. Use the matching CUDA 12.4
# wheels and pin the official Chatterbox source commit for a reproducible test.
tts_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .entrypoint([])
    .apt_install("git", "ffmpeg", "libsndfile1")
    .env(
        {
            "HF_HOME": str(HF_CACHE),
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .run_commands("python -m pip install --upgrade pip setuptools wheel")
    .run_commands(
        "pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 "
        "--index-url https://download.pytorch.org/whl/cu124"
    )
    .run_commands(
        f"pip install 'git+https://github.com/resemble-ai/chatterbox.git@{CHATTERBOX_COMMIT}'"
    )
    .run_commands("pip install soundfile==0.13.1 'huggingface_hub[hf_xet]>=0.34'")
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
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
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
    memory=12288,
)
class ChatterboxTurbo:
    @modal.enter()
    def load(self) -> None:
        import torch
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        if not MODEL_PATH.exists():
            raise RuntimeError("Chatterbox Turbo weights are missing. Run download_model first.")

        torch.set_float32_matmul_precision("high")
        self.model = ChatterboxTurboTTS.from_local(str(MODEL_PATH), device="cuda")
        print(
            "CHATTERBOX_TURBO_LOADED "
            f"device=cuda sample_rate={self.model.sr} commit={CHATTERBOX_COMMIT}"
        )

    def _to_wav_bytes(self, waveform) -> bytes:
        import soundfile as sf

        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().numpy()
        if getattr(waveform, "ndim", 1) > 1:
            waveform = waveform.squeeze()

        out = BytesIO()
        sf.write(out, waveform, int(self.model.sr), format="WAV", subtype="PCM_16")
        return out.getvalue()

    @modal.method()
    def synthesize(
        self,
        text: str,
        seed: int = 1234,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 1000,
        repetition_penalty: float = 1.2,
    ) -> bytes:
        import torch

        text = text.strip()
        if not text:
            raise ValueError("text cannot be empty")
        if len(text) > 650:
            raise ValueError(
                "Chunk is too long for this production benchmark; keep it at 650 characters or less."
            )

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        wav = self.model.generate(
            text,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        return self._to_wav_bytes(wav)
