from __future__ import annotations

from io import BytesIO
from pathlib import Path

import modal

APP_NAME = "voxcpm2-english-oneshot-test"
MODEL_ROOT = Path("/models")
MODEL_PATH = MODEL_ROOT / "VoxCPM2"

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("egyptian-voice-chat-models", create_if_missing=True)

tts_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11"
    )
    .entrypoint([])
    .apt_install("git", "build-essential", "ffmpeg", "libsndfile1", "ninja-build")
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .run_commands("python -m pip install --upgrade pip setuptools wheel")
    .run_commands(
        "pip install --index-url https://download.pytorch.org/whl/cu126 "
        "torch==2.7.1 torchaudio==2.7.1"
    )
    .run_commands(
        "pip install voxcpm==2.0.3 soundfile==0.13.1 huggingface_hub[hf_xet]>=0.34"
    )
)


@app.cls(
    image=tts_image,
    gpu="L4",
    volumes={str(MODEL_ROOT): model_volume},
    scaledown_window=30,
    max_containers=1,
    timeout=30 * 60,
    memory=16384,
)
class VoxEnglishOneShot:
    @modal.enter()
    def load(self) -> None:
        from voxcpm import VoxCPM

        if not MODEL_PATH.exists():
            raise RuntimeError("VoxCPM2 weights are missing from the shared model volume.")

        self.model = VoxCPM.from_pretrained(
            str(MODEL_PATH),
            load_denoiser=False,
            local_files_only=True,
            device="cuda",
            optimize=True,
        )
        self.sample_rate = int(self.model.tts_model.sample_rate)
        print(f"VOX_ENGLISH_ONESHOT_LOADED sample_rate={self.sample_rate} optimize=True")

    @modal.method()
    def synthesize(
        self,
        text: str,
        voice_description: str,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
    ) -> bytes:
        import soundfile as sf

        text = text.strip()
        if not text:
            raise ValueError("text cannot be empty")
        if len(text) > 5000:
            raise ValueError("text is too long; maximum is 5000 characters")

        conditioned_text = f"({voice_description.strip()}){text}"
        waveform = self.model.generate(
            text=conditioned_text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=True,
            denoise=False,
        )
        out = BytesIO()
        sf.write(out, waveform, self.sample_rate, format="WAV", subtype="PCM_16")
        return out.getvalue()
