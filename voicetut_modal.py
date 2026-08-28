from __future__ import annotations

from io import BytesIO
from pathlib import Path

import modal

APP_NAME = "voicetut-tts-test"
MODEL_ID = "mohammedaly22/VoiceTut-TTS"
MODEL_ROOT = Path("/models")
MODEL_PATH = MODEL_ROOT / "VoiceTut-TTS"
HF_CACHE = MODEL_ROOT / "hf-cache"
OMNIVOICE_COMMIT = "08be0b4ccbac3e13e374e86fbfead4b4cac343e2"
VOICETUT_COMMIT = "b5302e9420cce535ced742c2c1c630a189c2f28f"

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("egyptian-voice-chat-models", create_if_missing=True)


download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface_hub[hf_xet]>=0.34")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1"})
)

# Keep this model isolated from VoxCPM2 so the comparison cannot break the existing API.
# OmniVoice currently documents PyTorch 2.8 + CUDA 12.8 for NVIDIA GPUs.
tts_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.10"
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
        f"pip install git+https://github.com/k2-fsa/OmniVoice.git@{OMNIVOICE_COMMIT}"
    )
    .run_commands(
        f"pip install git+https://github.com/MohammedAly22/VoiceTuT-TTS.git@{VOICETUT_COMMIT}"
    )
    .run_commands("pip install soundfile==0.13.1 huggingface_hub[hf_xet]>=0.34")
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
    memory=12288,
)
class VoiceTut:
    @modal.enter()
    def load(self) -> None:
        from voicetut_tts import VoiceTutTTS

        if not MODEL_PATH.exists():
            raise RuntimeError("VoiceTut-TTS weights are missing. Run download_model first.")

        self.tts = VoiceTutTTS.from_pretrained(
            str(MODEL_PATH),
            device="cuda:0",
            dtype="float16",
            language="arz",
        )
        self.sample_rate = int(self.tts.sampling_rate)
        self.speaker_names = [getattr(s, "name", str(s)) for s in self.tts.list_speakers()]
        print(f"VOICETUT_LOADED sample_rate={self.sample_rate} speakers={self.speaker_names}")

    def _to_wav_bytes(self, waveform) -> bytes:
        import soundfile as sf

        out = BytesIO()
        sf.write(out, waveform, self.sample_rate, format="WAV", subtype="PCM_16")
        return out.getvalue()

    @modal.method()
    def synthesize(
        self,
        text: str,
        speaker: str = "Mohamed",
        num_step: int = 32,
        guidance_scale: float = 2.0,
        speed: float = 1.0,
        normalize: bool = True,
    ) -> bytes:
        text = text.strip()
        if not text:
            raise ValueError("text cannot be empty")
        if len(text) > 4000:
            raise ValueError("text is too long; maximum is 4000 characters")
        if not (8 <= num_step <= 64):
            raise ValueError("num_step must be between 8 and 64")
        if not (0.5 <= guidance_scale <= 4.0):
            raise ValueError("guidance_scale must be between 0.5 and 4.0")
        if not (0.6 <= speed <= 1.5):
            raise ValueError("speed must be between 0.6 and 1.5")

        waveform = self.tts.synthesize(
            text,
            speaker=speaker,
            language="arz",
            normalize=normalize,
            num_step=num_step,
            guidance_scale=guidance_scale,
            speed=speed,
        )
        return self._to_wav_bytes(waveform)

    @modal.method()
    def speakers(self) -> list[str]:
        return self.speaker_names
