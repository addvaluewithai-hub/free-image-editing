from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile

import modal

APP_NAME = "qwen3-tts-06b"
CUSTOM_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
BASE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

MODEL_ROOT = Path("/models")
CUSTOM_MODEL_PATH = MODEL_ROOT / "Qwen3-TTS-12Hz-0.6B-CustomVoice"
BASE_MODEL_PATH = MODEL_ROOT / "Qwen3-TTS-12Hz-0.6B-Base"
HF_CACHE = MODEL_ROOT / "hf-cache"

SUPPORTED_SPEAKERS = {
    "vivian": "Vivian",
    "serena": "Serena",
    "uncle_fu": "Uncle_Fu",
    "dylan": "Dylan",
    "eric": "Eric",
    "ryan": "Ryan",
    "aiden": "Aiden",
    "ono_anna": "Ono_Anna",
    "sohee": "Sohee",
}
SUPPORTED_LANGUAGES = {
    "auto": "Auto",
    "chinese": "Chinese",
    "english": "English",
    "japanese": "Japanese",
    "korean": "Korean",
    "german": "German",
    "french": "French",
    "russian": "Russian",
    "portuguese": "Portuguese",
    "spanish": "Spanish",
    "italian": "Italian",
}
MAX_TEXT_CHARS = 2400
MAX_REFERENCE_BYTES = 25 * 1024 * 1024

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("qwen3-tts-06b-models", create_if_missing=True)


download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface_hub[hf_xet]>=0.34")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1"})
)

# One reproducible CUDA image for both 0.6B checkpoints. FlashAttention 2 is
# enabled for production inference; historical benchmark numbers in README were
# measured on the older SDPA deployment and should be treated as a baseline.
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
            "MAX_JOBS": "4",
        }
    )
    .run_commands("python -m pip install --upgrade pip setuptools wheel packaging ninja")
    .run_commands(
        "pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 "
        "--extra-index-url https://download.pytorch.org/whl/cu128"
    )
    .run_commands(
        "pip install qwen-tts==0.1.1 soundfile==0.13.1 "
        "huggingface_hub[hf_xet]>=0.34"
    )
    .run_commands("MAX_JOBS=4 pip install 'flash-attn>=2.8,<3' --no-build-isolation")
)

api_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "fastapi[standard]", "python-multipart"
)


def _canonical_language(language: str) -> str:
    key = language.strip().lower()
    if key not in SUPPORTED_LANGUAGES:
        raise ValueError(
            "unsupported language; use one of: "
            + ", ".join(SUPPORTED_LANGUAGES.values())
        )
    return SUPPORTED_LANGUAGES[key]


def _canonical_speaker(speaker: str) -> str:
    key = speaker.strip().lower().replace(" ", "_")
    if key not in SUPPORTED_SPEAKERS:
        raise ValueError(
            "unsupported speaker; use one of: "
            + ", ".join(SUPPORTED_SPEAKERS.values())
        )
    return SUPPORTED_SPEAKERS[key]


def _validate_text(text: str) -> str:
    text = text.strip()
    if not text:
        raise ValueError("text cannot be empty")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(
            f"text is too long; maximum is {MAX_TEXT_CHARS} characters. "
            "Chunk long-form content before calling TTS."
        )
    return text


def _wav_bytes(waveform, sample_rate: int) -> bytes:
    import soundfile as sf

    out = BytesIO()
    sf.write(out, waveform, int(sample_rate), format="WAV", subtype="PCM_16")
    return out.getvalue()


@app.function(
    image=download_image,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={str(MODEL_ROOT): model_volume},
    timeout=60 * 60,
    cpu=4,
    memory=8192,
)
def download_models() -> dict[str, str]:
    """Cache the only two checkpoints intentionally supported by this repo."""
    import os
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    downloaded: dict[str, str] = {}
    for repo_id, local_dir in (
        (CUSTOM_MODEL_ID, CUSTOM_MODEL_PATH),
        (BASE_MODEL_ID, BASE_MODEL_PATH),
    ):
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            token=token,
        )
        downloaded[repo_id] = str(local_dir)

    model_volume.commit()
    return downloaded


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
class PresetTTS:
    """Qwen3-TTS 0.6B CustomVoice for built-in speakers such as Ryan/Aiden."""

    @modal.enter()
    def load(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        if not CUSTOM_MODEL_PATH.exists():
            raise RuntimeError("CustomVoice checkpoint missing; run download_models first")

        torch.set_float32_matmul_precision("high")
        self.model = Qwen3TTSModel.from_pretrained(
            str(CUSTOM_MODEL_PATH),
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        print(
            "QWEN06_CUSTOM_LOADED "
            "device=cuda:0 attention=flash_attention_2 dtype=bfloat16"
        )

    @modal.method()
    def synthesize(
        self,
        text: str,
        speaker: str = "Ryan",
        language: str = "English",
    ) -> bytes:
        text = _validate_text(text)
        speaker = _canonical_speaker(speaker)
        language = _canonical_language(language)

        # 0.6B CustomVoice does not provide reliable instruction/style control.
        # Do not add an `instruct` argument here unless the upstream model changes.
        wavs, sample_rate = self.model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
        )
        return _wav_bytes(wavs[0], int(sample_rate))


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
class VoiceCloneTTS:
    """Qwen3-TTS 0.6B Base voice cloning from a short reference clip."""

    @modal.enter()
    def load(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        if not BASE_MODEL_PATH.exists():
            raise RuntimeError("Base checkpoint missing; run download_models first")

        torch.set_float32_matmul_precision("high")
        self.model = Qwen3TTSModel.from_pretrained(
            str(BASE_MODEL_PATH),
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        print(
            "QWEN06_BASE_LOADED "
            "device=cuda:0 attention=flash_attention_2 dtype=bfloat16"
        )

    @modal.method()
    def synthesize(
        self,
        text: str,
        reference_audio: bytes,
        reference_filename: str = "reference.wav",
        reference_text: str | None = None,
        language: str = "English",
        x_vector_only: bool = False,
    ) -> bytes:
        text = _validate_text(text)
        language = _canonical_language(language)

        if not reference_audio:
            raise ValueError("reference audio cannot be empty")
        if len(reference_audio) > MAX_REFERENCE_BYTES:
            raise ValueError("reference audio must be smaller than 25 MB")
        if not x_vector_only and not (reference_text and reference_text.strip()):
            raise ValueError(
                "reference_text is required for high-fidelity cloning; "
                "set x_vector_only=true to clone without a transcript"
            )

        suffix = Path(reference_filename or "reference.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix) as ref:
            ref.write(reference_audio)
            ref.flush()
            wavs, sample_rate = self.model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref.name,
                ref_text=reference_text.strip() if reference_text else None,
                x_vector_only_mode=x_vector_only,
            )

        return _wav_bytes(wavs[0], int(sample_rate))


@app.function(image=api_image, timeout=15 * 60)
@modal.asgi_app(requires_proxy_auth=True)
def api():
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import Response
    from pydantic import BaseModel, Field

    web = FastAPI(title="Qwen3-TTS 0.6B on Modal", version="2.0.0")

    class TTSRequest(BaseModel):
        text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
        speaker: str = "Ryan"
        language: str = "English"

    @web.get("/")
    def root():
        return {
            "status": "ok",
            "app": APP_NAME,
            "preset_model": CUSTOM_MODEL_ID,
            "clone_model": BASE_MODEL_ID,
            "gpu": "L4 on demand",
            "scale_down_seconds": 30,
            "auth": "Modal proxy authentication required",
            "supported_speakers": list(SUPPORTED_SPEAKERS.values()),
            "supported_languages": list(SUPPORTED_LANGUAGES.values()),
            "ssml": False,
            "ipa_input": False,
            "style_instruction_on_0_6b": False,
        }

    @web.post("/tts")
    def tts(req: TTSRequest):
        try:
            data = PresetTTS().synthesize.remote(
                req.text,
                req.speaker,
                req.language,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(content=data, media_type="audio/wav")

    @web.post("/clone")
    def clone(
        reference: UploadFile = File(...),
        text: str = Form(...),
        reference_text: str | None = Form(None),
        language: str = Form("English"),
        x_vector_only: bool = Form(False),
    ):
        try:
            source = reference.file.read()
            data = VoiceCloneTTS().synthesize.remote(
                text,
                source,
                reference.filename or "reference.wav",
                reference_text,
                language,
                x_vector_only,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(content=data, media_type="audio/wav")

    return web
