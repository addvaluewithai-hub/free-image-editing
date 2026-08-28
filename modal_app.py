from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile

import modal

APP_NAME = "egyptian-voice-chat"
VOX_MODEL_ID = "openbmb/VoxCPM2"
QWEN_MODEL_ID = "theBOrg32/Egyptian_qwen_3.5_4B"
MODEL_ROOT = Path("/models")
VOX_MODEL_PATH = MODEL_ROOT / "VoxCPM2"
QWEN_MODEL_PATH = MODEL_ROOT / "Egyptian_qwen_3.5_4B"

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("egyptian-voice-chat-models", create_if_missing=True)

# Small CPU image used only to cache Hugging Face snapshots into a persistent Modal Volume.
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface_hub[hf_xet]>=0.34")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1"})
)

# VoxCPM2 needs PyTorch >=2.5 and CUDA >=12.0. A devel image is used because
# torch.compile/Triton may JIT-compile CUDA kernels on first use.
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

qwen_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-runtime-ubuntu22.04", add_python="3.11"
    )
    .entrypoint([])
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1"})
    .run_commands("python -m pip install --upgrade pip setuptools wheel")
    .run_commands(
        "pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.1"
    )
    .run_commands(
        "pip install transformers==5.5.0 accelerate==1.13.0 safetensors==0.8.0 "
        "huggingface_hub[hf_xet]>=0.34 sentencepiece==0.2.1"
    )
)

api_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "fastapi[standard]", "python-multipart"
)


@app.function(
    image=download_image,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={str(MODEL_ROOT): model_volume},
    timeout=60 * 60,
    cpu=4,
    memory=8192,
)
def download_models() -> dict[str, str]:
    import os
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    downloaded: dict[str, str] = {}
    for repo_id, path in (
        (VOX_MODEL_ID, VOX_MODEL_PATH),
        (QWEN_MODEL_ID, QWEN_MODEL_PATH),
    ):
        path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(path),
            token=token,
        )
        downloaded[repo_id] = str(path)
    model_volume.commit()
    return downloaded


@app.cls(
    image=tts_image,
    gpu="L4",
    volumes={str(MODEL_ROOT): model_volume},
    scaledown_window=30,
    max_containers=1,
    timeout=15 * 60,
    memory=16384,
)
class VoxTTS:
    @modal.enter()
    def load(self) -> None:
        from voxcpm import VoxCPM

        if not VOX_MODEL_PATH.exists():
            raise RuntimeError("VoxCPM2 weights are missing. Run download_models first.")

        self.model = VoxCPM.from_pretrained(
            str(VOX_MODEL_PATH),
            load_denoiser=False,
            local_files_only=True,
            device="cuda",
            optimize=True,
        )
        self.sample_rate = int(self.model.tts_model.sample_rate)

    def _to_wav_bytes(self, waveform) -> bytes:
        import soundfile as sf

        out = BytesIO()
        sf.write(out, waveform, self.sample_rate, format="WAV", subtype="PCM_16")
        return out.getvalue()

    @modal.method()
    def synthesize(
        self,
        text: str,
        voice_description: str | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        normalize: bool = True,
    ) -> bytes:
        text = text.strip()
        if not text:
            raise ValueError("text cannot be empty")
        if len(text) > 4000:
            raise ValueError("text is too long; maximum is 4000 characters")
        if not (1.0 <= cfg_value <= 3.0):
            raise ValueError("cfg_value must be between 1.0 and 3.0")
        if not (4 <= inference_timesteps <= 30):
            raise ValueError("inference_timesteps must be between 4 and 30")

        if voice_description and voice_description.strip():
            text = f"({voice_description.strip()}){text}"

        waveform = self.model.generate(
            text=text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            denoise=False,
        )
        return self._to_wav_bytes(waveform)

    @modal.method()
    def clone(
        self,
        reference_audio: bytes,
        filename: str,
        text: str,
        voice_description: str | None = None,
        prompt_text: str | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
    ) -> bytes:
        text = text.strip()
        if not text:
            raise ValueError("text cannot be empty")
        if not reference_audio:
            raise ValueError("reference audio cannot be empty")
        if len(reference_audio) > 25 * 1024 * 1024:
            raise ValueError("reference audio must be smaller than 25 MB")
        if not (1.0 <= cfg_value <= 3.0):
            raise ValueError("cfg_value must be between 1.0 and 3.0")
        if not (4 <= inference_timesteps <= 30):
            raise ValueError("inference_timesteps must be between 4 and 30")

        if voice_description and voice_description.strip():
            text = f"({voice_description.strip()}){text}"

        suffix = Path(filename or "reference.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix) as ref:
            ref.write(reference_audio)
            ref.flush()
            kwargs = {
                "text": text,
                "reference_wav_path": ref.name,
                "cfg_value": cfg_value,
                "inference_timesteps": inference_timesteps,
                "normalize": True,
                "denoise": False,
            }
            if prompt_text and prompt_text.strip():
                kwargs["prompt_wav_path"] = ref.name
                kwargs["prompt_text"] = prompt_text.strip()
            waveform = self.model.generate(**kwargs)
        return self._to_wav_bytes(waveform)


@app.cls(
    image=qwen_image,
    gpu="L4",
    volumes={str(MODEL_ROOT): model_volume},
    scaledown_window=30,
    max_containers=1,
    timeout=10 * 60,
    memory=24576,
)
class EgyptianQwen:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not QWEN_MODEL_PATH.exists():
            raise RuntimeError("Egyptian Qwen weights are missing. Run download_models first.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(QWEN_MODEL_PATH),
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(QWEN_MODEL_PATH),
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map={"": "cuda"},
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    @modal.method()
    def chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> str:
        import torch

        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt cannot be empty")
        if len(prompt) > 12000:
            raise ValueError("prompt is too long; maximum is 12000 characters")
        if not (1 <= max_new_tokens <= 2048):
            raise ValueError("max_new_tokens must be between 1 and 2048")
        if not (0.0 <= temperature <= 2.0):
            raise ValueError("temperature must be between 0 and 2")
        if not (0.05 <= top_p <= 1.0):
            raise ValueError("top_p must be between 0.05 and 1.0")

        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        else:
            messages.append(
                {
                    "role": "system",
                    "content": "أنت مساعد ذكي بتتكلم بالمصري الطبيعي. جاوب بوضوح ومن غير فصحى زيادة إلا لو المستخدم طلب.",
                }
            )
        messages.append({"role": "user", "content": prompt})

        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt").to("cuda")
        do_sample = temperature > 0
        generate_kwargs = {
            "**inputs": None,
        }
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@app.function(image=api_image, timeout=15 * 60)
@modal.asgi_app(requires_proxy_auth=True)
def api():
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import Response
    from pydantic import BaseModel, Field

    web = FastAPI(title="Egyptian Voice + Chat on Modal", version="1.0.0")

    class ChatRequest(BaseModel):
        prompt: str = Field(min_length=1, max_length=12000)
        system_prompt: str | None = Field(default=None, max_length=4000)
        max_new_tokens: int = Field(default=256, ge=1, le=2048)
        temperature: float = Field(default=0.6, ge=0.0, le=2.0)
        top_p: float = Field(default=0.9, ge=0.05, le=1.0)

    class TTSRequest(BaseModel):
        text: str = Field(min_length=1, max_length=4000)
        voice_description: str | None = Field(default=None, max_length=500)
        cfg_value: float = Field(default=2.0, ge=1.0, le=3.0)
        inference_timesteps: int = Field(default=10, ge=4, le=30)
        normalize: bool = True

    @web.get("/")
    def root():
        return {
            "status": "ok",
            "tts_model": VOX_MODEL_ID,
            "chat_model": QWEN_MODEL_ID,
            "gpu": "L4 per active model",
            "docs": "/docs",
            "auth": "Modal proxy authentication required",
        }

    @web.post("/chat")
    def chat(req: ChatRequest):
        try:
            answer = EgyptianQwen().chat.remote(
                req.prompt,
                req.system_prompt,
                req.max_new_tokens,
                req.temperature,
                req.top_p,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"model": QWEN_MODEL_ID, "response": answer}

    @web.post("/tts")
    def tts(req: TTSRequest):
        try:
            data = VoxTTS().synthesize.remote(
                req.text,
                req.voice_description,
                req.cfg_value,
                req.inference_timesteps,
                req.normalize,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(content=data, media_type="audio/wav")

    @web.post("/tts/clone")
    def tts_clone(
        reference: UploadFile = File(...),
        text: str = Form(...),
        voice_description: str | None = Form(None),
        prompt_text: str | None = Form(None),
        cfg_value: float = Form(2.0),
        inference_timesteps: int = Form(10),
    ):
        try:
            source = reference.file.read()
            data = VoxTTS().clone.remote(
                source,
                reference.filename or "reference.wav",
                text,
                voice_description,
                prompt_text,
                cfg_value,
                inference_timesteps,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(content=data, media_type="audio/wav")

    return web


@app.local_entrypoint()
def smoke_test(output_dir: str = "smoke-output") -> None:
    import json
    import time

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    chat_start = time.perf_counter()
    answer = EgyptianQwen().chat.remote(
        "رد بالمصري في سطرين: ليه النيل مهم لمصر؟",
        None,
        160,
        0.5,
        0.9,
    )
    chat_seconds = time.perf_counter() - chat_start
    (target / "qwen-response.txt").write_text(answer, encoding="utf-8")
    print(f"QWEN_OK seconds={chat_seconds:.3f} response={answer!r}")

    tts_start = time.perf_counter()
    audio = VoxTTS().synthesize.remote(
        "إزيك؟ دي تجربة لصوت مصري شغال من مودال. لو الصوت طالع طبيعي، يبقى كده إحنا تمام.",
        "Young Egyptian male voice, warm, natural and conversational, medium pace",
        2.0,
        10,
        True,
    )
    tts_seconds = time.perf_counter() - tts_start
    (target / "voxcpm2-egyptian.wav").write_bytes(audio)
    print(f"VOXCPM2_OK seconds={tts_seconds:.3f} bytes={len(audio)}")

    result = {
        "chat_model": QWEN_MODEL_ID,
        "tts_model": VOX_MODEL_ID,
        "gpu": "L4",
        "chat_seconds_including_cold_start": chat_seconds,
        "tts_seconds_including_cold_start": tts_seconds,
        "tts_bytes": len(audio),
    }
    (target / "smoke-test.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
