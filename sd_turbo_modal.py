from __future__ import annotations

import io

import modal

APP_NAME = "sd-turbo-simple-images"
MODEL_ID = "stabilityai/sd-turbo"
MODEL_DIR = "/models/sd-turbo"

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("sd-turbo-model-cache", create_if_missing=True)

runtime_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        "diffusers==0.35.1",
        "transformers==4.55.4",
        "accelerate==1.10.1",
        "safetensors==0.6.2",
        "huggingface_hub==0.34.4",
    )
)


@app.function(image=runtime_image, volumes={"/models": model_volume}, timeout=60 * 20)
def download_model() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.ckpt", "*.bin"],
    )
    model_volume.commit()
    print(f"Cached {MODEL_ID} at {MODEL_DIR}")


@app.cls(
    image=runtime_image,
    gpu="T4",
    volumes={"/models": model_volume},
    scaledown_window=30,
    timeout=60 * 10,
)
class SDTurboGenerator:
    @modal.enter()
    def load(self) -> None:
        import torch
        from diffusers import AutoPipelineForText2Image

        self.device = "cuda"
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=True,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        print("SD-Turbo loaded on T4")

    @modal.method()
    def generate(self, prompt: str, seed: int, width: int = 512, height: int = 512) -> bytes:
        import torch

        generator = torch.Generator(device=self.device).manual_seed(seed)
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=1,
            guidance_scale=0.0,
            width=width,
            height=height,
            generator=generator,
        ).images[0]

        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()
