from __future__ import annotations

import io
import json
import statistics
import time
from pathlib import Path

import modal

APP_NAME = "sd-turbo-simple-images"
MODEL_ID = "stabilityai/sd-turbo"
CACHE_DIR = "/cache/hf"
WIDTH = 512
HEIGHT = 512
STEPS = 1
T4_GPU_USD_PER_SECOND = 0.000164

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("sd-turbo-model-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "diffusers==0.35.1",
        "transformers==4.55.4",
        "accelerate==1.10.1",
        "safetensors==0.6.2",
        "huggingface_hub==0.34.4",
        "Pillow==11.3.0",
    )
)

PROMPTS = [
    ("01-apple", "A single fresh red apple centered on a seamless pure white background, simple studio product photo, soft natural shadow, crisp shape, no other objects."),
    ("02-car", "A blue compact modern hatchback car, clean side view, centered on a light gray seamless studio background, realistic proportions, soft shadow under the tires, no people."),
    ("03-chair", "A handcrafted wooden dining chair centered in a warm beige photography studio, three-quarter view, visible wood grain, soft diffused light, simple catalog product photography."),
    ("04-fruit-bowl", "A white ceramic bowl filled with red apples, yellow bananas and oranges on a light wooden kitchen table, morning window light from the left, clean uncluttered background, realistic still life."),
    ("05-bicycle-street", "A bright red city bicycle parked beside one leafy sidewalk tree on a quiet sunny European street, cream building facade behind it, gentle afternoon shadows, clean composition."),
    ("06-cat-window", "A fluffy orange tabby cat sitting upright on a wide white windowsill beside one small green potted plant, warm morning sunlight, soft curtains, detailed fur, peaceful home interior."),
    ("07-coffee-desk", "A tidy cozy wooden desk seen from a slightly elevated angle, open cream notebook, white ceramic coffee mug with visible steam, black eyeglasses and a small brass desk lamp, warm directional light, believable object spacing."),
    ("08-kitchen-scene", "A bright modern Scandinavian kitchen interior with white cabinets, stainless refrigerator, wooden dining table, fruit bowl, black kettle and exactly two chairs, large sunlit window, realistic spatial layout, clean architectural photography."),
    ("09-market-scene", "A small outdoor Mediterranean fruit market stall under a red-and-white striped canopy, baskets of apples bananas and oranges arranged by type, one smiling shopkeeper behind the counter and exactly two customers in front, sunny street, coherent anatomy and clear depth."),
    ("10-hard-scene", "A glossy red toy sports car on a walnut table beside a green apple and a transparent glass of water half full. Behind them is a rain-covered window at dusk. A small round mirror leans against the wall and shows a faint physically plausible reflection of the toy car. Soft lamp light from the right creates realistic highlights on the painted car, refraction through the water glass, cast shadows on the table, and subtle reflections, while every object keeps correct scale and spatial relationships."),
]


@app.function(image=image, volumes={"/cache": volume}, secrets=[hf_secret], timeout=60 * 20)
def download_model():
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=MODEL_ID, cache_dir=CACHE_DIR)
    volume.commit()
    print(f"MODEL_CACHED path={path}")


@app.cls(
    image=image,
    gpu="T4",
    volumes={"/cache": volume},
    secrets=[hf_secret],
    scaledown_window=30,
    timeout=60 * 20,
)
class SDTurboGenerator:
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import AutoPipelineForText2Image

        self.device = "cuda"
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=CACHE_DIR,
            local_files_only=True,
        )
        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        print("SD_TURBO_READY")

    @modal.method()
    def generate(self, prompt: str, seed: int) -> bytes:
        import torch

        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipe(
            prompt=prompt,
            num_inference_steps=STEPS,
            guidance_scale=0.0,
            width=WIDTH,
            height=HEIGHT,
            generator=generator,
        )
        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")
        return buf.getvalue()


@app.local_entrypoint()
def benchmark(output_dir: str = "sd-turbo-output"):
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    generator = SDTurboGenerator()
    durations: list[float] = []
    records = []
    total_start = time.perf_counter()

    for index, (slug, prompt) in enumerate(PROMPTS, start=1):
        started = time.perf_counter()
        data = generator.generate.remote(prompt, 66000 + index)
        elapsed = time.perf_counter() - started
        durations.append(elapsed)

        path = target / f"{index:02d}-{slug}.png"
        path.write_bytes(data)
        records.append(
            {
                "difficulty": index,
                "slug": slug,
                "seconds": elapsed,
                "bytes": len(data),
                "prompt": prompt,
            }
        )
        print(f"SD_TURBO_IMAGE difficulty={index} slug={slug} seconds={elapsed:.3f} bytes={len(data)}")

    wall_seconds = time.perf_counter() - total_start
    warm = durations[1:]
    gpu_cost = wall_seconds * T4_GPU_USD_PER_SECOND
    per_image = gpu_cost / len(PROMPTS)

    result = {
        "experiment": "sd-turbo-t4-graded-10",
        "model": MODEL_ID,
        "gpu": "T4",
        "resolution": f"{WIDTH}x{HEIGHT}",
        "steps": STEPS,
        "image_count": len(PROMPTS),
        "first_call_seconds": durations[0],
        "warm_average_seconds": statistics.mean(warm),
        "warm_min_seconds": min(warm),
        "warm_max_seconds": max(warm),
        "total_wall_seconds": wall_seconds,
        "t4_gpu_usd_per_second": T4_GPU_USD_PER_SECOND,
        "estimated_gpu_usd_upper_bound": gpu_cost,
        "estimated_usd_per_image_upper_bound": per_image,
        "estimated_images_per_30_upper_bound_basis": 30.0 / per_image,
        "images": records,
        "note": "Model weights are pre-cached on a Modal Volume before the GPU benchmark. Cost estimate uses end-to-end benchmark wall time times current T4 GPU rate and is intentionally conservative; exact Modal billing also includes CPU/RAM and container lifecycle details.",
    }
    (target / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("SD_TURBO_JSON=" + json.dumps(result, separators=(",", ":")))
