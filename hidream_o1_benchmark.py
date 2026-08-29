from __future__ import annotations

import io
import json
import statistics
import sys
import time
from pathlib import Path

import modal

APP_NAME = "hidream-o1-8b-benchmark"
MODEL_ID = "HiDream-ai/HiDream-O1-Image"
MODEL_ROOT = Path("/models")
MODEL_DIR = MODEL_ROOT / "hidream-o1-image"
REPO_DIR = "/opt/hidream"
WIDTH = 1024
HEIGHT = 1024
STEPS = 50
L4_GPU_USD_PER_SECOND = 0.000222

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

app = modal.App(APP_NAME)
models = modal.Volume.from_name("hidream-o1-models", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_xet]==0.36.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1"})
)

runtime_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128",
        "git clone --depth 1 https://github.com/HiDream-ai/HiDream-O1-Image.git /opt/hidream",
        "pip install -r /opt/hidream/requirements.txt",
        "python - <<'PY'\np='/opt/hidream/models/pipeline.py'\ns=open(p).read()\ns=s.replace('\\\"use_flash_attn\\\": True', '\\\"use_flash_attn\\\": False')\nopen(p,'w').write(s)\nPY",
    )
    .env({"HF_HOME": str(MODEL_ROOT / "hf-home"), "PYTHONUNBUFFERED": "1"})
)


@app.function(
    image=download_image,
    secrets=[hf_secret],
    volumes={str(MODEL_ROOT): models},
    timeout=60 * 90,
    cpu=4,
    memory=16384,
)
def download_model() -> str:
    import os
    from huggingface_hub import snapshot_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_DIR),
        token=os.environ.get("HF_TOKEN"),
    )
    models.commit()
    print(f"HIDREAM_MODEL_CACHED path={path}")
    return path


@app.cls(
    image=runtime_image,
    gpu="L4",
    secrets=[hf_secret],
    volumes={str(MODEL_ROOT): models},
    scaledown_window=30,
    max_containers=1,
    timeout=60 * 30,
    memory=32768,
)
class HiDreamGenerator:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoProcessor

        sys.path.insert(0, REPO_DIR)
        from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
        from inference import add_special_tokens, get_tokenizer

        if not MODEL_DIR.exists():
            raise RuntimeError("HiDream weights missing; run download_model first")

        self.processor = AutoProcessor.from_pretrained(str(MODEL_DIR), local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(MODEL_DIR),
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            local_files_only=True,
        ).eval()
        tokenizer = get_tokenizer(self.processor)
        add_special_tokens(tokenizer)
        print("HIDREAM_READY")

    @modal.method()
    def generate(self, prompt: str, seed: int) -> dict:
        sys.path.insert(0, REPO_DIR)
        from models.pipeline import generate_image

        started = time.perf_counter()
        image = generate_image(
            model=self.model,
            processor=self.processor,
            prompt=prompt,
            ref_image_paths=[],
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=STEPS,
            guidance_scale=5.0,
            shift=3.0,
            timesteps_list=None,
            scheduler_name="default",
            seed=seed,
            keep_original_aspect=False,
            layout_bboxes=None,
        )
        inference_seconds = time.perf_counter() - started
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return {"image": buf.getvalue(), "inference_seconds": inference_seconds}


@app.local_entrypoint()
def benchmark(output_dir: str = "hidream-output") -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    generator = HiDreamGenerator()
    call_durations: list[float] = []
    inference_durations: list[float] = []
    records = []
    total_start = time.perf_counter()

    for index, (slug, prompt) in enumerate(PROMPTS, start=1):
        started = time.perf_counter()
        result = generator.generate.remote(prompt, 88000 + index)
        call_seconds = time.perf_counter() - started
        inference_seconds = float(result["inference_seconds"])
        data = result["image"]
        call_durations.append(call_seconds)
        inference_durations.append(inference_seconds)
        (target / f"{index:02d}-{slug}.png").write_bytes(data)
        records.append({
            "difficulty": index,
            "slug": slug,
            "call_seconds": call_seconds,
            "inference_seconds": inference_seconds,
            "bytes": len(data),
            "prompt": prompt,
        })
        print(
            f"HIDREAM_IMAGE difficulty={index} slug={slug} call_seconds={call_seconds:.3f} "
            f"inference_seconds={inference_seconds:.3f} bytes={len(data)}"
        )

    wall_seconds = time.perf_counter() - total_start
    warm_calls = call_durations[1:]
    gpu_cost_upper = wall_seconds * L4_GPU_USD_PER_SECOND
    per_image_upper = gpu_cost_upper / len(PROMPTS)
    inference_only_cost = sum(inference_durations) * L4_GPU_USD_PER_SECOND

    summary = {
        "experiment": "hidream-o1-8b-full-l4-graded-10",
        "model": MODEL_ID,
        "license": "MIT",
        "gpu": "L4",
        "resolution": f"{WIDTH}x{HEIGHT}",
        "steps": STEPS,
        "guidance_scale": 5.0,
        "image_count": len(PROMPTS),
        "first_call_seconds": call_durations[0],
        "warm_call_average_seconds": statistics.mean(warm_calls),
        "warm_call_min_seconds": min(warm_calls),
        "warm_call_max_seconds": max(warm_calls),
        "average_inference_seconds": statistics.mean(inference_durations),
        "total_wall_seconds": wall_seconds,
        "l4_gpu_usd_per_second": L4_GPU_USD_PER_SECOND,
        "estimated_gpu_usd_upper_bound": gpu_cost_upper,
        "estimated_usd_per_image_upper_bound": per_image_upper,
        "estimated_images_per_30_upper_bound_basis": 30.0 / per_image_upper,
        "inference_only_gpu_usd": inference_only_cost,
        "images": records,
        "note": "Full HiDream-O1 8B at 1024x1024, 50 steps, on L4. End-to-end wall-time cost includes first model-load latency and is a conservative GPU-only estimate; exact Modal billing also includes CPU/RAM and lifecycle details.",
    }
    (target / "benchmark.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("HIDREAM_JSON=" + json.dumps(summary, separators=(",", ":")))
