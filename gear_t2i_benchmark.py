from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "gear-t2i-1b-benchmark"
GEAR_COMMIT = "bf9665806e3c3666580467a42c8b04f56e4a33c8"
AR_REPO = "BinLin203/GEAR-T2I-GPIC-1B"
VQ_REPO = "BinLin203/GEAR-VQ"
TEXT_REPO = "Qwen/Qwen3-1.7B"
MODEL_ROOT = Path("/models")
AR_DIR = MODEL_ROOT / "gear-ar"
VQ_DIR = MODEL_ROOT / "gear-vq"
TEXT_DIR = MODEL_ROOT / "qwen3-1.7b"
T4_GPU_USD_PER_SECOND = 0.000164

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
models = modal.Volume.from_name("gear-t2i-models", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

download_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("huggingface_hub[hf_xet]==0.36.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1"})
)

runtime_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128",
        "git clone https://github.com/Tencent-Hunyuan/GEAR.git /opt/gear && cd /opt/gear && git checkout " + GEAR_COMMIT,
        "pip install -r /opt/gear/requirements.txt",
    )
    .env({"HF_HOME": str(MODEL_ROOT / "hf-home"), "PYTHONUNBUFFERED": "1"})
)


@app.function(
    image=download_image,
    secrets=[hf_secret],
    volumes={str(MODEL_ROOT): models},
    timeout=60 * 60,
    cpu=4,
    memory=8192,
)
def download_models() -> dict:
    import os
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    for repo, dest in ((AR_REPO, AR_DIR), (VQ_REPO, VQ_DIR), (TEXT_REPO, TEXT_DIR)):
        dest.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo, local_dir=str(dest), token=token)
        print(f"DOWNLOADED repo={repo} dest={dest}")
    models.commit()
    return {"ar": str(AR_DIR), "vq": str(VQ_DIR), "text": str(TEXT_DIR)}


@app.function(
    image=runtime_image,
    gpu="T4",
    secrets=[hf_secret],
    volumes={str(MODEL_ROOT): models},
    scaledown_window=10,
    timeout=60 * 30,
    memory=24576,
)
def generate_batch(prompts: list[str], seed: int = 77000) -> dict:
    work = Path("/tmp/gear-benchmark")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    prompt_file = work / "prompts.jsonl"
    with prompt_file.open("w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            f.write(json.dumps({"key": f"p{i:02d}", "caption": prompt, "caption_type": "benchmark"}) + "\n")

    ar_ckpt = AR_DIR / "gear-t2i-gpic-1b.pt"
    vq_ckpt = VQ_DIR / "gear-vq.pt"
    if not ar_ckpt.exists() or not vq_ckpt.exists():
        raise RuntimeError("GEAR checkpoints are missing; run download_models first")

    out_dir = work / "out"
    cmd = [
        "python", "/opt/gear/src/inference_t2i.py",
        "--ckpt-path", str(ar_ckpt),
        "--vq-ckpt-path", str(vq_ckpt),
        "--ar-model", "LlamaGen-1B",
        "--image-size", "256",
        "--text-encoder", str(TEXT_DIR),
        "--text-max-len", "300",
        "--cls-token-num", "300",
        "--prompts-jsonl", str(prompt_file),
        "--output-dir", str(out_dir),
        "--per-proc-batch-size", "1",
        "--cfg-scale", "1.75",
        "--seed", str(seed),
        "--no-pack-npz",
        "--tag", "modal-benchmark",
    ]
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd="/opt/gear", text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError(f"GEAR inference failed with code {proc.returncode}")

    pngs = sorted(out_dir.rglob("*.png"))
    if len(pngs) != len(prompts):
        raise RuntimeError(f"Expected {len(prompts)} PNGs, found {len(pngs)}")
    return {
        "seconds": elapsed,
        "images": [p.read_bytes() for p in pngs],
        "stdout_tail": proc.stdout[-6000:],
    }


@app.local_entrypoint()
def benchmark(output_dir: str = "gear-output") -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = generate_batch.remote([p for _, p in PROMPTS], 77000)
    remote_seconds = time.perf_counter() - started

    image_bytes = result["images"]
    for i, ((slug, prompt), data) in enumerate(zip(PROMPTS, image_bytes), start=1):
        (target / f"{i:02d}-{slug}.png").write_bytes(data)

    gpu_seconds = float(result["seconds"])
    gpu_cost = gpu_seconds * T4_GPU_USD_PER_SECOND
    summary = {
        "experiment": "gear-t2i-gpic-1b-t4-graded-10",
        "model": AR_REPO,
        "release": "2026-06-30",
        "license": "Apache-2.0 (respect upstream licenses)",
        "gpu": "T4",
        "native_resolution": "256x256",
        "image_count": len(PROMPTS),
        "cfg_scale": 1.75,
        "gpu_function_seconds": gpu_seconds,
        "client_wall_seconds": remote_seconds,
        "estimated_gpu_usd": gpu_cost,
        "estimated_gpu_usd_per_image": gpu_cost / len(PROMPTS),
        "estimated_images_per_30_gpu_only": 30.0 / (gpu_cost / len(PROMPTS)),
        "prompts": [{"difficulty": i, "slug": slug, "prompt": prompt} for i, (slug, prompt) in enumerate(PROMPTS, 1)],
        "note": "GEAR native output is 256x256. The GPU estimate uses the full single T4 inference function wall time, including model loading, so it is conservative for batched use. CPU/RAM and storage are not included.",
    }
    (target / "benchmark.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (target / "gear-log-tail.txt").write_text(result["stdout_tail"], encoding="utf-8")
    print("GEAR_JSON=" + json.dumps(summary, separators=(",", ":")))
