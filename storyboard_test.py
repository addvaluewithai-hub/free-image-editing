from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from modal_app import Generator, app

L4_GPU_USD_PER_SECOND = 0.000222
WIDTH = 1024
HEIGHT = 576

STYLE_BIBLE = """
Create a premium flat 2D educational history explainer illustration in a polished vector-cartoon style.
Use bold confident black outlines, smooth flat colors, almost no shading, simple geometric anatomy,
clean silhouette design, strong visual hierarchy, and generous negative space. The image should feel
like a high-end YouTube documentary explainer frame: intelligent, elegant, playful but not childish.
Use a restrained ancient-Egypt palette: pale cyan-blue sky, warm sandy ochre, muted gold, dark navy,
off-white clothing, and only small accent colors. Favor simple readable shapes over visual clutter.

When Ramses II appears, keep his design consistent across scenes: round white face, large simple black
oval eyes, tiny expressive mouth, blue-and-gold striped nemes headdress, white sleeveless tunic,
muted-gold collar and belt, long simple limbs, and a calm confident presence.

Avoid photorealism, 3D rendering, painterly textures, anime aesthetics, realistic skin detail,
complex gradients, excessive ornament, dramatic cinematic lighting, and text unless explicitly requested.
Every frame must be a clear 16:9 composition suitable for an educational animated history video.
""".strip()

SCENES = [
    (
        "01-ramses-intro",
        "Hero introduction to Ramses II. Show Ramses standing confidently in the foreground in a broad Egyptian desert landscape. "
        "Place a simplified Abu Simbel temple facade behind him and two distant pyramids on the horizon. Make him feel iconic and powerful, "
        "but keep the composition clean, graphic and immediately readable. No text.",
    ),
    (
        "02-long-reign",
        "Show Ramses II standing proudly beside a simplified monumental Abu Simbel facade. Behind and around him, arrange a long elegant "
        "sequence of small sun-disc symbols and simple calendar-like marks to visually communicate an exceptionally long reign. "
        "Use a strong asymmetric composition with Ramses on one side and the visual timeline on the other. No text.",
    ),
    (
        "03-peace-treaty",
        "Historical diplomacy scene. Ramses II stands on the left shaking hands with a Hittite envoy on the right. Between them they jointly "
        "hold a small clay treaty tablet marked with simple ancient symbols. Use a calm desert setting with a subtle temple silhouette, "
        "balanced body language, and a clear visual message of peace and negotiation. No text.",
    ),
    (
        "04-extraordinary-longevity",
        "Infographic-like scene showing Ramses II living to an unusually old age. Put an elderly but recognizable Ramses on the left, still in "
        "his blue-and-gold nemes, with subtle age lines and a walking staff. On the right, show many neat rows of small sun symbols receding across "
        "the frame to suggest decades passing. Keep it elegant, spacious and educational. No text.",
    ),
    (
        "05-ct-scan-2016",
        "Modern medical research room in the same polished flat 2D explainer style. A wrapped ancient Egyptian mummy lies on a table entering a "
        "large white CT scanner. Two Egyptian medical researchers in clean lab coats stand beside a monitor showing a simplified grayscale skull scan. "
        "Bright, clean, friendly scientific environment. No standing pharaoh character. No text.",
    ),
    (
        "06-ct-slices",
        "Clear educational infographic explaining CT imaging. Far left: a simplified wrapped mummy silhouette. Across the center: several evenly spaced "
        "translucent cross-section slice panels, each slightly separated, connected by simple arrows. Far right: a clean stylized skull and jaw scan. "
        "Strong left-to-right visual logic, no standing people, no text labels, no clutter.",
    ),
    (
        "07-dental-damage",
        "Large medical explainer close-up of an ancient Egyptian jaw and teeth. Show several teeth severely worn down, two missing teeth, and one rear molar "
        "with a clearly visible red-orange abscess pocket around the root and nearby inflamed bone. Make the anatomy simple, non-gory and immediately understandable, "
        "like a polished vector medical infographic in the same history-video art style. No text.",
    ),
    (
        "08-king-toothache",
        "Visual irony scene. Ramses II sits on a simple golden throne inside a stylized palace chamber, leaning slightly and pressing one cheek in obvious tooth pain. "
        "Nearby, an ancient physician and priest look concerned. Include restrained symbols of enormous wealth and power such as a small gold chest and ceremonial objects. "
        "The frame should instantly communicate: the most powerful man in the room is defeated by an ordinary toothache. No text.",
    ),
    (
        "09-king-vs-farmer",
        "Comparative split composition in one continuous illustration. Left side: Ramses II at a refined royal table eating coarse ancient bread and touching a sore cheek. "
        "Right side: a poor Nile Delta farmer at a humble wooden table eating similar bread and touching the same side of his face in pain. Make the visual parallel obvious and elegant: "
        "different status, same dental suffering. No text.",
    ),
    (
        "10-ordinary-day",
        "Lively but uncluttered everyday-life scene in ancient Egypt, focused on ordinary people rather than tombs or treasure. Show a clerk writing on papyrus at a desk, "
        "workers carrying baskets, a baker shaping bread, and two townspeople mildly arguing near simple mud-brick buildings. Add a small palm tree and distant Nile greenery. "
        "The mood should feel like an ordinary workday in a living city. Ramses should not be the main subject. No text.",
    ),
]


@app.local_entrypoint()
def storyboard(output_dir: str = "storyboard-output") -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    gen = Generator()
    durations: list[float] = []
    records = []
    overall_start = time.perf_counter()

    for index, (slug, scene) in enumerate(SCENES, start=1):
        prompt = STYLE_BIBLE + "\n\nSCENE DIRECTION:\n" + scene
        started = time.perf_counter()
        data = gen.generate.remote(prompt, WIDTH, HEIGHT, 44000 + index)
        elapsed = time.perf_counter() - started
        durations.append(elapsed)
        path = target / f"{index:02d}-{slug}.png"
        path.write_bytes(data)
        records.append({
            "index": index,
            "slug": slug,
            "mode": "generate",
            "seconds": elapsed,
            "bytes": len(data),
        })
        print(
            f"GENERATE_IMAGE index={index} slug={slug} seconds={elapsed:.3f} "
            f"bytes={len(data)} path={path}"
        )

    wall_seconds = time.perf_counter() - overall_start
    warm = durations[1:]
    estimated_gpu_usd = wall_seconds * L4_GPU_USD_PER_SECOND

    result = {
        "experiment": "ramses-generate-only-strong-prompts-v3",
        "model": "SceneWorks/Mage-Flow-Turbo",
        "gpu": "L4",
        "resolution_requested": f"{WIDTH}x{HEIGHT}",
        "steps": 4,
        "image_count": len(SCENES),
        "first_call_seconds": durations[0],
        "warm_average_seconds": statistics.mean(warm),
        "warm_min_seconds": min(warm),
        "warm_max_seconds": max(warm),
        "total_wall_seconds": wall_seconds,
        "l4_gpu_usd_per_second": L4_GPU_USD_PER_SECOND,
        "estimated_gpu_usd_upper_bound": estimated_gpu_usd,
        "estimated_usd_per_image_upper_bound": estimated_gpu_usd / len(SCENES),
        "estimated_images_per_30_upper_bound_basis": 30.0 / (estimated_gpu_usd / len(SCENES)),
        "scenes": records,
        "note": "Generate-only test: no reference image, no Mage Edit, no Arabic text. Cost uses end-to-end wall time times the published L4 GPU rate, so it is a conservative estimate rather than exact billing telemetry.",
    }
    (target / "storyboard-benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("GENERATE_ONLY_JSON=" + json.dumps(result, separators=(",", ":")))
