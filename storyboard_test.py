from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from modal_app import Editor, Generator, app

L4_GPU_USD_PER_SECOND = 0.000222
WIDTH = 1024
HEIGHT = 576

STYLE_LOCK = """
Use the supplied image as the strict character-design and art-direction reference.
Preserve the SAME Ramses character identity and costume whenever Ramses appears:
round white face, simple black oval eyes, tiny expressive mouth, bold black outlines,
blue-and-gold striped nemes headdress, white sleeveless tunic, gold belt, long simple limbs.
Preserve the SAME visual language across every frame: flat 2D educational explainer cartoon,
clean vector-like shapes, thick black outlines, minimal or no shading, simple geometric anatomy,
limited warm Egyptian desert palette, pale blue sky, sandy ochre and muted gold, uncluttered
composition, friendly documentary infographic feeling, crisp 16:9 frame. No photorealism,
no 3D render, no painterly texture, no cinematic realism. Create a NEW composition for the
requested scene while keeping the style and recurring character visually consistent.
""".strip()

CONCEPT_PROMPT = """
Create the master visual reference for an educational animated explainer about Ramses II.
Flat 2D vector-cartoon style with bold clean black outlines, simple geometric characters,
minimal shading, limited pastel desert palette. Ramses II stands centered in a simple Egyptian
desert landscape with two small pyramids far in the background. Give him a round white face,
large simple black oval eyes, a subtle friendly expression, a blue-and-gold striped nemes
headdress, white sleeveless tunic, gold belt and collar, and long simple limbs. The framing
should feel like a modern YouTube history explainer animation: clean, readable, playful but
not childish, lots of negative space, pale blue sky and warm sand. No text. No photorealism.
This image will be used as the exact character and style reference for all following scenes.
""".strip()

SCENES = [
    (
        "02-rule-66-years",
        "Show Ramses proudly standing in front of a simplified Abu Simbel temple facade. "
        "At the top, render ONLY the exact Arabic text '٦٦ سنة' in large bold black lettering. "
        "Add a simple visual motif suggesting a very long reign, but keep the frame clean and easy to read.",
    ),
    (
        "03-peace-treaty",
        "Show Ramses on the left shaking hands with a Hittite envoy on the right. Between them, "
        "they hold a small clay treaty tablet with simple ancient symbols. Use a clean desert background "
        "and a simplified temple silhouette. The mood is diplomatic and historic. No modern text.",
    ),
    (
        "04-long-life",
        "Create an infographic-style scene about Ramses living to around ninety years old. Ramses stands "
        "on the left looking elderly but recognizable, while many small sun icons form neat rows across the frame "
        "to suggest passing years. Keep it simple, graphic and educational. No extra text.",
    ),
    (
        "05-ct-scan-2016",
        "Move the story to a modern medical research room while preserving the exact same flat explainer style. "
        "Show a wrapped ancient Egyptian mummy entering a simplified CT scanner. Two modern Egyptian researchers "
        "in lab coats observe a monitor showing a simple skull scan. Ramses himself does not need to appear.",
    ),
    (
        "06-ct-slices-explainer",
        "Create a clear educational infographic explaining a CT scan as many slices through the body. Show a simplified "
        "mummy silhouette on the left, then several evenly spaced cross-section slice panels moving toward a skull image "
        "on the right. Use arrows and clean diagram logic, but no labels or text.",
    ),
    (
        "07-dental-damage",
        "Show a simplified close-up medical diagram of Ramses' jaw and teeth in the same cartoon style. Several teeth are "
        "worn down, a few are missing, and one back molar area on the left side is highlighted with a red-orange swollen "
        "abscess near the root. Educational, non-gory, very clear anatomy diagram.",
    ),
    (
        "08-king-toothache",
        "Show Ramses seated on a simple golden throne inside a stylized palace, holding one cheek in obvious tooth pain. "
        "Around him are symbols of wealth and power: a small gold chest, a priest and an ancient physician looking concerned. "
        "Make the irony visually clear: powerful king, ordinary unbearable toothache.",
    ),
    (
        "09-king-and-farmer",
        "Split the composition visually into two simple halves while keeping one continuous illustration. On the left, Ramses "
        "at a royal table; on the right, a poor Nile Delta farmer at a humble table. Both are eating similar coarse ancient bread, "
        "and both touch a sore cheek. Emphasize that wealth did not protect their teeth. No text.",
    ),
    (
        "10-ordinary-day",
        "Create a lively but uncluttered ancient Egyptian everyday-life scene: a clerk writing on papyrus at a desk, workers "
        "carrying baskets, a baker preparing bread, two ordinary people arguing mildly, and Ramses only as a tiny distant royal "
        "figure or wall image. The feeling should be 'an ordinary Tuesday in ancient Egypt', not tombs and treasure. No text.",
    ),
]


@app.local_entrypoint()
def storyboard(output_dir: str = "storyboard-output") -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    overall_start = time.perf_counter()

    # 1) Master concept visual / reference anchor.
    gen = Generator()
    t0 = time.perf_counter()
    anchor = gen.generate.remote(CONCEPT_PROMPT, WIDTH, HEIGHT, 22001)
    anchor_seconds = time.perf_counter() - t0
    anchor_path = target / "01-concept-anchor.png"
    anchor_path.write_bytes(anchor)
    print(
        f"STORY_IMAGE index=1 role=anchor seconds={anchor_seconds:.3f} "
        f"bytes={len(anchor)} requested={WIDTH}x{HEIGHT} path={anchor_path}"
    )

    # 2-10) Every frame edits the SAME anchor, avoiding cumulative style drift.
    editor = Editor()
    edit_durations: list[float] = []
    output_sizes = [len(anchor)]
    scene_records = []

    for index, (slug, scene_instruction) in enumerate(SCENES, start=2):
        prompt = STYLE_LOCK + "\n\nNEW SCENE:\n" + scene_instruction
        started = time.perf_counter()
        data = editor.edit.remote(anchor, prompt, 1024, 22000 + index)
        elapsed = time.perf_counter() - started
        edit_durations.append(elapsed)
        output_sizes.append(len(data))
        path = target / f"{index:02d}-{slug}.png"
        path.write_bytes(data)
        scene_records.append(
            {
                "index": index,
                "slug": slug,
                "seconds": elapsed,
                "bytes": len(data),
            }
        )
        print(
            f"STORY_IMAGE index={index} role=edit slug={slug} seconds={elapsed:.3f} "
            f"bytes={len(data)} path={path}"
        )

    wall_seconds = time.perf_counter() - overall_start
    editor_cold = edit_durations[0]
    editor_warm = edit_durations[1:]

    # Wall-time * L4 rate is intentionally a conservative GPU-only upper-bound estimate.
    estimated_gpu_usd = wall_seconds * L4_GPU_USD_PER_SECOND
    estimated_images_per_30 = 30.0 / (estimated_gpu_usd / 10.0)

    result = {
        "model_generation": "SceneWorks/Mage-Flow-Turbo",
        "model_editing": "SceneWorks/Mage-Flow-Edit-Turbo",
        "gpu": "L4",
        "l4_gpu_usd_per_second": L4_GPU_USD_PER_SECOND,
        "resolution_requested": f"{WIDTH}x{HEIGHT}",
        "steps": 4,
        "image_count": 10,
        "anchor_seconds": anchor_seconds,
        "editor_first_call_seconds": editor_cold,
        "editor_warm_average_seconds": statistics.mean(editor_warm),
        "editor_warm_min_seconds": min(editor_warm),
        "editor_warm_max_seconds": max(editor_warm),
        "total_wall_seconds": wall_seconds,
        "estimated_gpu_usd_upper_bound": estimated_gpu_usd,
        "estimated_usd_per_image_upper_bound": estimated_gpu_usd / 10.0,
        "estimated_images_per_30_upper_bound_basis": estimated_images_per_30,
        "output_bytes": output_sizes,
        "scenes": scene_records,
        "note": "Cost estimate uses end-to-end wall time times published L4 GPU rate; actual Modal billing may differ because caller/queue overhead is not necessarily billable GPU time.",
    }
    (target / "storyboard-benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("STORYBOARD_JSON=" + json.dumps(result, separators=(",", ":")))
