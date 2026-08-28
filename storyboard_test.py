from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from modal_app import Editor, Generator, app

L4_GPU_USD_PER_SECOND = 0.000222
WIDTH = 1024
HEIGHT = 576

STYLE_BIBLE = """
Flat 2D educational history explainer cartoon. Clean vector-like shapes, thick confident black
outlines, almost no shading, simple geometric anatomy, friendly expressive faces with white
circular heads and simple black oval eyes. Limited warm ancient-Egypt palette: pale blue sky,
sandy ochre, muted gold, dark navy blue, white clothing, small restrained accent colors.
Simple readable 16:9 composition, uncluttered negative space, infographic clarity, modern
YouTube documentary explainer feeling. Avoid photorealism, 3D rendering, painterly textures,
complex gradients, cinematic realism, anime, detailed realistic skin, or ornate visual noise.
Do not render any written text unless explicitly requested.
""".strip()

RAMSES_LOCK = """
Use the supplied image as the strict Ramses II character-design reference. Whenever Ramses
appears, preserve his identity: round white face, simple black oval eyes, tiny expressive mouth,
bold black outlines, blue-and-gold striped nemes headdress, white sleeveless tunic, muted-gold
collar and belt, long simple limbs. Preserve the same flat 2D educational explainer art direction,
but you MAY substantially rearrange the composition, pose, camera framing, props and background
to create the requested NEW scene. Do not simply keep the reference pose in the center.
""".strip()

CONCEPT_PROMPT = STYLE_BIBLE + """

Create the MASTER CHARACTER / STYLE REFERENCE for an educational animated explainer about
Ramses II. Ramses stands centered, full body, in a simple Egyptian desert landscape with two
small pyramids far in the background. He has a round white face, large simple black oval eyes,
a subtle friendly expression, blue-and-gold striped nemes headdress, white sleeveless tunic,
muted-gold belt and collar, and long simple limbs. Clean readable silhouette, lots of negative
space. No text. This must look like a reusable animation model-sheet frame, not a realistic painting.
""".strip()

# Scenes where character identity matters: use Mage Edit with the concept anchor.
EDIT_SCENES = [
    (
        2,
        "02-rule-66-years",
        "Create a NEW wide composition: Ramses stands proudly off-center in front of a simplified Abu Simbel temple facade. "
        "Show a clean visual motif of many small sun symbols or calendar marks suggesting an exceptionally long reign. "
        "Leave generous empty space at the top for a title that will be added later in post-production. No generated text.",
    ),
    (
        3,
        "03-peace-treaty",
        "Create a NEW wide composition: Ramses on the left shakes hands with a Hittite envoy on the right. Between them they "
        "hold a small clay treaty tablet with simple ancient marks. Simplified temple silhouette and desert background. "
        "Clear diplomatic body language; Ramses must remain recognizable from the reference. No text.",
    ),
    (
        4,
        "04-long-life",
        "Create a NEW infographic composition about Ramses reaching roughly ninety years old. Put an elderly but clearly "
        "recognizable Ramses on the left, with subtle age lines and a walking staff, while neat rows of small sun icons fill "
        "the right side to suggest many passing years. Keep the frame spacious and educational. No text.",
    ),
    (
        8,
        "08-king-toothache",
        "Create a NEW palace scene: Ramses sits on a simple golden throne, leaning slightly and holding one cheek in obvious "
        "tooth pain. Nearby are a concerned ancient physician and priest plus simple symbols of royal wealth such as a small "
        "gold chest. The visual joke is clear: immense power, ordinary toothache. No text.",
    ),
    (
        9,
        "09-king-and-farmer",
        "Create a NEW split visual composition. Left half: recognizable Ramses at a royal table eating coarse ancient bread and "
        "touching a sore cheek. Right half: a poor Nile Delta farmer at a humble table eating similar bread and touching the same "
        "side of his face. Make the visual comparison immediate and clean. No text.",
    ),
]

# Scenes needing a radically different composition: generate from scratch using the same style bible.
GENERATE_SCENES = [
    (
        5,
        "05-ct-scan-2016",
        "Modern medical research room in the exact visual style described above. A wrapped ancient Egyptian mummy lies on a "
        "table entering a simplified white CT scanner. Two modern Egyptian researchers in lab coats stand to one side observing "
        "a monitor with a simple grayscale skull scan. Absolutely no pharaoh character standing in the room. No text.",
    ),
    (
        6,
        "06-ct-slices-explainer",
        "Educational infographic in the exact visual style described above explaining CT imaging as slices. A simplified wrapped "
        "mummy silhouette at far left, then a sequence of evenly spaced translucent cross-section slice panels progressing toward "
        "a simplified skull image at far right, connected by clean arrows. No standing characters, no pharaoh, no labels, no text.",
    ),
    (
        7,
        "07-dental-damage",
        "Medical explainer diagram in the exact visual style described above. Large clean side-view cutaway of an ancient jaw and "
        "teeth filling most of the frame. Several teeth visibly worn down, two missing teeth, and one rear molar on the left has a "
        "clear red-orange abscess pocket around its root. Non-gory, anatomically readable, simple vector infographic. No characters, no text.",
    ),
    (
        10,
        "10-ordinary-day",
        "Ancient Egyptian everyday-life scene in the exact visual style described above, with NO pharaoh as the main subject. "
        "A clerk writes on papyrus at a small desk, workers carry baskets, a baker shapes bread, and two ordinary people mildly "
        "argue in the background. Mud-brick buildings and a small palm tree establish the setting. The feeling is an ordinary workday, "
        "not tombs, treasure or royalty. Clean wide composition, no text.",
    ),
]


@app.local_entrypoint()
def storyboard(output_dir: str = "storyboard-output") -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    overall_start = time.perf_counter()

    # A) Generate anchor, then all scratch-generation scenes while Generator stays hot.
    gen = Generator()
    t0 = time.perf_counter()
    anchor = gen.generate.remote(CONCEPT_PROMPT, WIDTH, HEIGHT, 33001)
    anchor_seconds = time.perf_counter() - t0
    anchor_path = target / "01-concept-anchor.png"
    anchor_path.write_bytes(anchor)
    print(f"HYBRID_IMAGE index=1 mode=anchor seconds={anchor_seconds:.3f} bytes={len(anchor)} path={anchor_path}")

    generated_durations: list[float] = []
    edit_durations: list[float] = []
    records = [{"index": 1, "slug": "concept-anchor", "mode": "anchor", "seconds": anchor_seconds, "bytes": len(anchor)}]

    for index, slug, scene in GENERATE_SCENES:
        prompt = STYLE_BIBLE + "\n\nSCENE:\n" + scene
        started = time.perf_counter()
        data = gen.generate.remote(prompt, WIDTH, HEIGHT, 33000 + index)
        elapsed = time.perf_counter() - started
        generated_durations.append(elapsed)
        path = target / f"{index:02d}-{slug}.png"
        path.write_bytes(data)
        records.append({"index": index, "slug": slug, "mode": "generate", "seconds": elapsed, "bytes": len(data)})
        print(f"HYBRID_IMAGE index={index} mode=generate slug={slug} seconds={elapsed:.3f} bytes={len(data)} path={path}")

    # B) Ramses scenes use Edit from the SAME original anchor. Editor gets one cold start, then stays hot.
    editor = Editor()
    for index, slug, scene in EDIT_SCENES:
        prompt = RAMSES_LOCK + "\n\n" + STYLE_BIBLE + "\n\nNEW SCENE:\n" + scene
        started = time.perf_counter()
        data = editor.edit.remote(anchor, prompt, 1024, 33000 + index)
        elapsed = time.perf_counter() - started
        edit_durations.append(elapsed)
        path = target / f"{index:02d}-{slug}.png"
        path.write_bytes(data)
        records.append({"index": index, "slug": slug, "mode": "edit", "seconds": elapsed, "bytes": len(data)})
        print(f"HYBRID_IMAGE index={index} mode=edit slug={slug} seconds={elapsed:.3f} bytes={len(data)} path={path}")

    records.sort(key=lambda r: r["index"])
    wall_seconds = time.perf_counter() - overall_start
    estimated_gpu_usd = wall_seconds * L4_GPU_USD_PER_SECOND

    result = {
        "experiment": "hybrid-style-consistency-v2",
        "gpu": "L4",
        "resolution_requested": f"{WIDTH}x{HEIGHT}",
        "steps": 4,
        "image_count": 10,
        "anchor_seconds": anchor_seconds,
        "generator_warm_average_seconds": statistics.mean(generated_durations),
        "generator_warm_min_seconds": min(generated_durations),
        "generator_warm_max_seconds": max(generated_durations),
        "editor_first_call_seconds": edit_durations[0],
        "editor_warm_average_seconds": statistics.mean(edit_durations[1:]),
        "editor_warm_min_seconds": min(edit_durations[1:]),
        "editor_warm_max_seconds": max(edit_durations[1:]),
        "total_wall_seconds": wall_seconds,
        "l4_gpu_usd_per_second": L4_GPU_USD_PER_SECOND,
        "estimated_gpu_usd_upper_bound": estimated_gpu_usd,
        "estimated_usd_per_image_upper_bound": estimated_gpu_usd / 10.0,
        "estimated_images_per_30_upper_bound_basis": 30.0 / (estimated_gpu_usd / 10.0),
        "routing": {
            "generate_from_scratch": [x[0] for x in GENERATE_SCENES],
            "edit_from_anchor": [x[0] for x in EDIT_SCENES],
        },
        "scenes": records,
        "note": "Generation scenes intentionally do not consume the anchor image; consistency comes from a strict shared style bible. Ramses scenes use the anchor with Mage Edit. Cost estimate is end-to-end wall time times published L4 rate and is conservative rather than exact billing telemetry.",
    }
    (target / "storyboard-benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("HYBRID_STORYBOARD_JSON=" + json.dumps(result, separators=(",", ":")))
