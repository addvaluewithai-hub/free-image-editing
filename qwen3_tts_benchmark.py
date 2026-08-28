from __future__ import annotations

import io
import json
import statistics
import time
import wave
from pathlib import Path

from qwen3_tts_modal import Qwen3TTS, app

L4_GPU_USD_PER_SECOND = 0.000222
SPEAKER = "Ryan"
LANGUAGE = "English"

SCENES = [
    (
        "01-ramses-intro",
        "Ramses the Second. That name alone meant power. He ruled Egypt for sixty-six years, built Abu Simbel, and became one of the biggest names in ancient history.",
        "Confident cinematic documentary narration, warm and natural, with a slight sense of awe. Strong opening emphasis, but do not sound theatrical.",
    ),
    (
        "02-peace-treaty",
        "One of his biggest chapters was the conflict with the Hittites. After a massive war, the two powers eventually reached one of the earliest known peace treaties in history.",
        "Authoritative documentary tone, measured pace, clear diction, with subtle historical gravitas.",
    ),
    (
        "03-long-life",
        "And here is the surprising part. Ramses lived to around ninety years old. Ninety. In the ancient world, that was extraordinary.",
        "Sound genuinely surprised and impressed. Slow down slightly on the word ninety, then return to a natural pace.",
    ),
    (
        "04-ct-scan",
        "Now jump to twenty sixteen. A team led by Doctor Sahar Saleem and Doctor Zahi Hawass used a CT scan to examine the mummy of Ramses in remarkable detail without damaging it.",
        "Curious investigative documentary tone, modern and precise, with natural pacing.",
    ),
    (
        "05-ct-explainer",
        "A CT scan works by taking image after image, slice by slice, and combining them into a detailed three-dimensional view. It lets researchers inspect bones and teeth without surgery.",
        "Friendly science explainer. Clear, precise, slightly slower than normal, but still conversational.",
    ),
    (
        "06-dental-damage",
        "Then came the surprise. His teeth showed severe wear, several were missing, and there was obvious damage around the jaw. This was not just a minor toothache.",
        "Serious and controlled. Add a short dramatic pause before the final sentence without becoming overly dramatic.",
    ),
    (
        "07-abscess",
        "There was a large abscess near the root of a tooth on the left side of his jaw. That means infection and pressure around the root and bone, probably causing persistent pain.",
        "Tense medical storytelling, empathetic and restrained. Emphasize the painful detail naturally.",
    ),
    (
        "08-powerful-king-in-pain",
        "Think about the contrast. One of the most powerful men in the world had an empire, gold, doctors, and priests, yet he could still be defeated by a painful tooth. That is the plot twist.",
        "Lightly ironic storytelling with a believable sense of disbelief. Keep the same narrator identity.",
    ),
    (
        "09-king-vs-farmer",
        "And the same kind of dental problem could affect the poorest farmer in the Nile Delta. A king on a throne and a farmer in a field. Different lives, same teeth problem. So why did this happen?",
        "Conversational and reflective. Make the comparison feel surprising, then end the question with curiosity.",
    ),
    (
        "10-daily-life",
        "To answer that, we need to leave the tombs and the gold behind and look at daily life. What did ancient Egyptians eat? How did they work? What annoyed them? That is where the next part of the story begins.",
        "Playful, curious ending with a sense of anticipation. Warm documentary narrator, natural pace.",
    ),
]


def wav_duration_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


@app.local_entrypoint()
def benchmark(output_dir: str = "qwen3-tts-output") -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    tts = Qwen3TTS()
    records = []
    generation_times = []
    audio_durations = []
    overall_start = time.perf_counter()

    for index, (slug, text, instruct) in enumerate(SCENES, start=1):
        started = time.perf_counter()
        audio = tts.synthesize.remote(text, SPEAKER, instruct, LANGUAGE)
        generation_seconds = time.perf_counter() - started
        audio_seconds = wav_duration_seconds(audio)
        rtf = generation_seconds / audio_seconds if audio_seconds else None
        generation_times.append(generation_seconds)
        audio_durations.append(audio_seconds)

        path = target / f"{index:02d}-{slug}.wav"
        path.write_bytes(audio)
        records.append(
            {
                "index": index,
                "slug": slug,
                "speaker": SPEAKER,
                "language": LANGUAGE,
                "text": text,
                "instruct": instruct,
                "generation_seconds": generation_seconds,
                "audio_seconds": audio_seconds,
                "real_time_factor": rtf,
                "bytes": len(audio),
            }
        )
        print(
            f"QWEN3_TTS_SCENE index={index} slug={slug} speaker={SPEAKER} "
            f"generation_seconds={generation_seconds:.3f} audio_seconds={audio_seconds:.3f} "
            f"rtf={rtf:.3f} bytes={len(audio)}"
        )

    total_wall = time.perf_counter() - overall_start
    warm = records[1:]
    warm_generation = [r["generation_seconds"] for r in warm]
    warm_audio = [r["audio_seconds"] for r in warm]
    warm_total_gen = sum(warm_generation)
    warm_total_audio = sum(warm_audio)
    warm_rtf = warm_total_gen / warm_total_audio if warm_total_audio else None

    gpu_raw = total_wall * L4_GPU_USD_PER_SECOND
    gpu_with_tail = (total_wall + 30.0) * L4_GPU_USD_PER_SECOND
    warm_cost_per_audio_min = (
        warm_rtf * 60.0 * L4_GPU_USD_PER_SECOND if warm_rtf is not None else None
    )

    result = {
        "experiment": "qwen3-tts-06b-customvoice-english-ramses-10-scene",
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "gpu": "L4",
        "attention": "sdpa",
        "speaker": SPEAKER,
        "language": LANGUAGE,
        "scene_count": len(SCENES),
        "first_call_seconds_including_cold_start": generation_times[0],
        "warm_average_generation_seconds": statistics.mean(warm_generation),
        "warm_min_generation_seconds": min(warm_generation),
        "warm_max_generation_seconds": max(warm_generation),
        "total_generation_wall_seconds": total_wall,
        "total_audio_seconds": sum(audio_durations),
        "warm_real_time_factor": warm_rtf,
        "l4_gpu_usd_per_second": L4_GPU_USD_PER_SECOND,
        "estimated_gpu_usd_raw_wall": gpu_raw,
        "estimated_gpu_usd_with_30s_scaledown": gpu_with_tail,
        "estimated_warm_gpu_usd_per_audio_minute": warm_cost_per_audio_min,
        "estimated_warm_audio_minutes_per_30_usd": (
            30.0 / warm_cost_per_audio_min if warm_cost_per_audio_min else None
        ),
        "scenes": records,
        "note": "Ten sequential English calls use the same built-in English speaker Ryan with scene-specific natural-language style instructions. The first remote call includes container/model cold start. GPU estimates use the L4 rate only.",
    }
    (target / "benchmark.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("QWEN3_TTS_BENCHMARK_JSON=" + json.dumps(result, separators=(",", ":")))
