from __future__ import annotations

import io
import json
import time
import wave
from pathlib import Path

from english_expression_script import QWEN_INSTRUCT, SCRIPT
from qwen3_tts_modal import Qwen3TTS, app

L4_GPU_USD_PER_SECOND = 0.000222
SPEAKER = "Ryan"
LANGUAGE = "English"


def wav_duration_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


@app.local_entrypoint()
def benchmark(output_dir: str = "qwen3-oneshot-output") -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "input-script.txt").write_text(SCRIPT, encoding="utf-8")
    (target / "style-instruction.txt").write_text(QWEN_INSTRUCT, encoding="utf-8")

    tts = Qwen3TTS()
    runs = []

    for label in ("cold", "warm"):
        started = time.perf_counter()
        audio = tts.synthesize.remote(SCRIPT, SPEAKER, QWEN_INSTRUCT, LANGUAGE)
        elapsed = time.perf_counter() - started
        audio_seconds = wav_duration_seconds(audio)
        rtf = elapsed / audio_seconds if audio_seconds else None
        (target / f"qwen3-{label}-oneshot.wav").write_bytes(audio)
        row = {
            "label": label,
            "generation_seconds": elapsed,
            "audio_seconds": audio_seconds,
            "real_time_factor": rtf,
            "bytes": len(audio),
        }
        runs.append(row)
        print(
            f"QWEN3_ONESHOT label={label} generation_seconds={elapsed:.3f} "
            f"audio_seconds={audio_seconds:.3f} rtf={rtf:.3f} bytes={len(audio)}"
        )

    cold, warm = runs
    warm_cost_per_audio_minute = warm["real_time_factor"] * 60.0 * L4_GPU_USD_PER_SECOND
    result = {
        "experiment": "qwen3-tts-06b-customvoice-long-english-one-shot-expressions",
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "gpu": "L4",
        "attention": "sdpa",
        "speaker": SPEAKER,
        "language": LANGUAGE,
        "request_text_characters": len(SCRIPT),
        "request_text_words": len(SCRIPT.split()),
        "request_count": 2,
        "same_full_script_per_request": True,
        "runs": runs,
        "cold_estimated_gpu_usd": cold["generation_seconds"] * L4_GPU_USD_PER_SECOND,
        "warm_estimated_gpu_usd": warm["generation_seconds"] * L4_GPU_USD_PER_SECOND,
        "warm_estimated_gpu_usd_per_audio_minute": warm_cost_per_audio_minute,
        "warm_estimated_audio_minutes_per_30_usd": 30.0 / warm_cost_per_audio_minute,
        "estimated_two_request_gpu_usd_plus_30s_tail": (
            cold["generation_seconds"] + warm["generation_seconds"] + 30.0
        ) * L4_GPU_USD_PER_SECOND,
        "style_instruction": QWEN_INSTRUCT,
        "note": (
            "Both calls send the entire long English script in one generate_custom_voice request. "
            "The first call measures cold start plus generation; the second measures warm one-shot throughput. "
            "No FlashAttention is enabled so this isolates the one-shot effect versus the previous SDPA benchmark."
        ),
    }
    (target / "benchmark.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("QWEN3_ONESHOT_BENCHMARK_JSON=" + json.dumps(result, separators=(",", ":")))
