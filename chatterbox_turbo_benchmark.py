from __future__ import annotations

import io
import json
import time
import wave
from pathlib import Path

from chatterbox_turbo_modal import ChatterboxTurbo, app

L4_GPU_USD_PER_SECOND = 0.000222
SILENCE_BETWEEN_CHUNKS_SECONDS = 0.28

# Same Northstar story used in the Qwen/Vox long-form comparison, now with
# Chatterbox Turbo's native paralinguistic tags placed where they make sense.
# Chunks are intentional: Turbo is optimized for lower-latency speech segments,
# and this is the production-style path we would actually ship for long-form work.
CHUNKS = [
    "At 2:17 in the morning, the Northstar Deep Space Array went silent. No alarms. No explosion. Just twelve enormous antennas, frozen under a moonless sky, suddenly pointing at nothing.",
    "For engineer Elena Marquez, that silence was worse than noise. Three minutes earlier, Northstar had been tracking a probe two hundred and eighty million miles from Earth. Its signal was weak, barely more than a whisper in the static, but it was there. Then - gone. [gasp] Elena stared at the telemetry and said the one thing nobody in the control room wanted to hear: That wasn't the probe.",
    "At first, the team blamed the weather. Easy answer. Comfortable answer. Also... completely wrong. Wind speed was normal. Power was stable. The cryogenic receivers were cold, the clocks were synchronized, and every diagnostic light was green. Perfect. [sigh] Which is exactly when you start to worry.",
    "Then a junior technician noticed something strange. Antenna seven had moved by zero point three degrees. That sounds tiny. It isn't. At that distance, zero point three degrees is the difference between listening to a spacecraft... and listening to empty space.",
    "Elena leaned toward the console. Run the command history again. This time her voice was lower. Slower. Nobody joked. Nobody moved. The log showed a steering command at 2:16:42 a.m. It had valid credentials. It had passed every security check. And according to the system, Elena herself had sent it. She hadn't.",
    "Now the room changed. Curiosity became tension. Tension became fear. Someone killed the external network link. Another engineer started reading the command stack line by line. And then, buried between two routine calibration messages, they found it: a malformed packet with a timestamp from tomorrow. [gasp] Tomorrow.",
    "For about five seconds, the entire room just stared at the screen. Then Marcus, the systems lead, broke the silence: Great. So either we've been hacked by a time traveler... or our clock is lying to us. [chuckle] A few people laughed. [laugh] Not because it was funny. Because sometimes your brain needs somewhere to put the panic.",
    "The clock wasn't lying. But one backup server was. A firmware bug had pushed its date forward by exactly twenty-four hours, causing it to replay an old steering command as if it were new. No attacker. No sabotage. No science-fiction mystery. Just one tiny software error, hiding inside a machine that had worked perfectly for six years.",
    "And here is the part that still bothers Elena. [sigh] The probe's signal returned at 2:29 a.m. - twelve minutes after it vanished. Twelve minutes doesn't sound like much. But when your spacecraft is millions of miles away, twelve minutes feels enormous. You cannot walk outside and fix it. You cannot restart the universe.",
    "You can only send a command into the dark... and wait. [clear throat] At 2:31, the first clean telemetry packet arrived. Battery voltage: normal. Guidance: normal. Memory: intact. [gasp] The room erupted.",
    "One technician actually hugged a printer. Marcus claimed he had never been worried, which was a spectacular lie. [chuckle] By sunrise, Northstar was tracking the probe again, the faulty server was isolated, and somebody had written NO TIME TRAVEL on the whiteboard in red marker.",
    "But the lesson wasn't really about a bad clock. It was about confidence. Complex systems rarely fail with dramatic sparks and smoke. Sometimes they fail quietly, politely, with every status light glowing green. And the most dangerous sentence in any control room might be the simplest one: It can't be that.",
    "Because sometimes... [sigh] it absolutely can.",
]


def wav_info(data: bytes) -> dict:
    with wave.open(io.BytesIO(data), "rb") as wf:
        return {
            "channels": wf.getnchannels(),
            "sample_width": wf.getsampwidth(),
            "sample_rate": wf.getframerate(),
            "frames": wf.getnframes(),
            "audio_seconds": wf.getnframes() / float(wf.getframerate()),
        }


def combine_wavs(wavs: list[bytes], silence_seconds: float) -> bytes:
    if not wavs:
        raise ValueError("No WAVs to combine")

    infos = [wav_info(data) for data in wavs]
    first = infos[0]
    for info in infos[1:]:
        if (
            info["channels"] != first["channels"]
            or info["sample_width"] != first["sample_width"]
            or info["sample_rate"] != first["sample_rate"]
        ):
            raise RuntimeError("Generated WAV formats do not match")

    silence_frames = int(first["sample_rate"] * silence_seconds)
    silence = b"\x00" * silence_frames * first["channels"] * first["sample_width"]

    out = io.BytesIO()
    with wave.open(out, "wb") as dst:
        dst.setnchannels(first["channels"])
        dst.setsampwidth(first["sample_width"])
        dst.setframerate(first["sample_rate"])
        for index, data in enumerate(wavs):
            with wave.open(io.BytesIO(data), "rb") as src:
                dst.writeframes(src.readframes(src.getnframes()))
            if index != len(wavs) - 1:
                dst.writeframes(silence)
    return out.getvalue()


def run_pass(tts: ChatterboxTurbo, target: Path, label: str, seed_base: int) -> dict:
    chunk_records = []
    wavs = []
    pass_started = time.perf_counter()

    for index, text in enumerate(CHUNKS, start=1):
        seed = seed_base + index
        started = time.perf_counter()
        audio = tts.synthesize.remote(text, seed)
        generation_seconds = time.perf_counter() - started
        info = wav_info(audio)
        wavs.append(audio)

        chunk_path = target / f"{label}-{index:02d}.wav"
        chunk_path.write_bytes(audio)
        record = {
            "index": index,
            "text": text,
            "characters": len(text),
            "seed": seed,
            "generation_seconds": generation_seconds,
            **info,
            "real_time_factor": generation_seconds / info["audio_seconds"]
            if info["audio_seconds"]
            else None,
        }
        chunk_records.append(record)
        print(
            f"CHATTERBOX_CHUNK pass={label} index={index} chars={len(text)} "
            f"generation_seconds={generation_seconds:.3f} "
            f"audio_seconds={info['audio_seconds']:.3f} "
            f"rtf={record['real_time_factor']:.3f}"
        )

    pass_wall = time.perf_counter() - pass_started
    combined = combine_wavs(wavs, SILENCE_BETWEEN_CHUNKS_SECONDS)
    combined_path = target / f"chatterbox-turbo-{label}-northstar-expressive.wav"
    combined_path.write_bytes(combined)
    combined_info = wav_info(combined)
    generated_audio_seconds = sum(r["audio_seconds"] for r in chunk_records)
    generation_seconds_sum = sum(r["generation_seconds"] for r in chunk_records)

    return {
        "label": label,
        "chunk_count": len(CHUNKS),
        "generation_seconds_sum": generation_seconds_sum,
        "pass_wall_seconds": pass_wall,
        "generated_audio_seconds_excluding_inserted_silence": generated_audio_seconds,
        "combined_audio_seconds_including_inserted_silence": combined_info["audio_seconds"],
        "real_time_factor_excluding_inserted_silence": generation_seconds_sum / generated_audio_seconds,
        "combined_wav": str(combined_path),
        "chunks": chunk_records,
    }


@app.local_entrypoint()
def benchmark(output_dir: str = "chatterbox-turbo-output") -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    (target / "script.txt").write_text("\n\n".join(CHUNKS), encoding="utf-8")

    tts = ChatterboxTurbo()

    # Pass one includes the model/container cold start on its first chunk.
    cold = run_pass(tts, target, "cold", seed_base=7000)
    # Pass two immediately repeats the exact same workload on the warm container.
    warm = run_pass(tts, target, "warm", seed_base=7000)

    warm_rtf = warm["real_time_factor_excluding_inserted_silence"]
    warm_cost_per_audio_minute = warm_rtf * 60.0 * L4_GPU_USD_PER_SECOND
    total_wall = cold["pass_wall_seconds"] + warm["pass_wall_seconds"]

    result = {
        "experiment": "chatterbox-turbo-english-northstar-expressive-chunked-cold-warm",
        "model": "ResembleAI/chatterbox-turbo",
        "model_size": "350M",
        "license": "MIT",
        "gpu": "L4",
        "mode": "production-style sequential chunks with native paralinguistic tags",
        "native_tags_used": ["[gasp]", "[sigh]", "[chuckle]", "[laugh]", "[clear throat]"],
        "chunk_count": len(CHUNKS),
        "script_characters_with_tags": sum(len(x) for x in CHUNKS),
        "l4_gpu_usd_per_second": L4_GPU_USD_PER_SECOND,
        "cold": cold,
        "warm": warm,
        "estimated_warm_gpu_usd_per_audio_minute": warm_cost_per_audio_minute,
        "estimated_warm_audio_minutes_per_30_usd": 30.0 / warm_cost_per_audio_minute,
        "estimated_two_pass_gpu_usd_raw_wall": total_wall * L4_GPU_USD_PER_SECOND,
        "estimated_two_pass_gpu_usd_with_30s_tail": (total_wall + 30.0)
        * L4_GPU_USD_PER_SECOND,
        "note": (
            "Cold pass and warm pass use the same text and per-chunk seeds. The first cold chunk includes "
            "container/model startup. Long-form audio is assembled from short sequential Turbo generations, "
            "which is the intended production test here. Cost estimates are L4 GPU-only and exclude storage/network."
        ),
    }

    (target / "benchmark.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("CHATTERBOX_BENCHMARK_JSON=" + json.dumps(result, separators=(",", ":")))
