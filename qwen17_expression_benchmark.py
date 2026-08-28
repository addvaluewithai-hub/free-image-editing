from __future__ import annotations

import io
import json
import time
import wave
from pathlib import Path

from english_expression_script import SCRIPT
from qwen17_tts_modal import Qwen17TTS, app

L4_GPU_USD_PER_SECOND = 0.000222
SPEAKER = "Ryan"
LANGUAGE = "English"
JOIN_SILENCE_MS = 120

P = [part.strip() for part in SCRIPT.split("\n\n") if part.strip()]
if len(P) != 15:
    raise RuntimeError(f"Expected 15 Northstar paragraphs, found {len(P)}")

# Keep the original Northstar wording, but group very short dramatic paragraphs
# with their surrounding scene so the model has enough context for natural delivery.
SCENES = [
    (
        "01-silence",
        P[0],
        "Cinematic documentary narration. Calm, confident, and mysterious. Start controlled and intimate, then let a subtle sense of unease enter by the final phrase. Natural American English, never theatrical.",
    ),
    (
        "02-signal-gone",
        P[1],
        "Warm documentary storytelling with quiet concern. Make the weak signal feel delicate and distant. On 'Then—gone', pause naturally and let genuine worry enter the voice. Keep the same Ryan narrator identity.",
    ),
    (
        "03-green-lights",
        P[2],
        "Dry, intelligent skepticism. The first two 'Easy answer. Comfortable answer.' lines should sound almost reassuring, then turn subtly ironic on 'completely wrong'. Finish with restrained suspicion, not melodrama.",
    ),
    (
        "04-zero-point-three",
        P[3],
        "Curious technical explainer with growing seriousness. Make 'zero point three degrees' sound initially tiny, then emphasize why it matters. Slow slightly on the final contrast between spacecraft and empty space.",
    ),
    (
        "05-command-history",
        P[4] + "\n\n" + P[5],
        "Lower the voice and slow the pace as Elena asks to run the command history again. Build quiet tension through the security checks. After 'Elena herself had sent it', leave a meaningful pause. Deliver 'She hadn't.' as a short, genuinely shocked realization.",
    ),
    (
        "06-tomorrow",
        P[6] + "\n\n" + P[7],
        "Controlled urgency turning into fear. Increase tension as the malformed packet is discovered. On the final word 'tomorrow', sound genuinely startled and disbelieving, then leave a short silence. Do not shout.",
    ),
    (
        "07-time-traveler",
        P[8],
        "Begin in stunned silence. Deliver Marcus's time-traveler line with very dry deadpan humor. Let out a tiny natural restrained chuckle or breath of disbelief after the joke, then immediately return to nervous seriousness. It must feel spontaneous, not like a canned laugh effect.",
    ),
    (
        "08-firmware-bug",
        P[9],
        "Relieved but still thoughtful. Let the repetition 'No attacker. No sabotage. No science-fiction mystery.' land with understated relief and a hint of irony. Emphasize how such a tiny software error could hide for six years.",
    ),
    (
        "09-waiting",
        P[10],
        "Quiet, empathetic, reflective narration. Make twelve minutes feel painfully long. Before 'You can only send a command into the dark', allow a soft audible breath or sigh, then slow down and make the final word 'wait' feel heavy and human.",
    ),
    (
        "10-telemetry-relief",
        P[11],
        "Start focused and tense on the telemetry readings, then let real relief and excitement break through when everything is normal. Smile naturally at the printer joke. Give Marcus's 'never been worried' line playful disbelief, with a very small amused chuckle if it feels natural.",
    ),
    (
        "11-no-time-travel",
        P[12],
        "Warm post-crisis relief. Sound tired but genuinely amused. Let 'NO TIME TRAVEL' land as a subtle inside joke; a light smile or brief natural chuckle is welcome, but keep the documentary narrator believable.",
    ),
    (
        "12-lesson",
        P[13] + "\n\n" + P[14],
        "Calm, thoughtful, memorable closing narration. Pull the energy back and become reflective. Emphasize the danger of false confidence without sounding preachy. Pause before the final sentence, then deliver 'it absolutely can' quietly and with conviction.",
    ),
]


def wav_info(data: bytes) -> dict:
    with wave.open(io.BytesIO(data), "rb") as wf:
        return {
            "channels": wf.getnchannels(),
            "sample_width": wf.getsampwidth(),
            "sample_rate": wf.getframerate(),
            "frames": wf.getnframes(),
            "seconds": wf.getnframes() / float(wf.getframerate()),
            "pcm": wf.readframes(wf.getnframes()),
        }


def combine_wavs(parts: list[bytes], output: Path, silence_ms: int = 120) -> float:
    infos = [wav_info(data) for data in parts]
    first = infos[0]
    for info in infos[1:]:
        for key in ("channels", "sample_width", "sample_rate"):
            if info[key] != first[key]:
                raise RuntimeError(f"WAV format mismatch for {key}")

    silence_frames = int(first["sample_rate"] * silence_ms / 1000.0)
    silence = b"\x00" * silence_frames * first["channels"] * first["sample_width"]

    with wave.open(str(output), "wb") as wf:
        wf.setnchannels(first["channels"])
        wf.setsampwidth(first["sample_width"])
        wf.setframerate(first["sample_rate"])
        for index, info in enumerate(infos):
            if index:
                wf.writeframes(silence)
            wf.writeframes(info["pcm"])

    with wave.open(str(output), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


@app.local_entrypoint()
def benchmark(output_dir: str = "qwen17-expressions-output") -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    tts = Qwen17TTS()

    # Deliberately separate cold-start cost from the production-style warm story pass.
    warmup_text = "Northstar control, audio system online."
    warmup_instruct = "Neutral, natural English system check in the same Ryan voice."
    cold_started = time.perf_counter()
    warmup_audio = tts.synthesize.remote(
        warmup_text, SPEAKER, warmup_instruct, LANGUAGE
    )
    cold_seconds = time.perf_counter() - cold_started
    warmup_path = target / "00-cold-warmup.wav"
    warmup_path.write_bytes(warmup_audio)
    warmup_audio_seconds = wav_info(warmup_audio)["seconds"]
    print(
        f"QWEN17_COLD_WARMUP generation_seconds={cold_seconds:.3f} "
        f"audio_seconds={warmup_audio_seconds:.3f}"
    )

    records = []
    story_wavs: list[bytes] = []
    story_generation_seconds = 0.0
    story_audio_seconds = 0.0

    for index, (slug, text, instruct) in enumerate(SCENES, start=1):
        started = time.perf_counter()
        audio = tts.synthesize.remote(text, SPEAKER, instruct, LANGUAGE)
        generation_seconds = time.perf_counter() - started
        info = wav_info(audio)
        audio_seconds = info["seconds"]
        rtf = generation_seconds / audio_seconds if audio_seconds else None

        story_generation_seconds += generation_seconds
        story_audio_seconds += audio_seconds
        story_wavs.append(audio)

        path = target / f"{index:02d}-{slug}.wav"
        path.write_bytes(audio)
        record = {
            "index": index,
            "slug": slug,
            "speaker": SPEAKER,
            "language": LANGUAGE,
            "text": text,
            "instruct": instruct,
            "characters": len(text),
            "generation_seconds": generation_seconds,
            "audio_seconds": audio_seconds,
            "real_time_factor": rtf,
            "bytes": len(audio),
        }
        records.append(record)
        print(
            f"QWEN17_SCENE index={index} slug={slug} "
            f"generation_seconds={generation_seconds:.3f} "
            f"audio_seconds={audio_seconds:.3f} rtf={rtf:.3f}"
        )

    combined_path = target / "Qwen3-TTS-1.7B-Ryan-Northstar-expressive-warm.wav"
    combined_seconds = combine_wavs(story_wavs, combined_path, JOIN_SILENCE_MS)

    warm_rtf = story_generation_seconds / story_audio_seconds if story_audio_seconds else None
    warm_cost_per_audio_min = (
        warm_rtf * 60.0 * L4_GPU_USD_PER_SECOND if warm_rtf is not None else None
    )
    story_gpu_cost = story_generation_seconds * L4_GPU_USD_PER_SECOND
    session_with_tail_cost = (
        cold_seconds + story_generation_seconds + 30.0
    ) * L4_GPU_USD_PER_SECOND

    result = {
        "experiment": "qwen3-tts-1.7b-customvoice-english-natural-expressions",
        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "gpu": "L4",
        "dtype": "bfloat16",
        "attention": "flash_attention_2",
        "speaker": SPEAKER,
        "language": LANGUAGE,
        "scene_count": len(SCENES),
        "script_characters": len(SCRIPT),
        "script_words": len(SCRIPT.split()),
        "cold_warmup_generation_seconds": cold_seconds,
        "cold_warmup_audio_seconds": warmup_audio_seconds,
        "warm_story_generation_seconds": story_generation_seconds,
        "warm_story_generated_audio_seconds": story_audio_seconds,
        "combined_audio_seconds_including_join_silence": combined_seconds,
        "warm_story_real_time_factor": warm_rtf,
        "l4_gpu_usd_per_second": L4_GPU_USD_PER_SECOND,
        "estimated_story_gpu_usd": story_gpu_cost,
        "estimated_session_gpu_usd_with_cold_and_30s_tail": session_with_tail_cost,
        "estimated_warm_gpu_usd_per_audio_minute": warm_cost_per_audio_min,
        "estimated_warm_audio_minutes_per_30_usd": (
            30.0 / warm_cost_per_audio_min if warm_cost_per_audio_min else None
        ),
        "combined_wav": combined_path.name,
        "scenes": records,
        "note": (
            "Cold-start is measured with a short warm-up utterance. The full Northstar story is then "
            "generated warm in expression-specific chunks using one built-in Ryan identity. "
            "Instructions deliberately request natural shock, breath/sigh, dry humor, subtle chuckles, "
            "relief and reflection rather than bracketed paralinguistic tags. GPU estimates use L4 only."
        ),
    }

    (target / "benchmark.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (target / "script.txt").write_text(SCRIPT, encoding="utf-8")
    print("QWEN17_BENCHMARK_JSON=" + json.dumps(result, separators=(",", ":")))
