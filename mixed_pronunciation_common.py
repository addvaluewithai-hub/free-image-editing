from __future__ import annotations

import io
import json
import time
import wave
from pathlib import Path

L4_GPU_USD_PER_SECOND = 0.000222

# User-provided lesson text. Only Markdown escaping was removed so these are real
# XML/SSML-looking tags rather than literal backslash-escaped angle brackets.
CHUNKS = [
    """أهلًا بيك في أول درس.\nتخيّل إنك قابلت شخص جديد، وبعد السلام سألك:\n<lang xml:lang=\"en-US\"><phoneme alphabet=\"ipa\" ph=\"tɛl mi əˈbaʊt jɚˈsɛlf\">Tell me about yourself.</phoneme></lang>""",
    """اسمع النموذج ده أولًا.\n<lang xml:lang=\"en-US\"><phoneme alphabet=\"ipa\" ph=\"haɪ | aɪm ˈmaɪə | aɪm frəm ˈdʒɔɹdən | aɪ lɪv ɪn əˈmɑːn | aɪm ə dɪˈzaɪnɚ | aɪ laɪk fəˈtɑgɹəfi\">Hi, I'm Maya. I'm from Jordan. I live in Amman. I'm a designer. I like photography.</phoneme></lang>""",
    """بس كده\nأول حاجة، التحية والاسم.\n<lang xml:lang=\"en-US\"><phoneme alphabet=\"ipa\" ph=\"haɪ | aɪm ˈmaɪə\">Hi, I'm Maya.</phoneme></lang>""",
    """كلمة <lang xml:lang=\"en-US\"><phoneme alphabet=\"ipa\" ph=\"aɪm\">I'm</phoneme></lang> هي الصورة المختصرة والطبيعية من:\n<lang xml:lang=\"en-US\"><phoneme alphabet=\"ipa\" ph=\"aɪ æm\">I am.</phoneme></lang>""",
    """لكن في المحادثات العادية، الأغلب إنك هتسمع وتستخدم:""",
]

MAIN_SCRIPT = "\n\n".join(CHUNKS)

# Deliberately mismatched display text vs IPA. This is a separate diagnostic,
# not part of the user's lesson. A real SSML/IPA implementation should say
# "red" rather than "green" here.
IPA_CONTROL = '<lang xml:lang="en-US"><phoneme alphabet="ipa" ph="rɛd">GREEN</phoneme></lang>'


def wav_duration_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def join_wavs(parts: list[bytes], gap_ms: int = 280) -> bytes:
    if not parts:
        raise ValueError("No WAV parts to join")

    readers = [wave.open(io.BytesIO(part), "rb") for part in parts]
    try:
        first = readers[0]
        params = (first.getnchannels(), first.getsampwidth(), first.getframerate())
        for reader in readers[1:]:
            p = (reader.getnchannels(), reader.getsampwidth(), reader.getframerate())
            if p != params:
                raise ValueError(f"WAV format mismatch: {p} != {params}")

        channels, sample_width, sample_rate = params
        silence_frames = int(sample_rate * gap_ms / 1000)
        silence = b"\x00" * silence_frames * channels * sample_width
        out = io.BytesIO()
        with wave.open(out, "wb") as dst:
            dst.setnchannels(channels)
            dst.setsampwidth(sample_width)
            dst.setframerate(sample_rate)
            for index, reader in enumerate(readers):
                if index:
                    dst.writeframes(silence)
                dst.writeframes(reader.readframes(reader.getnframes()))
        return out.getvalue()
    finally:
        for reader in readers:
            reader.close()


def run_benchmark(
    *,
    output_dir: str,
    model_name: str,
    synth_chunk,
    synth_control,
    notes: str,
) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "input-ssml.txt").write_text(MAIN_SCRIPT, encoding="utf-8")
    (target / "ipa-control.txt").write_text(IPA_CONTROL, encoding="utf-8")

    records = []
    audio_parts: list[bytes] = []
    total_start = time.perf_counter()

    for index, text in enumerate(CHUNKS, start=1):
        started = time.perf_counter()
        try:
            audio = synth_chunk(text, index)
            elapsed = time.perf_counter() - started
            duration = wav_duration_seconds(audio)
            path = target / f"{index:02d}-main.wav"
            path.write_bytes(audio)
            audio_parts.append(audio)
            row = {
                "index": index,
                "status": "ok",
                "generation_seconds": elapsed,
                "audio_seconds": duration,
                "real_time_factor": elapsed / duration if duration else None,
                "characters": len(text),
            }
            print(
                f"MIXED_PRONUNCIATION model={model_name} chunk={index} "
                f"seconds={elapsed:.3f} audio={duration:.3f} rtf={row['real_time_factor']:.3f}"
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            row = {
                "index": index,
                "status": "error",
                "generation_seconds": elapsed,
                "characters": len(text),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"MIXED_PRONUNCIATION_ERROR model={model_name} chunk={index} error={exc}")
        records.append(row)

    if audio_parts:
        combined = join_wavs(audio_parts)
        (target / "main-combined.wav").write_bytes(combined)
        combined_audio_seconds = wav_duration_seconds(combined)
    else:
        combined_audio_seconds = None

    control = {"text": IPA_CONTROL}
    started = time.perf_counter()
    try:
        control_audio = synth_control(IPA_CONTROL)
        elapsed = time.perf_counter() - started
        duration = wav_duration_seconds(control_audio)
        (target / "ipa-control.wav").write_bytes(control_audio)
        control.update(
            {
                "status": "ok",
                "generation_seconds": elapsed,
                "audio_seconds": duration,
                "real_time_factor": elapsed / duration if duration else None,
                "expected_if_ssml_ipa_supported": "red",
                "expected_if_phoneme_ignored": "green",
            }
        )
    except Exception as exc:
        control.update(
            {
                "status": "error",
                "generation_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        print(f"MIXED_PRONUNCIATION_CONTROL_ERROR model={model_name} error={exc}")

    total_wall = time.perf_counter() - total_start
    successful_audio = sum(r.get("audio_seconds", 0.0) for r in records if r["status"] == "ok")
    estimated_cost = total_wall * L4_GPU_USD_PER_SECOND
    result = {
        "experiment": "mixed-arabic-english-ssml-ipa-pronunciation",
        "model": model_name,
        "gpu": "L4",
        "main_script_characters": len(MAIN_SCRIPT),
        "chunk_count": len(CHUNKS),
        "successful_chunk_audio_seconds_excluding_join_gaps": successful_audio,
        "combined_audio_seconds_including_join_gaps": combined_audio_seconds,
        "total_wall_seconds_including_control": total_wall,
        "estimated_l4_gpu_usd_from_client_wall": estimated_cost,
        "records": records,
        "ipa_control": control,
        "notes": notes,
        "interpretation": (
            "Listen to ipa-control.wav. If it says RED, the ph=IPA attribute appears to have controlled pronunciation. "
            "If it says GREEN, the phoneme attribute was ignored. If it reads markup/tag names, SSML is unsupported at the input layer."
        ),
    }
    (target / "benchmark.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("MIXED_PRONUNCIATION_BENCHMARK_JSON=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))
