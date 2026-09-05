from __future__ import annotations

import io
import json
import re
import wave
from pathlib import Path

from qwen3_tts_modal import PresetTTS, app

CONTROL_CUE_RE = re.compile(r"(?m)^\s*\[[A-Z][A-Z0-9 ,.'’/&+\-]*\]\s*")
TAG_RE = re.compile(r"</?[^>]+>")


def speakable_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n")
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end < 0:
            raise ValueError("front matter has no closing --- line")
        text = text[end + 5 :]
    text = CONTROL_CUE_RE.sub("", text)
    text = TAG_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def wav_info(data: bytes) -> dict[str, int | float]:
    with wave.open(io.BytesIO(data), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return {
            "channels": wf.getnchannels(),
            "sample_width": wf.getsampwidth(),
            "sample_rate": rate,
            "frames": frames,
            "duration_seconds": frames / rate,
        }


def concat_wavs(parts: list[bytes], gap_ms: int) -> tuple[bytes, list[dict[str, float | int]]]:
    if not parts:
        raise ValueError("no WAV parts")

    decoded: list[tuple[dict[str, int | float], bytes]] = []
    for part in parts:
        meta = wav_info(part)
        with wave.open(io.BytesIO(part), "rb") as wf:
            pcm = wf.readframes(wf.getnframes())
        decoded.append((meta, pcm))

    first = decoded[0][0]
    key = (first["channels"], first["sample_width"], first["sample_rate"])
    for meta, _ in decoded[1:]:
        if (meta["channels"], meta["sample_width"], meta["sample_rate"]) != key:
            raise ValueError("Qwen returned incompatible WAV formats")

    rate = int(first["sample_rate"])
    channels = int(first["channels"])
    width = int(first["sample_width"])
    gap_frames = int(rate * max(0, gap_ms) / 1000)
    silence = b"\x00" * gap_frames * channels * width

    timeline: list[dict[str, float | int]] = []
    cursor_frames = 0
    chunks: list[bytes] = []
    for index, (meta, pcm) in enumerate(decoded):
        frames = int(meta["frames"])
        timeline.append({
            "index": index + 1,
            "start_seconds": cursor_frames / rate,
            "duration_seconds": frames / rate,
            "end_seconds": (cursor_frames + frames) / rate,
        })
        chunks.append(pcm)
        cursor_frames += frames
        if index != len(decoded) - 1:
            chunks.append(silence)
            cursor_frames += gap_frames

    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        wf.writeframes(b"".join(chunks))
    return out.getvalue(), timeline


@app.local_entrypoint()
def render_video_factory_job(
    input_dir: str,
    output_dir: str,
    speaker: str = "Aiden",
    language: str = "English",
    gap_ms: int = 300,
) -> None:
    source = Path(input_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".md"})
    if not files:
        raise SystemExit(f"No transcript files found in {source}")

    renderer = PresetTTS()
    wav_parts: list[bytes] = []
    manifest_parts: list[dict[str, object]] = []

    for index, path in enumerate(files, start=1):
        text = speakable_text(path.read_text(encoding="utf-8"))
        if not text:
            raise SystemExit(f"No speakable text in {path}")
        if len(text) > 2400:
            raise SystemExit(f"Transcript part exceeds Qwen 2400-char limit: {path}")

        print(f"Rendering {index}/{len(files)}: {path.name} ({len(text)} chars) speaker={speaker}")
        data = renderer.synthesize.remote(text, speaker, language)
        info = wav_info(data)
        output_path = target / f"{path.stem}.wav"
        output_path.write_bytes(data)
        wav_parts.append(data)
        manifest_parts.append({
            "index": index,
            "source": path.name,
            "output": output_path.name,
            "text": text,
            **info,
        })

    master, timeline = concat_wavs(wav_parts, gap_ms=gap_ms)
    (target / "master.wav").write_bytes(master)
    master_info = wav_info(master)

    for part, timing in zip(manifest_parts, timeline):
        part["start_seconds"] = timing["start_seconds"]
        part["end_seconds"] = timing["end_seconds"]

    manifest = {
        "schema": 1,
        "provider": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "speaker": speaker,
        "language": language,
        "gap_ms": gap_ms,
        "parts": manifest_parts,
        "master": {"file": "master.wav", **master_info},
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Rendered {len(files)} parts; master duration={master_info['duration_seconds']:.3f}s")
