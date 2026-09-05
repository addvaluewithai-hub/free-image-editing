from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests


def _headers() -> dict[str, str]:
    token_id = os.environ.get("MODAL_PROXY_TOKEN_ID")
    token_secret = os.environ.get("MODAL_PROXY_TOKEN_SECRET")
    if not token_id or not token_secret:
        raise SystemExit(
            "Set MODAL_PROXY_TOKEN_ID and MODAL_PROXY_TOKEN_SECRET first. "
            "Create them with: modal workspace proxy-tokens create"
        )
    return {"Authorization": f"Bearer {token_id}.{token_secret}"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the private Qwen3-TTS 0.6B Modal API")
    parser.add_argument("--url", default=os.environ.get("QWEN_TTS_API_URL"))
    sub = parser.add_subparsers(dest="command", required=True)

    tts = sub.add_parser("tts", help="Generate with a built-in CustomVoice speaker")
    tts.add_argument("text")
    tts.add_argument("--speaker", default="Ryan")
    tts.add_argument("--language", default="English")
    tts.add_argument("--out", default="tts.wav")

    clone = sub.add_parser("clone", help="Clone a voice with the 0.6B Base model")
    clone.add_argument("reference")
    clone.add_argument("text")
    clone.add_argument("--reference-text")
    clone.add_argument("--language", default="English")
    clone.add_argument("--x-vector-only", action="store_true")
    clone.add_argument("--out", default="clone.wav")

    args = parser.parse_args()
    if not args.url:
        raise SystemExit("Pass --url or set QWEN_TTS_API_URL to your Modal API URL")

    base = args.url.rstrip("/")
    headers = _headers()

    if args.command == "tts":
        response = requests.post(
            f"{base}/tts",
            headers={**headers, "Content-Type": "application/json"},
            json={
                "text": args.text,
                "speaker": args.speaker,
                "language": args.language,
            },
            timeout=900,
        )
    else:
        if not args.x_vector_only and not args.reference_text:
            raise SystemExit(
                "--reference-text is required for high-fidelity cloning. "
                "Use --x-vector-only if no transcript is available."
            )
        reference = Path(args.reference)
        with reference.open("rb") as f:
            response = requests.post(
                f"{base}/clone",
                headers=headers,
                files={"reference": (reference.name, f)},
                data={
                    "text": args.text,
                    "reference_text": args.reference_text or "",
                    "language": args.language,
                    "x_vector_only": "true" if args.x_vector_only else "false",
                },
                timeout=900,
            )

    if not response.ok:
        raise SystemExit(f"HTTP {response.status_code}: {response.text}")

    Path(args.out).write_bytes(response.content)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
