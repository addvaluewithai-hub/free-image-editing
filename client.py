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
    parser = argparse.ArgumentParser(description="Call the private Mage-Flow Modal API")
    parser.add_argument("--url", default=os.environ.get("MAGE_API_URL"))
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("prompt")
    gen.add_argument("--width", type=int, default=1024)
    gen.add_argument("--height", type=int, default=1024)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--out", default="generated.png")

    edit = sub.add_parser("edit")
    edit.add_argument("image")
    edit.add_argument("prompt")
    edit.add_argument("--max-size", type=int, default=1024)
    edit.add_argument("--seed", type=int, default=42)
    edit.add_argument("--out", default="edited.png")

    args = parser.parse_args()
    if not args.url:
        raise SystemExit("Pass --url or set MAGE_API_URL to your Modal API URL")

    base = args.url.rstrip("/")
    headers = _headers()

    if args.command == "generate":
        response = requests.post(
            f"{base}/generate",
            headers={**headers, "Content-Type": "application/json"},
            json={
                "prompt": args.prompt,
                "width": args.width,
                "height": args.height,
                "seed": args.seed,
            },
            timeout=600,
        )
    else:
        with open(args.image, "rb") as f:
            response = requests.post(
                f"{base}/edit",
                headers=headers,
                files={"image": (Path(args.image).name, f)},
                data={
                    "prompt": args.prompt,
                    "max_size": str(args.max_size),
                    "seed": str(args.seed),
                },
                timeout=600,
            )

    response.raise_for_status()
    Path(args.out).write_bytes(response.content)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
