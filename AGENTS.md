# AGENTS.md

This repository is intentionally small. Treat this file as the operating contract for coding agents.

## Current architecture

Only these production model paths are in scope:

- `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` for built-in speakers.
- `Qwen/Qwen3-TTS-12Hz-0.6B-Base` for voice cloning.

Primary implementation: `qwen3_tts_modal.py`.

Do not reintroduce VoxCPM2, VoiceTut, Chatterbox, Qwen 1.7B, image-generation models, or historical benchmark harnesses into the production tree unless the user explicitly asks for a new model comparison or architecture change.

## Parked Modal resources

Historical Modal apps/volumes are intentionally retained for possible future experiments. They are **parked**, not production dependencies.

Known historical TTS resources:

- `egyptian-voice-chat` / `egyptian-voice-chat-models`
- `voicetut-tts-test` (shared the `egyptian-voice-chat-models` volume)
- `chatterbox-turbo-test` / `chatterbox-turbo-models`
- `qwen3-tts-17b-expressions` / `qwen3-tts-17b-models`
- `qwen3-tts-06b-test` (historical app only; its `qwen3-tts-models` volume is actively reused by production)

Known historical image-generation resources recovered from Git history:

- `free-image-editing` / `mage-flow-models` — Mage-Flow Turbo + Mage-Flow Edit Turbo.
- `sd-turbo-simple-images` / `sd-turbo-model-cache` — Stability AI SD-Turbo.
- `gear-t2i-1b-benchmark` / `gear-t2i-models` — GEAR T2I GPIC 1B.
- `hidream-o1-8b-benchmark` / `hidream-o1-models` — HiDream-O1-Image 8B.

The GitHub connector confirms these names from project history, but it does not directly enumerate the live Modal workspace. If one of these resources is needed, verify that it is still present/deployed in Modal before depending on it.

Rules:

1. Do **not** delete parked apps or volumes automatically.
2. Do **not** use parked apps for production traffic unless explicitly requested.
3. Idle Modal apps scale to zero and do not consume compute merely by remaining deployed.
4. Persistent Volume storage is billed separately. At the time this note was written Modal listed `$0.09/GiB/month` with `1 TiB/month` included free; check current pricing before making a future cost-based deletion decision.
5. If inspecting or reviving a parked experiment, recover its code from Git history rather than re-polluting the current production tree by default.

## Capability rules

1. The 0.6B CustomVoice path does **not** have reliable natural-language style/instruction control. Do not add an `instruct`, `style`, `emotion`, or similar API parameter and pretend it works.
2. Do not send SSML or IPA markup to Qwen3-TTS. This service does not parse `<lang>`, `<phoneme>`, or IPA attributes.
3. Pronunciation overrides must happen upstream with normal speakable text or a pronunciation dictionary.
4. Arabic is outside the supported production language list for this Qwen3-TTS service. Mixed Arabic/English documents must be segmented before TTS; do not silently feed Arabic spans to this API.
5. Keep requests bounded. Long content must be chunked upstream rather than increasing timeouts/character limits casually.

## Choosing the endpoint

Use `POST /tts` when a built-in voice is acceptable. For English, prefer `Ryan` or `Aiden`.

Use `POST /clone` when the product needs a persistent branded/custom speaker identity. Prefer a clean reference file plus its exact transcript. `x_vector_only=true` is the fallback when no transcript exists.

Do not use the Base clone endpoint merely to change emotion; it is for speaker identity cloning.

## Cost and Modal guardrails

- Keep GPU class at `L4` unless a measured requirement proves it insufficient.
- Keep scale-to-zero behavior.
- Keep `max_containers=1` unless concurrency is explicitly requested and the cost impact is accepted.
- Do not add minimum warm containers by default.
- Deployment is manual through `.github/workflows/deploy-modal.yml`. Do not add `push` deployment triggers without explicit approval.
- Do not create benchmark workflows that run GPUs automatically on ordinary commits.

## Secrets

Expected GitHub Actions secrets:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`
- `HF_TOKEN`

Never print, commit, echo, log, transform, or expose their values. Modal proxy tokens are runtime/client credentials and must also stay out of the repository.

## Model cache

Use the existing Modal volume:

```text
qwen3-tts-models
```

The CustomVoice checkpoint is already associated with this project history. The Base checkpoint should be cached alongside it. Avoid creating another volume for the same 0.6B checkpoints unless migration is deliberate.

## Safe change checklist

Before finishing a code change:

1. Keep the public API capability claims truthful.
2. Do not add unsupported model features.
3. Preserve proxy authentication.
4. Preserve L4/scale-to-zero cost controls.
5. Keep secrets out of diffs and output.
6. Run a syntax check where possible:

```bash
python -m py_compile qwen3_tts_modal.py client.py
```

7. Do not deploy automatically just to validate a documentation/refactor change.
8. Do not delete parked Modal resources without explicit approval.

## Benchmark context

Historical 0.6B CustomVoice baseline on Modal L4, before the cleaned FlashAttention 2 production image:

- warm RTF ~1.74
- ~$0.0232 per generated audio minute
- ~$30 -> ~1,295 audio minutes (~21.6 hours)

Do not present voice-clone cost as measured until the 0.6B Base path receives its own benchmark.

## Design preference

Prefer explicit, boring, production-safe code over adding knobs that the underlying model does not actually support. If a feature requires a different model family or size, surface that tradeoff instead of silently emulating it.
