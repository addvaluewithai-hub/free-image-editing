# Qwen3-TTS 0.6B on Modal

> The repository name is legacy. The current scope is **text-to-speech only**.

This repository is the production home for a small, cost-conscious Qwen3-TTS stack on Modal. It intentionally supports only two checkpoints:

1. `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` — built-in speakers such as **Ryan** and **Aiden**.
2. `Qwen/Qwen3-TTS-12Hz-0.6B-Base` — **voice cloning** from a reference recording.

Both checkpoints are Apache-2.0 and run on an on-demand Modal **L4** GPU with scale-to-zero.

## Production decisions

The repo has been cleaned of the old experiments. Do not treat Git history as the current architecture.

**Current:** Qwen3-TTS 0.6B CustomVoice + Qwen3-TTS 0.6B Base.

**Removed from the production tree:** VoxCPM2, VoiceTut, Chatterbox, Qwen 1.7B, image-generation experiments, mixed-pronunciation benchmarks, and long-form comparison scripts.

### Parked legacy Modal resources

Some historical Modal apps and model volumes are intentionally left in the workspace for possible future experiments. They are **not production dependencies** and agents must not route production traffic to them unless explicitly asked.

Known parked resources include:

- `egyptian-voice-chat` / `egyptian-voice-chat-models` — historical VoxCPM2 + Egyptian-Qwen work.
- `voicetut-tts-test` — historical VoiceTut-TTS experiment; it used the shared `egyptian-voice-chat-models` volume.
- `chatterbox-turbo-test` / `chatterbox-turbo-models` — historical Chatterbox Turbo experiment.
- `qwen3-tts-17b-expressions` / `qwen3-tts-17b-models` — historical Qwen3-TTS 1.7B expression experiment.
- `qwen3-tts-06b-test` — historical 0.6B test deployment. Its `qwen3-tts-models` volume is **not legacy-only**: the cleaned production app deliberately reuses that volume for the 0.6B checkpoints.

Do not delete parked apps or volumes automatically. Modal compute scales to zero, so an idle deployed app does not consume GPU/CPU compute. Persistent Volume storage is a separate billing dimension: at the time this note was written Modal listed `$0.09/GiB/month` with `1 TiB/month` included free. Re-check current Modal pricing before making cleanup decisions based on cost.

### Important model limitations

- **0.6B CustomVoice does not provide reliable `instruct` / voice-style control.** Do not expose or silently pass style prompts.
- **No SSML or IPA input is supported by this API.** Parse/strip markup before calling TTS.
- If a pronunciation override is required, do it in an upstream text-normalization / pronunciation dictionary layer using ordinary speakable text. Do not send `<phoneme>` tags.
- Qwen3-TTS officially supports Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian. **Arabic is not a supported production language here.** If a source document mixes Arabic and English, segment it upstream and send only supported-language spans to this service.
- Long-form content must be chunked upstream. The API currently caps one request at **2400 characters** to avoid runaway latency/timeouts.

## Files an agent should care about

```text
qwen3_tts_modal.py              # Modal app + private HTTP API
client.py                       # Small CLI client for manual testing
.github/workflows/deploy-modal.yml  # Manual production deployment
AGENTS.md                       # Rules for coding agents
```

Everything else should be documentation/configuration only.

## Models

### Preset voice

Endpoint: `POST /tts`

Default speaker is `Ryan`. For English, the two native-English built-ins are:

- `Ryan` — dynamic male voice with rhythmic drive.
- `Aiden` — sunny American male voice with a clear midrange.

Other upstream built-in speakers are exposed, but using a speaker in its native language is generally the safest quality choice.

Example JSON:

```json
{
  "text": "Tell me about yourself.",
  "speaker": "Aiden",
  "language": "English"
}
```

### Voice cloning

Endpoint: `POST /clone`

Uses `Qwen/Qwen3-TTS-12Hz-0.6B-Base`.

Multipart fields:

- `reference`: reference audio file.
- `text`: target text to synthesize.
- `reference_text`: exact transcript of the reference audio. Recommended for higher-fidelity cloning.
- `language`: default `English`.
- `x_vector_only`: default `false`. Set to `true` only when a transcript is unavailable.

For a reusable branded voice, prefer a clean reference recording and provide the exact transcript. The current API extracts the voice conditioning per request; persistent/cached clone prompts can be added later if repeated-clone benchmarks justify it.

## Modal deployment

### GitHub Actions secrets

The repository expects these Actions secrets:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`
- `HF_TOKEN`

Never put secret values in source files, logs, issues, prompts, or documentation.

### Why deployment is manual

Deployment is intentionally **not triggered on every push**. An agent changing documentation or refactoring code should not automatically start model-download/deployment work and spend credits.

Run the workflow manually from:

```text
Actions -> Deploy Qwen3-TTS 0.6B -> Run workflow
```

The workflow can cache both checkpoints into the existing Modal volume `qwen3-tts-models`, then deploy `qwen3_tts_modal.py`.

Manual CLI equivalent:

```bash
pip install modal==1.5.2
modal setup
modal run qwen3_tts_modal.py::download_models
modal deploy qwen3_tts_modal.py
```

The deployed app name is:

```text
qwen3-tts-06b
```

## Security

The HTTP API uses Modal proxy authentication. GPU endpoints are not intentionally public.

Create a proxy token with:

```bash
modal workspace proxy-tokens create
```

Then configure the client:

```bash
export QWEN_TTS_API_URL='https://YOUR-ENDPOINT.modal.run'
export MODAL_PROXY_TOKEN_ID='wk-...'
export MODAL_PROXY_TOKEN_SECRET='ws-...'
```

Proxy credentials are sent as a Bearer token in the form `wk-....ws-...`, which Modal supports for proxy-authenticated endpoints.

## Client examples

Install the only local client dependency:

```bash
pip install requests
```

Preset voice:

```bash
python client.py tts \
  "Hi, I'm Maya. I'm from Jordan." \
  --speaker Aiden \
  --language English \
  --out maya.wav
```

High-fidelity voice clone:

```bash
python client.py clone reference.wav \
  "Hi, I'm Maya. I'm from Jordan." \
  --reference-text "This is the exact sentence spoken in the reference recording." \
  --language English \
  --out cloned.wav
```

Clone without a transcript, with lower-fidelity x-vector-only conditioning:

```bash
python client.py clone reference.wav \
  "Hello from the cloned voice." \
  --x-vector-only \
  --out cloned.wav
```

## Cost guardrails

The Modal classes use:

- `gpu="L4"`
- `scaledown_window=30`
- `max_containers=1` per TTS class
- no minimum warm containers
- a bounded request length
- private proxy-authenticated HTTP access

Historical measured baseline for the old **0.6B CustomVoice SDPA** deployment on L4:

- warm RTF: about **1.74**
- about **$0.0232 per generated audio minute**
- `$30` of L4 compute: about **1,295 audio minutes / 21.6 hours**

The production image now enables FlashAttention 2, so treat those numbers as a conservative historical baseline rather than a fresh benchmark. The 0.6B Base voice-clone path has **not yet been cost-benchmarked in this cleaned production setup**.

## API discovery

`GET /` returns the active model IDs, supported speakers/languages, GPU policy, and explicit capability flags including:

```json
{
  "ssml": false,
  "ipa_input": false,
  "style_instruction_on_0_6b": false
}
```

Agents should use those capability flags rather than assuming features from another Qwen3-TTS model size.

## Before changing architecture

Read `AGENTS.md` first. In particular, do not re-add old benchmark models or switch to a larger checkpoint unless the task explicitly requires a new architecture decision and its cost/quality tradeoff is documented.
