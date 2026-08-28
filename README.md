# Free Image Editing on Modal — Mage-Flow Turbo

This repository deploys Microsoft's **Mage-Flow-Turbo** and **Mage-Flow-Edit-Turbo** to Modal with conservative defaults for a Starter workspace.

## What it deploys

- Text-to-image: `microsoft/Mage-Flow-Turbo` (4 steps)
- Image editing: `microsoft/Mage-Flow-Edit-Turbo` (4 steps)
- GPU: Modal **L4 24 GB**
- Persistent model cache: Modal Volume `mage-flow-models`
- Scale-to-zero: GPU containers shut down after 30 seconds idle
- Cost guardrail: one generation container + one editing container maximum
- Private HTTP API protected by **Modal Proxy Auth**
- Automatic deploy from GitHub Actions on pushes to `main`

The Microsoft Mage source is pinned to commit `76bec2bb3818863f470de7e867c2dc7f1d0bfd83` for reproducibility.

## GitHub secrets

The deploy workflow expects these repository Actions secrets:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

Do not put their values in the repository.

## Automatic deployment

`.github/workflows/deploy-modal.yml` does two things:

1. Downloads both Hugging Face checkpoints into the persistent Modal Volume.
2. Deploys `modal_app.py`.

The first run can take a while because it downloads the model weights and builds FlashAttention. Later deploys should reuse Modal caches.

## Find the API URL

After the GitHub Action succeeds, open the Modal dashboard and select the `free-image-editing` app. The `api` web function will show its URL.

The endpoint is intentionally private so strangers cannot burn your GPU credits.

## Create a Proxy Token

For API access, install the Modal CLI locally and authenticate, then create a proxy token:

```bash
pip install modal
modal setup
modal workspace proxy-tokens create
```

Save the printed `wk-...` ID and `ws-...` secret somewhere safe. They are different from your Modal API token used by GitHub Actions.

## Test generation

Install the tiny local client dependency:

```bash
pip install requests
```

Then set:

```bash
export MAGE_API_URL='https://YOUR-MODAL-URL.modal.run'
export MODAL_PROXY_TOKEN_ID='wk-...'
export MODAL_PROXY_TOKEN_SECRET='ws-...'
```

Generate an image:

```bash
python client.py generate "A premium black coffee package with the words NIGHT ROAST in elegant gold typography" --out coffee.png
```

## Test editing

```bash
python client.py edit input.png "Replace the background with a neon Tokyo street at night while preserving the subject" --out edited.png
```

## API

### `POST /generate`

JSON body:

```json
{
  "prompt": "a product poster with legible text",
  "width": 1024,
  "height": 1024,
  "seed": 42
}
```

Returns `image/png`.

### `POST /edit`

Multipart form fields:

- `image`: source image file
- `prompt`: edit instruction
- `max_size`: 512–1536, default 1024
- `seed`: default 42

Returns `image/png`.

## Cost protection

The code intentionally uses:

- `gpu="L4"`
- `scaledown_window=30`
- `max_containers=1`
- 4-step Turbo checkpoints
- a 1536 px API ceiling even though Mage can support larger images
- Modal proxy authentication

These choices prioritize making the $30 Starter credit last. If L4 proves too slow or runs out of VRAM for a specific workload, change the two GPU class decorators to `gpu="L40S"`.

## Manual deploy

If GitHub Actions is unavailable:

```bash
pip install modal
modal setup
modal run modal_app.py::download_models
modal deploy modal_app.py
```
