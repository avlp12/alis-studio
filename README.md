# Alis Studio

A local, **model-agnostic** image-generation studio for **Apple silicon** — a clean, native-feeling
web UI that runs text-to-image models entirely on your Mac with [MLX](https://github.com/ml-explore/mlx).
No cloud, no accounts, your images never leave your machine.

The first model it ships with is **[Krea 2 Turbo](https://github.com/avlp12/krea2_alis_mlx)**
(a 12.9B text-to-image model, pure-MLX). Other models plug in as small backends — see
[Adding a model](#adding-a-model).

---

## Quickstart

**Requires an Apple-silicon Mac (M1+) with ≥ 24 GB unified memory.** On macOS use `python3`.

```bash
git clone https://github.com/avlp12/alis-studio.git
cd alis-studio
python3 -m pip install -r requirements.txt
python3 app.py            # opens http://localhost:7860 in your browser
```

Type a prompt, pick a model, click **Generate**. The first run downloads the model weights from
Hugging Face (a few minutes); after that it's instant to start. A 1024² image takes ~50 s on an
M3 Ultra (8-step Turbo; slower chips take longer).

- **Detailed settings** — a model-adaptive panel on the right: resolution with aspect-ratio
  presets, steps, batch size, seed (with randomize), guidance, sampler, negative prompt. Each
  model exposes exactly the controls it supports; the panel renders itself from the backend.
- **Model manager** — the **Models** button (top-right) opens a manager to browse, **download**
  (with a live progress bar), and delete model weights, and see total disk usage.
- **Live progress** — a per-step bar as the model denoises. **Light + dark** follow your system.
- **NSFW safety filter** runs by default (pure-MLX, no PyTorch); toggle it with the shield icon.
- Bind to your LAN with `ALIS_HOST=0.0.0.0 python3 app.py` (only on networks you trust); change
  the port with `ALIS_PORT=7861`.

---

## Models

The **model** dropdown and the **Models** manager are built from whatever backends are installed —
the UI discovers them at startup via `/api/models` and `/api/catalog`, so adding a model needs no
UI changes. Browse, download, and delete builds in the manager; switching the active model loads
that build on demand (and frees the previous one — two 12.9B models won't fit in memory at once).

| Model | Backend | Builds |
|---|---|---|
| **Krea 2 Turbo** | `krea2-alis-mlx` | 8-bit (best quality, 14.2 GB) · mixed-4/8 (smaller, 9.8 GB). 8-step Turbo. |

---

## Adding a model

Drop a file in `studio/backends/`, subclass `Backend`, and register it. The backend *declares*
its settings (`params`) and downloadable builds (`catalog`); the UI renders the settings panel and
the model manager from those declarations — no UI code to touch.

```python
# studio/backends/mymodel.py
from .base import Backend

class MyBackend(Backend):
    id = "my-model"
    label = "My Model"
    variants = [{"id": "default", "label": "default"}]
    # settings the UI renders into the right-hand panel (see studio/backends/base.py for all types):
    params = [
        {"key": "resolution", "label": "Resolution", "type": "resolution", "group": "Output",
         "sizes": [512, 1024], "default_size": 1024, "aspects": ["1:1", "3:2", "16:9"],
         "default_aspect": "1:1", "min": 256, "max": 1536, "multiple": 16},
        {"key": "steps", "label": "Steps", "type": "int", "group": "Output", "min": 1, "max": 50, "default": 20},
        {"key": "guidance", "label": "Guidance", "type": "float", "group": "Sampling",
         "min": 0, "max": 12, "step": 0.5, "default": 7.5},
        {"key": "seed", "label": "Seed", "type": "seed", "group": "Sampling", "default": 0},
    ]
    catalog = [{"variant": "default", "label": "default", "size_gb": 6.0, "note": "fp16"}]

    @classmethod
    def is_available(cls):
        try:
            import my_package  # noqa
            return True
        except Exception:
            return False

    def generate(self, *, prompt, variant, params, step_callback):
        # params carries width, height, steps, seed, num_images + your custom keys.
        # call step_callback(step, total) per denoising step for the live progress bar.
        return [pil_image, ...]

    # optional — enables the model manager's download / delete / installed state:
    def is_installed(self, variant): ...
    def download(self, variant, progress): ...   # call progress(done_bytes, total_bytes)
    def delete(self, variant): ...
```

Then add it to `studio/registry.py`:

```python
from .backends.mymodel import MyBackend
BACKENDS = [Krea2Backend, MyBackend]
```

Backends whose dependencies aren't installed are skipped automatically, so the app always runs
with whatever you have.

---

## How it works

- `app.py` → `studio/server.py` — a Python **standard-library** HTTP server (zero web dependencies).
  Serves `web/index.html` and a small JSON/NDJSON API:
  - `GET /api/models` — registered models + their settings schema (drives the dropdown + settings panel)
  - `POST /api/generate` — streams NDJSON: one line per denoising step, then the base64 PNGs
  - `GET /api/catalog` · `POST /api/download` (streamed progress) · `POST /api/delete` — the model manager
- `studio/registry.py` — the list of available models.
- `studio/backends/` — one file per model; each wraps a model behind the small `Backend` interface.
- `studio/download.py` — the resilient HTTP-bridge downloader (resumable, integrity-checked).

Generation is serialized (one GPU, one job at a time). The NSFW filter is applied at the app level
to every backend's output, reusing Krea 2's pure-MLX classifier.

---

## Privacy & license

Everything runs **locally** — prompts and images never leave your Mac.

This application is **MIT** licensed ([`LICENSE`](LICENSE)). **Each model carries its own license:**
the Krea 2 Turbo backend uses weights under the
[Krea 2 Community License](https://krea.ai/krea-2-licensing) (commercial use requires annual revenue
under $1M; content filtering required for deployments — the built-in filter is on by default). You're
responsible for complying with the license of any model you load.

*Part of the **Alis** MLX line — see also [krea2_alis_mlx](https://github.com/avlp12/krea2_alis_mlx).*
