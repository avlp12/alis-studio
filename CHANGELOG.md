# Changelog

All notable changes to Alis Studio are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The version lives in exactly one place — `studio/__version__` (in `studio/__init__.py`).
`pyproject.toml` reads it via `[tool.setuptools.dynamic]`, the server injects it into the
web UI, and the DMG build stamps it into the app bundle.

## [0.7.3] — 2026-07-02

### Added
- **Shape tools** — the canvas editor grows from Sketch/Text to five tools: **Sketch, Circle, Box,
  Arrow, Text**. Shapes are drag-drawn with a live preview; a bare click with a shape tool is
  ignored (no accidental dots).
- **Stroke sizes** — Thin / Medium / Thick, applied to strokes, shapes, and text.
- **Redo** — full undo/redo for every mark (including text), with **⌘Z / ⇧⌘Z** while the editor
  is open (text fields keep their native undo).
- **Step history** — every edit result joins a thumbnail strip (*Original → Step 1 → …*); click a
  step to put it back on the canvas and continue from there (results also live in the gallery).
- **Fast / Fine quality** — 4-step (default) or 8-step edits, one tap.
- Each run now uses a **fresh random seed**, so "Edit" again on the same marks gives a new take.
- **Discard guards** — closing the editor (✕ / backdrop / Esc) or switching steps now asks before
  throwing away unsaved marks; Esc during a drag abandons just that drag; Clear can be brought back
  with Redo. ⌘Z/⇧⌘Z use the physical key, so undo works under a Korean input source.

### Changed
- Editor side panel reorganized ("Quiet Workshop" pass): sectioned Mark it / Describe the change /
  Steps, an icon toolbar, size dots, and disabled-state Undo/Redo. The base image is pre-rendered
  once, so long strokes stay smooth on big images; coordinates are guarded against a not-yet-laid-out
  canvas.

## [0.7.2] — 2026-07-02

### Added
- **Canvas editor (draw + instruct)** — hit **Edit** on any image (a fresh result or a gallery item)
  to open a Gemini-style editor: **Sketch** freehand in a chosen color, drop **Text** labels (click a spot →
  a transparent-background text box), then describe the change ("make the circled area blue"). The
  marks are baked into the image sent to **Qwen-Image Edit**, which follows them and paints the
  drawing out. The result replaces the canvas so you can **keep editing**, and every step is saved
  to the gallery. Colour palette, Undo, and Clear included.
  - Requires the Qwen-Image Edit backend (≥ 64 GB Mac); the same RAM confirm/gate applies before a run.

## [0.7.1] — 2026-07-02

### Added
- **Qwen-Image Edit** — instruction-based image editing (Tongyi, Apache-2.0): attach an image and
  describe the change ("make the hat red", understands Korean and other languages). Unlike
  strength-based img2img this follows an edit instruction; the output keeps the input's aspect ratio,
  normalized to ~1 MP (≈1024²). Offered in **8-bit** (needs ~64 GB RAM) and **bf16** (~96 GB);
  downloads ~54 GB on first use. The model picker now **warns** — and the app refuses with a confirm
  override — when a chosen build's RAM floor exceeds this Mac's memory (measured peaks: 8-bit ~39 GB,
  bf16 ~58 GB).
  - *4-bit is deliberately not offered* — mflux's 4-bit quantization of this model decodes to grainy
    noise regardless of step count (reproduced with the stock mflux CLI; same upstream issue that
    removed Qwen-Image 4-bit in #9).

### Changed
- The generation result now reports the **actual output size** (edit models keep the input's size;
  others may round to a multiple of 16) instead of the requested size.
- **Krea 2 Turbo** 2K sampling now pins the timestep shift to `mu = 1.15` at 2048² (matches Krea's
  own 2K recipe) instead of extrapolating past it (`krea2-alis-mlx` 0.1.2).

### Fixed
- Upscale status now hints at the first-use download size, and the gallery-save path was
  de-duplicated (single `_gallery_write` primitive).

## [0.7.0] — 2026-07-01

### Added
- **Image-to-image** — attach an image and transform it with a prompt. Z-Image Turbo, Qwen-Image,
  and FLUX get a drag-and-drop **Input image** control plus a **Strength** slider (Krea 2 Turbo
  stays text-to-image). The uploaded image never leaves your Mac; the NSFW filter still runs on the
  output.
- **Upscale (2× / 3×)** — a diffusion super-resolution step powered by **SeedVR2** (3B, Apache-2.0).
  Open any gallery image and upscale it to a crisp 2K–4K; the result is saved back to the gallery.
  Downloads on first use; auto-hidden on Macs under 24 GB.
- **Gallery lightbox** — click an image (or its prompt) to open a full view with the **full,
  editable prompt** and icon actions: Use (send the edited prompt to Generate), Copy, Download,
  Delete. Cards also get a quick Copy-prompt button.

### Changed
- **Larger resolutions per model** — Krea 2 Turbo up to **2048²** (it's a native 1K–2K model),
  Qwen-Image up to 1536², Z-Image / FLUX up to 1280². On a low-memory Mac the resolution control
  now **warns** when a chosen size may not fit (≤16 GB → above 1024², ≤24 GB → above 1536²).

## [0.6.3] — 2026-06-29

### Fixed
- **Qwen-Image 4-bit produced noisy/grainy images** ([#9](https://github.com/avlp12/alis-studio/issues/9)).
  Qwen-Image's ~20B transformer is too sensitive to 4-bit quantization — mflux blanket-quantizes its
  AdaLN modulation and output-projection layers, so the composition is right but the image is grainy
  (mflux's own docs warn that ≤6-bit "degrades a lot more compared to Flux"). Reproduced and confirmed
  4-bit noisy vs 8-bit clean. **Removed the 4-bit option for Qwen-Image; it now defaults to 8-bit**
  (bf16 also available). FLUX 4-bit is unaffected and stays.

## [0.6.2] — 2026-06-29

### Added
- **Recommended model for your Mac.** On launch, Alis Studio detects your unified memory and marks
  the best-fitting model **★ Recommended** in the picker — and selects it by default: **Z-Image Turbo**
  on a 16 GB Mac, **Krea 2 Turbo** on ≥ 24 GB. Each backend declares a `min_ram_gib` floor and
  `/api/system` now returns `recommended` / `recommended_label`.

### Changed
- README updated for Z-Image Turbo, 16 GB-Mac support, the low-memory tiling, and the recommendation.

## [0.6.1] — 2026-06-29

### Added
- **Lower memory on small Macs**: the mflux models (Z-Image Turbo, Qwen-Image, FLUX) now use mflux's
  VAE tiling for large (≥1024²) renders on ≤24 GB Macs, cutting Z-Image 4-bit's 1024² peak from
  ~12.9 GB to ~8.5 GB with no visible quality change (verified across every supported resolution).
  Smaller renders (≤768², which already fit) and big-RAM Macs keep the exact untiled decode; force it
  on or off with `ALIS_VAE_TILING=1` / `0`.
- A **"Loading…" status** now shows while a model is being loaded for the first time (and, on the very
  first use, while it downloads its weights) — so the wait isn't a blank progress bar.

### Changed
- The progress bar now advances **cumulatively across a multi-image batch** instead of restarting at
  step 1 for each image.

## [0.6.0] — 2026-06-28

### Added
- **Z-Image-Turbo model** (Tongyi-MAI / Alibaba, Apache-2.0): a ~6B single-stream DiT with a
  Qwen3-4B text encoder (understands Korean and other languages natively) and the FLUX VAE,
  distilled to ~9 steps with no CFG. At **4-bit (~6 GB) it runs on a 16 GB Mac** (comfortable at
  512–768px; 1024px is tight) — the first backend that doesn't effectively need a big-RAM machine
  the way the 12.9B Krea 2 Turbo default does. Powered by mflux (no model port), it downloads on
  first use: the 4-bit pre-quantized build (~6 GB) by default, or 8-bit/bf16 (~33 GB) from the
  official repo. Adds `mflux>=0.18` as a direct dependency.
- **Live progress bar + responsive Stop for the mflux backends** (Z-Image, Qwen-Image, FLUX):
  generation now reports each denoising step to the UI, and the Stop button interrupts
  mid-generation instead of only taking effect after the current image finishes.

## [0.5.4] — 2026-06-28

### Added
- **Gallery**: generated images are now saved to disk (`~/Library/Application Support/Alis Studio/
  gallery/`) with their prompt and settings, and a new **Gallery** view (top-bar toggle) shows your
  past work as a grid — reuse a prompt, download, or delete each one. Adds `GET /api/gallery`,
  `GET /api/gallery/<id>.png`, and `POST /api/gallery/delete`. (History starts now — images from
  earlier sessions weren't persisted.)

### Fixed
- Scrolling no longer rubber-bands the whole window into blank space (WKWebView elastic overscroll).
  The header is now a fixed bar and only the content area scrolls.

## [0.5.3] — 2026-06-28

### Changed
- **New brand icon**: a pine tree (echoing the project's profile imagery) with a generative
  sparkle and a "STUDIO" wordmark on the clay squircle — generated by Alis Studio itself. The
  in-app top-bar logo now uses the same pine mark, so the app icon, the in-app logo, and the
  profile image share one identity.

## [0.5.2] — 2026-06-28

### Fixed
- **Image generation crashed with `There is no Stream(gpu, 0) in current thread`** — on recent MLX
  this hit every generation through the app (and always after using the prompt enhancer), making
  0.5.0–0.5.1 unable to generate. MLX keeps its default GPU stream per-thread, but the stdlib
  threaded server ran model work on whichever request thread arrived, and the enhancer's mlx-lm
  corrupted that state further. Fix: all MLX work (generation, NSFW filter, model-cache scan, and
  registry build) now runs on a single dedicated GPU thread warmed up at startup, and the prompt
  enhancer's mlx-lm runs in its own isolated subprocess so it can't touch the image model's GPU state.

## [0.5.1] — 2026-06-28

### Added
- **Prompt enhancer (optional, fully on-device)**: a Settings toggle enables a local LLM
  (`Qwen3-4B-Instruct-2507`, via mlx-lm) that rewrites the prompt — translating non-English
  prompts (e.g. Korean) into English and enriching them into vivid image captions. Click the
  **Enhance** button to rewrite the prompt in place; you review/edit the English before
  generating, so seeds stay reproducible. Deterministic (greedy + cached). Adds
  `GET /api/system` and `POST /api/enhance`.
- **Memory-aware gating**: on Macs with ≤ 24 GB the toggle defaults off and asks for
  confirmation before enabling (the ~2.3 GB enhancer model competes with the image model for
  unified memory). The self-contained `.dmg` bundles mlx-lm; the model downloads on first use.

## [0.5.0] — 2026-06-27

### Added
- **Per-model prompt-language hint**: a one-line note under the prompt shows whether the active
  model understands non-English prompts natively (Krea 2 Turbo / Qwen-Image, via their Qwen text
  encoders) or works best in English (FLUX, via T5/CLIP) — so Korean users know Krea 2 Turbo and
  Qwen-Image take Korean directly, no translation needed.
- **Stop button**: while an image is generating, the Generate button becomes a Stop button.
  Clicking it cancels the run — the frontend aborts the request and `POST /api/cancel` flags the
  server, which halts at the next denoise step and reports `cancelled`. (Per-step backends like
  Krea 2 Turbo stop promptly; a fresh generation afterwards is unaffected.)
- App version is now shown in the UI — a small tag next to the brand in the top bar — and
  in the native window's title bar. New `GET /api/version` endpoint exposes it programmatically.
- **Self-contained `.dmg`**: `packaging/build_dmg.sh` builds an `Alis Studio.app` that bundles
  its own standalone Python interpreter **and every runtime dependency** (mlx, mflux, transformers,
  pywebview, the Krea 2 Turbo backend, …) under `Contents/Resources`. Double-click to run — no
  system Python, no `pip install`. The bundle is ad-hoc signed and wrapped in a drag-to-Applications
  disk image. `packaging/make_icon.py` renders the app's sparkle mark into `AppIcon.icns`.
  (Model **weights** are not bundled — they download from Hugging Face on the first generation.)

### Changed
- Single source of truth for the version: `pyproject.toml` no longer hard-codes it
  (`dynamic = ["version"]`), so the package version can never drift from `studio.__version__`.
- `pyproject.toml` uses the PEP 639 SPDX license expression (`license = "MIT"`) and requires
  `setuptools>=77`, replacing the deprecated `license = { text = "MIT" }` table form.

## [0.4.0] — 2026-06-27

### Added
- Native desktop app: runs in its own window via pywebview (WKWebView on macOS), no browser tab.

## [0.3.0] — 2026-06-27

### Added
- Multi-model support: Qwen-Image and FLUX (schnell/dev) backends alongside Krea 2 Turbo.
- Model names are shown in the UI; model picker reworked into a button + popover.

## [0.2.0] — 2026-06-27

### Added
- LM Studio-style detailed settings panel and an in-app model manager/downloader.

## [0.1.0] — 2026-06-27

### Added
- Initial release: a local, model-agnostic, standard-library image-generation studio for
  Apple silicon (MLX), with the pure-MLX Krea 2 Turbo backend.

[0.5.4]: https://github.com/avlp12/alis-studio/releases/tag/v0.5.4
[0.5.3]: https://github.com/avlp12/alis-studio/releases/tag/v0.5.3
[0.5.2]: https://github.com/avlp12/alis-studio/releases/tag/v0.5.2
[0.5.1]: https://github.com/avlp12/alis-studio/releases/tag/v0.5.1
[0.5.0]: https://github.com/avlp12/alis-studio/releases/tag/v0.5.0
[0.4.0]: https://github.com/avlp12/alis-studio/releases/tag/v0.4.0
[0.3.0]: https://github.com/avlp12/alis-studio/releases/tag/v0.3.0
[0.2.0]: https://github.com/avlp12/alis-studio/releases/tag/v0.2.0
[0.1.0]: https://github.com/avlp12/alis-studio/releases/tag/v0.1.0
