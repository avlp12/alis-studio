"""Shared infrastructure for the mflux-backed backends (FLUX, Qwen-Image, Z-Image).

Kept in a neutral module so no backend imports another: both ``mflux_models`` and ``z_image``
import the live-progress bridge and the low-memory VAE-tiling policy from here.
"""

from __future__ import annotations

import logging
import os

_log = logging.getLogger(__name__)

# --- low-memory policy: VAE tiling for large decodes on constrained Macs ----------------------
# A large VAE decode is the memory spike: measured on Z-Image 4-bit, the 1024² peak is ~12.9 GB
# untiled vs ~8.5 GB tiled (output visually identical; ≤768² already fits untiled at ≤9.8 GB).
# So tile only (a) on a Mac that needs it and (b) for genuinely large outputs.
_TILE_RAM_GIB = 24                 # ≤24 GiB Macs (16/24 GB) — where an untiled 1024² risks pressure.
_TILE_MIN_PIXELS = 1024 * 1024     # only tile ≥1024²; ≤768² keeps the exact untiled decode.
_constrained: bool | None = None   # memoized hardware check (RAM doesn't change at runtime)


def _total_ram_gib() -> float:
    try:  # canonical on macOS
        import subprocess
        out = subprocess.run(["/usr/sbin/sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2)
        return int(out.stdout.strip()) / (1024 ** 3)
    except Exception:
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
        except Exception:
            return 0.0


def _is_constrained() -> bool:
    global _constrained
    if _constrained is None:
        ram = _total_ram_gib()
        _constrained = 0 < ram <= _TILE_RAM_GIB
    return _constrained


def _tiling_enabled() -> bool:
    """Whether VAE tiling is wanted. ``ALIS_VAE_TILING=1/0`` forces it on/off (lets a big-RAM user
    opt in, or a constrained user keep the exact decode); otherwise it auto-enables on low-RAM Macs."""
    v = os.environ.get("ALIS_VAE_TILING")
    if v is not None:
        return v.strip().lower() not in ("", "0", "false", "no", "off")
    return _is_constrained()


def _apply_memory_policy(model, width: int, height: int) -> None:
    """Turn mflux's VAE tiling on for a large decode on a memory-constrained Mac, off otherwise.
    ``tiling_config`` is just a flag read at decode time (no weight change), so it is safe to toggle
    per generation on a cached model. Big-RAM Macs and ≤768² outputs keep the exact untiled decode.
    Applies to any mflux backend, though in practice only Z-Image runs on such Macs. Best-effort —
    a tiling-config change must never break generation."""
    try:
        from mflux.models.common.vae.tiling_config import TilingConfig
        tile = _tiling_enabled() and (width * height) >= _TILE_MIN_PIXELS
        model.tiling_config = TilingConfig() if tile else None
        if tile:
            _log.info("VAE tiling enabled for %dx%d decode (low-memory policy)", width, height)
    except Exception as e:
        _log.warning("VAE tiling policy not applied: %s", e)


# --- live progress bridge ---------------------------------------------------------------------
class _StepProgress:
    """Bridges mflux's per-step callback to Alis Studio's step_callback(step, total).

    Registered on an mflux model's callback registry, it fires once per denoise step so the
    UI can show a live progress bar — and, because step_callback raises when the user clicks
    Stop, it also lets a generation be interrupted mid-loop instead of only after it finishes.
    The Stop signal (server._Cancelled) is a plain Exception, NOT a KeyboardInterrupt, so it
    escapes mflux's `except KeyboardInterrupt` denoise loop and surfaces as a cancel — don't
    change _Cancelled to subclass KeyboardInterrupt or Stop becomes a silently-completed run.
    """

    def __init__(self, step_callback, base=0, batches=1):
        # base = index of the current image in a multi-image batch; batches = batch size.
        # Reporting (base*total + step, batches*total) makes the bar advance monotonically across
        # the whole batch instead of restarting from 1 for each image.
        self._cb = step_callback
        self._base = base
        self._batches = batches

    def call_in_loop(self, *, t, seed, prompt, latents, config, time_steps):
        total = getattr(config, "num_inference_steps", 0) or len(time_steps)
        self._cb(self._base * total + t + 1, self._batches * total)


def _wire_progress(model, step_callback, base=0, batches=1) -> None:
    """Attach a single per-step progress callback to an mflux model, replacing any previous one
    (the model is cached across generations, so we must not stack subscribers; re-wired per image
    in a batch so progress is cumulative). Best-effort: progress is a nicety, so a change in mflux
    internals must never break generation — but leave a trace, since a silent failure would degrade
    Stop to job-boundary granularity."""
    try:
        model.callbacks.in_loop = [_StepProgress(step_callback, base, batches)]
    except Exception as e:
        _log.warning("mflux progress/Stop wiring failed: %s", e)


# --- image-to-image (shared by every mflux backend; mflux's generate_image takes image_path/strength) ---
def _img2img_params():
    """Optional input-image + strength controls. The UI renders type 'image' as a drop zone and
    puts the picked image (data URI) in params['init_image']; the server decodes it to a temp file
    and sets params['image_path'] before generate()."""
    return [
        {"key": "init_image", "label": "Input image", "type": "image", "group": "Image-to-image",
         "hint": "Optional — attach an image to transform it with your prompt (img2img)."},
        {"key": "strength", "label": "Strength", "type": "float", "group": "Image-to-image",
         "min": 0.1, "max": 1.0, "step": 0.05, "default": 0.6,
         "hint": "How much to change the input — higher = more change. Used only with an input image."},
    ]


def _img2img_args(params):
    """Resolve (image_path, image_strength) for a generate_image() call from params; (None, None)
    means plain txt2img. image_path is set by the server after decoding the uploaded image."""
    path = params.get("image_path")
    if not path:
        return None, None
    try:
        strength = float(params.get("strength", 0.6))
    except (TypeError, ValueError):
        strength = 0.6
    return path, strength
