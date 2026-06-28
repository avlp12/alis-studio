"""FLUX backends via mflux (the MLX FLUX implementation, already a dependency).

These are catalog-less: mflux downloads the weights from Hugging Face on first use (and selects
the right files for the chosen quantization), so the model manager doesn't track per-byte download
state for them — the build is always selectable and fetches on the first Generate. Krea 2 Turbo,
by contrast, ships explicit download management (it uses our own HTTP-bridge downloader).
"""

from __future__ import annotations

from .base import Backend

_QUANT = {"8bit": 8, "4bit": 4, "bf16": None}


class _StepProgress:
    """Bridges mflux's per-step callback to Alis Studio's step_callback(step, total).

    Registered on an mflux model's callback registry, it fires once per denoise step so the
    UI can show a live progress bar — and, because step_callback raises when the user clicks
    Stop, it also lets a generation be interrupted mid-loop instead of only after it finishes.
    The Stop signal (server._Cancelled) is a plain Exception, NOT a KeyboardInterrupt, so it
    escapes mflux's `except KeyboardInterrupt` denoise loop and surfaces as a cancel — don't
    change _Cancelled to subclass KeyboardInterrupt or Stop becomes a silently-completed run.
    """

    def __init__(self, step_callback):
        self._cb = step_callback

    def call_in_loop(self, *, t, seed, prompt, latents, config, time_steps):
        total = getattr(config, "num_inference_steps", 0) or len(time_steps)
        self._cb(t + 1, total)


def _wire_progress(model, step_callback):
    """Attach a single per-step progress callback to an mflux model, replacing any previous one
    (the model is cached across generations, so we must not stack subscribers). Best-effort:
    progress is a nicety, so a change in mflux internals must never break generation — but leave
    a trace, since a silent failure would degrade Stop to job-boundary granularity."""
    try:
        model.callbacks.in_loop = [_StepProgress(step_callback)]
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("mflux progress/Stop wiring failed: %s", e)


def _flux_params(*, default_steps, max_steps, guidance_default, guidance_fixed, negative):
    return [
        {"key": "resolution", "label": "Resolution", "type": "resolution", "group": "Output",
         "sizes": [512, 768, 1024], "default_size": 1024,
         "aspects": ["1:1", "3:2", "2:3", "16:9", "9:16"], "default_aspect": "1:1",
         "min": 256, "max": 1536, "multiple": 16},
        {"key": "steps", "label": "Steps", "type": "int", "group": "Output",
         "min": 1, "max": max_steps, "default": default_steps},
        {"key": "num_images", "label": "Images", "type": "int", "group": "Output", "min": 1, "max": 4, "default": 1},
        {"key": "seed", "label": "Seed", "type": "seed", "group": "Sampling", "default": 0},
        {"key": "guidance", "label": "Guidance (CFG)", "type": "float", "group": "Sampling",
         "min": 0, "max": 10, "step": 0.5, "default": guidance_default, "fixed": guidance_fixed,
         **({"hint": "FLUX schnell is distilled — keep guidance at 0."} if guidance_fixed else {})},
        {"key": "negative", "label": "Negative prompt", "type": "text", "group": "Advanced",
         "default": "", "enabled": negative,
         **({} if negative else {"hint": "schnell runs without guidance, so a negative prompt has no effect."})},
    ]


class _MfluxFlux(Backend):
    """Shared FLUX-family backend (txt2img via mflux's Flux1)."""
    prompt_note = "Works best with English prompts — its T5/CLIP text encoder is English-centric."
    mflux_name = ""   # mflux model alias: "schnell" / "dev"
    repo = ""         # HF repo, for the gated-access message
    variants = [{"id": "8bit", "label": "8-bit"}, {"id": "4bit", "label": "4-bit"}, {"id": "bf16", "label": "bf16"}]

    @classmethod
    def is_available(cls) -> bool:
        try:
            import mflux.models.flux.variants.txt2img.flux  # noqa: F401
            return True
        except Exception:
            return False

    def __init__(self):
        self._model = None
        self._variant = None

    def _get(self, variant):
        import gc
        import mlx.core as mx
        from mflux.models.flux.variants.txt2img.flux import Flux1
        if self._model is None or self._variant != variant:
            self._model, self._variant = None, None
            gc.collect()
            mx.clear_cache()
            try:
                self._model = Flux1.from_name(self.mflux_name, quantize=_QUANT.get(variant, 8))
            except Exception as e:  # FLUX repos are gated — give an actionable message, not a traceback
                m = str(e).lower()
                if any(k in m for k in ("gated", "403", "restricted", "authorized", "awaiting")):
                    raise ValueError(
                        f"{self.label} is a gated model. Accept its license at "
                        f"https://huggingface.co/{self.repo} , run `huggingface-cli login`, then retry."
                    ) from None
                raise
            self._variant = variant
        return self._model

    def generate(self, *, prompt, variant, params, step_callback):
        model = self._get(variant)
        _wire_progress(model, step_callback)
        neg = (params.get("negative") or "").strip() or None
        out = []
        for i in range(int(params.get("num_images", 1))):
            img = model.generate_image(
                seed=int(params.get("seed", 0)) + i, prompt=prompt,
                num_inference_steps=int(params.get("steps", 4)),
                height=int(params.get("height", 1024)), width=int(params.get("width", 1024)),
                guidance=float(params.get("guidance", 0) or 0), negative_prompt=neg,
            )
            out.append(img.image)
        return out


class FluxSchnellBackend(_MfluxFlux):
    id = "flux-schnell"
    label = "FLUX.1 schnell"
    info = "gated on HF — accept license + login, then ~24 GB on first use"
    mflux_name = "schnell"
    repo = "black-forest-labs/FLUX.1-schnell"
    params = _flux_params(default_steps=4, max_steps=8, guidance_default=0, guidance_fixed=True, negative=False)


class FluxDevBackend(_MfluxFlux):
    id = "flux-dev"
    label = "FLUX.1 dev"
    info = "gated, non-commercial — accept license + login, then ~24 GB on first use"
    mflux_name = "dev"
    repo = "black-forest-labs/FLUX.1-dev"
    params = _flux_params(default_steps=25, max_steps=50, guidance_default=3.5, guidance_fixed=False, negative=True)


class QwenImageBackend(Backend):
    """Qwen-Image (open, Apache-2.0) via mflux — downloads on first use, no HF gating."""
    id = "qwen-image"
    label = "Qwen-Image"
    prompt_note = "Understands Korean and other languages natively (Qwen2.5 text encoder)."
    info = "Apache-2.0 (open) · large (~40 GB), downloads on first use via mflux"
    variants = [{"id": "8bit", "label": "8-bit"}, {"id": "4bit", "label": "4-bit"}, {"id": "bf16", "label": "bf16"}]
    params = _flux_params(default_steps=20, max_steps=50, guidance_default=4.0, guidance_fixed=False, negative=True)

    @classmethod
    def is_available(cls) -> bool:
        try:
            import mflux.models.qwen.variants.txt2img.qwen_image  # noqa: F401
            return True
        except Exception:
            return False

    def __init__(self):
        self._model = None
        self._variant = None

    def _get(self, variant):
        import gc
        import mlx.core as mx
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
        if self._model is None or self._variant != variant:
            self._model, self._variant = None, None
            gc.collect()
            mx.clear_cache()
            self._model = QwenImage(quantize=_QUANT.get(variant, 8))
            self._variant = variant
        return self._model

    def generate(self, *, prompt, variant, params, step_callback):
        model = self._get(variant)
        _wire_progress(model, step_callback)
        neg = (params.get("negative") or "").strip() or None
        out = []
        for i in range(int(params.get("num_images", 1))):
            img = model.generate_image(
                seed=int(params.get("seed", 0)) + i, prompt=prompt,
                num_inference_steps=int(params.get("steps", 20)),
                height=int(params.get("height", 1024)), width=int(params.get("width", 1024)),
                guidance=float(params.get("guidance", 4) or 4), negative_prompt=neg,
            )
            out.append(img.image)
        return out
