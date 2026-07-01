"""FLUX backends via mflux (the MLX FLUX implementation, already a dependency).

These are catalog-less: mflux downloads the weights from Hugging Face on first use (and selects
the right files for the chosen quantization), so the model manager doesn't track per-byte download
state for them — the build is always selectable and fetches on the first Generate. Krea 2 Turbo,
by contrast, ships explicit download management (it uses our own HTTP-bridge downloader).
"""

from __future__ import annotations

from .base import Backend
from .mflux_common import _apply_memory_policy, _img2img_args, _img2img_params, _wire_progress

_QUANT = {"8bit": 8, "4bit": 4, "bf16": None}


def _flux_params(*, default_steps, max_steps, guidance_default, guidance_fixed, negative,
                 sizes=(512, 768, 1024, 1280), max_res=1536):
    return [
        {"key": "resolution", "label": "Resolution", "type": "resolution", "group": "Output",
         "sizes": list(sizes), "default_size": 1024,
         "aspects": ["1:1", "3:2", "2:3", "16:9", "9:16"], "default_aspect": "1:1",
         "min": 256, "max": max_res, "multiple": 16},
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
        *_img2img_params(),
    ]


class _MfluxFlux(Backend):
    """Shared FLUX-family backend (txt2img via mflux's Flux1)."""
    min_ram_gib = 24   # 12B; ~24 GB on first use
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

    def will_load(self, variant):
        return self._model is None or self._variant != variant

    def generate(self, *, prompt, variant, params, step_callback):
        model = self._get(variant)
        w, h = int(params.get("width", 1024)), int(params.get("height", 1024))
        _apply_memory_policy(model, w, h)
        neg = (params.get("negative") or "").strip() or None
        img_path, strength = _img2img_args(params)
        n = int(params.get("num_images", 1))
        out = []
        for i in range(n):
            _wire_progress(model, step_callback, base=i, batches=n)
            img = model.generate_image(
                seed=int(params.get("seed", 0)) + i, prompt=prompt,
                num_inference_steps=int(params.get("steps", 4)),
                height=h, width=w,
                guidance=float(params.get("guidance", 0) or 0), negative_prompt=neg,
                image_path=img_path, image_strength=strength,
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
    min_ram_gib = 32   # ~20B; large (~40 GB at 8-bit) → wants a roomy Mac
    prompt_note = "Understands Korean and other languages natively (Qwen2.5 text encoder)."
    info = "Apache-2.0 (open) · large (~40 GB), downloads on first use via mflux"
    # No 4-bit: Qwen-Image's ~20B transformer is too sensitive to 4-bit (mflux blanket-quantizes
    # the AdaLN modulation + output projection → grainy/noisy output; mflux's own docs warn ≤6-bit
    # "degrades a lot more compared to Flux"). 8-bit is the floor for clean output. See issue #9.
    variants = [{"id": "8bit", "label": "8-bit"}, {"id": "bf16", "label": "bf16"}]
    params = _flux_params(default_steps=20, max_steps=50, guidance_default=4.0, guidance_fixed=False, negative=True,
                          sizes=(512, 768, 1024, 1280, 1536), max_res=1536)   # Qwen-Image native ~1328

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

    def will_load(self, variant):
        return self._model is None or self._variant != variant

    def generate(self, *, prompt, variant, params, step_callback):
        model = self._get(variant)
        w, h = int(params.get("width", 1024)), int(params.get("height", 1024))
        _apply_memory_policy(model, w, h)
        neg = (params.get("negative") or "").strip() or None
        img_path, strength = _img2img_args(params)
        n = int(params.get("num_images", 1))
        out = []
        for i in range(n):
            _wire_progress(model, step_callback, base=i, batches=n)
            img = model.generate_image(
                seed=int(params.get("seed", 0)) + i, prompt=prompt,
                num_inference_steps=int(params.get("steps", 20)),
                height=h, width=w,
                guidance=float(params.get("guidance", 4) or 4), negative_prompt=neg,
                image_path=img_path, image_strength=strength,
            )
            out.append(img.image)
        return out
