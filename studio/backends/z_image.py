"""Z-Image-Turbo backend via mflux (Tongyi-MAI / Alibaba, Apache-2.0).

Z-Image-Turbo is a ~6B single-stream DiT with a Qwen3-4B text encoder (so it understands
Korean and other languages natively) and the FLUX VAE, distilled to ~9 steps with no CFG.
At 4-bit the whole pipeline is small enough (~6 GB) to run on a 16 GB Mac — unlike the 12.9B
Krea 2 Turbo default, which effectively needs a big-RAM machine.

mflux already ships a first-class MLX implementation (the same library this app uses for FLUX
and Qwen-Image), so this is a thin wiring backend like the others — no model port.

The default 4-bit build loads mflux's pre-quantized repo (``filipstrand/Z-Image-Turbo-mflux-4bit``,
~6 GB) so a clean 16 GB machine never has to hold the full-precision weights. The 8-bit and bf16
builds quantize on the fly from the official ``Tongyi-MAI/Z-Image-Turbo`` repo (~33 GB download).
"""

from __future__ import annotations

from .base import Backend
from .mflux_common import _apply_memory_policy, _img2img_args, _img2img_params, _wire_progress

# variant id -> (mflux model_path, quantize).
# 4-bit: a ready pre-quantized repo (no on-the-fly quantization, light download, 16 GB-friendly).
# 8-bit/bf16: the official full-precision repo, quantized at load (8) or kept as-is (None=bf16).
_BUILDS = {
    "4bit": ("filipstrand/Z-Image-Turbo-mflux-4bit", None),
    "8bit": (None, 8),
    "bf16": (None, None),
}


class ZImageTurboBackend(Backend):
    """Z-Image-Turbo (open, Apache-2.0) via mflux — downloads on first use, no HF gating."""

    id = "z-image-turbo"
    label = "Z-Image Turbo"
    min_ram_gib = 16   # 4-bit pipeline ~6 GB resident; 1024² peaks ~8.5 GB with VAE tiling → runs on 16 GB
    prompt_note = "Understands Korean and other languages natively (Qwen3 text encoder). Distilled — fast at ~9 steps."
    info = "Apache-2.0 (open) · 4-bit runs on a 16 GB Mac (best at 512–768px) · downloads on first use via mflux"
    # 4-bit is listed first on purpose: it is the default (variants[0]) and the only 16 GB-friendly build.
    variants = [
        {"id": "4bit", "label": "4-bit · ~6 GB · 16 GB-Mac friendly"},
        {"id": "8bit", "label": "8-bit · ~33 GB download"},
        {"id": "bf16", "label": "bf16 · full precision, ~33 GB"},
    ]
    params = [
        {"key": "resolution", "label": "Resolution", "type": "resolution", "group": "Output",
         "sizes": [512, 768, 1024, 1280], "default_size": 1024,   # native 1024; usable to ~1280
         "aspects": ["1:1", "3:2", "2:3", "16:9", "9:16"], "default_aspect": "1:1",
         "min": 256, "max": 1536, "multiple": 16},
        {"key": "steps", "label": "Steps", "type": "int", "group": "Output",
         "min": 1, "max": 20, "default": 9},
        {"key": "num_images", "label": "Images", "type": "int", "group": "Output", "min": 1, "max": 4, "default": 1},
        {"key": "seed", "label": "Seed", "type": "seed", "group": "Sampling", "default": 0},
        {"key": "guidance", "label": "Guidance (CFG)", "type": "float", "group": "Sampling",
         "min": 0, "max": 10, "step": 0.5, "default": 0, "fixed": True,
         "hint": "Z-Image Turbo is distilled — it runs without CFG (guidance 0)."},
        {"key": "negative", "label": "Negative prompt", "type": "text", "group": "Advanced",
         "default": "", "enabled": False,
         "hint": "Turbo runs without guidance, so a negative prompt has no effect."},
        *_img2img_params(),
    ]

    @classmethod
    def is_available(cls) -> bool:
        try:
            import mflux.models.z_image.variants.z_image  # noqa: F401
            return True
        except Exception:
            return False

    def __init__(self):
        self._model = None
        self._variant = None

    def _get(self, variant):
        import gc
        import mlx.core as mx
        from mflux.models.common.config import ModelConfig
        from mflux.models.z_image.variants.z_image import ZImage
        if self._model is None or self._variant != variant:
            self._model, self._variant = None, None
            gc.collect()
            mx.clear_cache()
            model_path, quantize = _BUILDS.get(variant, _BUILDS["4bit"])
            try:
                self._model = ZImage(model_config=ModelConfig.z_image_turbo(),
                                     quantize=quantize, model_path=model_path)
            except Exception as e:  # the 4-bit build is a community pre-quant repo — guide, don't traceback
                m = str(e).lower()
                if model_path and any(k in m for k in ("not found", "404", "401", "403",
                                                       "repository", "gated", "restricted")):
                    raise ValueError(
                        "The 4-bit Z-Image build (filipstrand/Z-Image-Turbo-mflux-4bit) couldn't be "
                        "downloaded — it may have moved or you may be offline. Try the 8-bit or bf16 build."
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
        img_path, strength = _img2img_args(params)
        n = int(params.get("num_images", 1))
        out = []
        for i in range(n):
            _wire_progress(model, step_callback, base=i, batches=n)
            img = model.generate_image(
                seed=int(params.get("seed", 0)) + i, prompt=prompt,
                num_inference_steps=int(params.get("steps", 9)),
                height=h, width=w,
                guidance=0.0,   # turbo is distilled (supports_guidance=False); mflux forces 0 anyway
                image_path=img_path, image_strength=strength,
            )
            out.append(img.image)
        return out
