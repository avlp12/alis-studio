"""SeedVR2 diffusion super-resolution (mflux, Apache-2.0).

Not a text-to-image backend — a post-process "upscale" action applied to an image the user
already has (a generation result or a gallery item). Loaded lazily on first use and, like every
other model, run on the server's single dedicated GPU thread. The 3B weights download on first use.
"""

from __future__ import annotations


class Upscaler:
    label = "SeedVR2 upscaler"
    min_ram_gib = 24   # a 3B SR model plus a 2–4 MP target, often on top of a cached image model

    @classmethod
    def is_available(cls) -> bool:
        try:
            import mflux.models.seedvr2.variants.upscale.seedvr2  # noqa: F401
            return True
        except Exception:
            return False

    def __init__(self):
        self._model = None

    def will_load(self) -> bool:
        return self._model is None

    def _get(self):
        import gc
        import mlx.core as mx
        from mflux.models.common.config import ModelConfig
        from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
        if self._model is None:
            gc.collect()
            mx.clear_cache()
            self._model = SeedVR2(model_config=ModelConfig.seedvr2_3b())
        return self._model

    def upscale(self, image_path, scale):
        """Upscale the image at image_path by an integer factor (2 or 3). Returns a PIL image."""
        from mflux.utils.scale_factor import ScaleFactor
        model = self._get()
        img = model.generate_image(seed=0, image_path=image_path,
                                   resolution=ScaleFactor(value=int(scale)), softness=0.0)
        return getattr(img, "image", img)
