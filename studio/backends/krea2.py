"""Krea 2 Turbo backend — wraps the `krea2-alis-mlx` package (the pure-MLX port).

Weights are pulled from Hugging Face on first use (cached under ~/.cache/krea2_alis_mlx);
drop a transformer_*.safetensors in the working directory to use a local copy instead.
"""

from __future__ import annotations

import os

from .base import Backend
from .mflux_common import _img2img_args, _img2img_params


class Krea2Backend(Backend):
    min_ram_gib = 24   # 12.9B DiT; mixed-4/8 ~9.8 GB + activations → wants ≥24 GB
    id = "krea2-turbo"
    label = "Krea 2 Turbo"
    prompt_note = "Understands Korean and other languages natively (Qwen3 text encoder)."
    variants = [
        {"id": "8bit", "label": "8-bit · best quality"},
        {"id": "mixed-4-8", "label": "mixed-4/8 · smaller"},
    ]
    params = [
        {"key": "resolution", "label": "Resolution", "type": "resolution", "group": "Output",
         "sizes": [512, 768, 1024, 1280, 1536, 2048], "default_size": 1024,   # Krea 2 Turbo is a native 1K–2K model
         "aspects": ["1:1", "3:2", "2:3", "16:9", "9:16"], "default_aspect": "1:1",
         "min": 256, "max": 2048, "multiple": 16},
        {"key": "steps", "label": "Steps", "type": "int", "group": "Output",
         "min": 1, "max": 50, "default": 8},
        {"key": "num_images", "label": "Images", "type": "int", "group": "Output",
         "min": 1, "max": 4, "default": 1},
        {"key": "seed", "label": "Seed", "type": "seed", "group": "Sampling", "default": 0},
        {"key": "guidance", "label": "Guidance (CFG)", "type": "float", "group": "Sampling",
         "min": 0, "max": 10, "step": 0.5, "default": 0, "fixed": True,
         "hint": "Krea 2 Turbo is distilled — keep guidance at 0."},
        {"key": "sampler", "label": "Sampler", "type": "select", "group": "Sampling",
         "options": [{"value": "euler", "label": "Euler (flow-match)"}], "default": "euler", "fixed": True},
        {"key": "negative", "label": "Negative prompt", "type": "text", "group": "Advanced",
         "default": "", "enabled": False,
         "hint": "Distilled Turbo runs without guidance, so a negative prompt has no effect."},
        *_img2img_params(),   # Input image + Strength — krea2-alis-mlx >= 0.2 does rectified-flow img2img
    ]
    catalog = [
        {"variant": "8bit", "label": "8-bit · best quality", "size_gb": 14.2, "note": "near-lossless"},
        {"variant": "mixed-4-8", "label": "mixed-4/8 · smaller", "size_gb": 9.8, "note": "smallest near-lossless"},
    ]

    @classmethod
    def is_available(cls) -> bool:
        try:
            import krea2.pipeline  # noqa: F401
            return True
        except Exception:
            return False

    def __init__(self):
        self._pipe = None
        self._variant = None

    def will_load(self, variant: str) -> bool:
        return self._pipe is None or self._variant != variant   # mirrors the reload check in _get

    def _get(self, variant: str):
        import gc
        import mlx.core as mx
        from krea2.pipeline import Krea2Pipeline, resolve_weights

        if self._pipe is None or self._variant != variant:
            prec, path = resolve_weights(os.getcwd(), precision=variant, download=True)
            # free the previous build before loading another — two 12.9B transformers won't fit
            self._pipe, self._variant = None, None
            gc.collect()
            mx.clear_cache()
            self._pipe = Krea2Pipeline(path, precision=prec, base_dir=os.environ.get("KREA2_BASE_DIR"))
            self._variant = prec
        return self._pipe

    def generate(self, *, prompt, variant, params, step_callback):
        image_path, strength = _img2img_args(params)   # (None, None) = plain txt2img
        kwargs = {}
        if image_path:
            import inspect
            from krea2.pipeline import Krea2Pipeline
            # capability check BEFORE _get — don't load 14 GB of weights just to fail on an old package
            if "init_image" not in inspect.signature(Krea2Pipeline.generate).parameters:  # pre-0.2
                raise ValueError("This build of krea2-alis-mlx predates img2img — update it with "
                                 "`pip install -U git+https://github.com/avlp12/krea2_alis_mlx` "
                                 "(or reinstall the app), or remove the input image.")
            kwargs = {"init_image": image_path, "strength": strength}
        pipe = self._get(variant)
        return pipe.generate(
            prompt,
            width=int(params.get("width", params.get("size", 1024))),
            height=int(params.get("height", params.get("size", 1024))),
            steps=int(params.get("steps", 8)),
            seed=int(params.get("seed", 0)),
            num_images=int(params.get("num_images", 1)),
            step_callback=step_callback,
            **kwargs,
        )

    # --- model management ---
    @staticmethod
    def _transformer_path(variant: str):
        from krea2.pipeline import _CACHE, BUILDS
        repo, fname = BUILDS[variant]
        return repo, fname, os.path.join(_CACHE, repo.replace("/", "__"), fname)

    def is_installed(self, variant: str) -> bool:
        _, fname, cache = self._transformer_path(variant)
        return os.path.exists(cache) or os.path.exists(os.path.join(os.getcwd(), fname))

    def _specs(self, variant: str):
        """(url, dest) for the variant's transformer + the shared base (encoder/VAE/tokenizer)."""
        from huggingface_hub import HfApi
        from krea2.pipeline import _CACHE, BASE_REPO

        def url(repo, f):
            return f"https://huggingface.co/{repo}/resolve/main/{f}"

        repo, fname, cache = self._transformer_path(variant)
        specs = [(url(repo, fname), cache)]
        base_root = os.path.join(_CACHE, BASE_REPO.replace("/", "__"))
        exts = (".safetensors", ".json", ".jinja", ".txt", ".model")
        for s in HfApi().model_info(BASE_REPO).siblings:
            f = s.rfilename
            if (f.startswith(("vae/", "text_encoder/", "tokenizer/")) or f == "model_index.json") and f.endswith(exts):
                specs.append((url(BASE_REPO, f), os.path.join(base_root, f)))
        return specs

    def download(self, variant: str, progress) -> None:
        from studio.download import download_files
        download_files(self._specs(variant), progress)

    def delete(self, variant: str) -> None:
        _, _, cache = self._transformer_path(variant)
        if os.path.exists(cache):
            os.remove(cache)
        if self._variant == variant:  # drop the loaded build if we just deleted it
            self._pipe, self._variant = None, None
