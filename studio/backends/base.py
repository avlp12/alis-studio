"""Backend interface — a pluggable image-generation model.

To add a model to Alis Studio, drop a `studio/backends/<name>.py` that subclasses Backend,
declares its settings (`params`) and downloadable builds (`catalog`), implements `generate(...)`,
and register it in `studio/registry.py`. The web UI discovers everything via `/api/models` and
`/api/catalog` and renders itself — no UI changes needed to add a model.

`params` is a list of control specs the UI renders into the settings panel and passes back to
`generate(..., params=...)`. Each spec: {key, label, type, group, default, ...type-specific}. Types:
  - "resolution": {sizes:[...], default_size, aspects:["1:1",...], min, max, multiple} → emits width,height
  - "int" / "float": {min, max, step, default}
  - "select": {options:[{value,label}], default}
  - "seed": {default}                      (number + randomize + lock)
  - "text": {default}
Flags: "fixed": shown read-only; "enabled": False shown disabled; "hint": one-line note.
"""

from __future__ import annotations


class Backend:
    id = ""                      # stable id, e.g. "krea2-turbo"
    label = ""                   # display name, e.g. "Krea 2 Turbo"
    info = ""                    # one-line note shown for non-downloadable models (e.g. "auto-downloads on first use")
    prompt_note = ""             # one-line hint about prompt language (shown under the prompt box)
    variants: list[dict] = []    # selectable builds: [{"id": "8bit", "label": "8-bit · best quality"}]
    params: list[dict] = []      # settings schema (see module docstring)
    catalog: list[dict] = []     # downloadable builds (see studio/registry conventions); empty = managed elsewhere
    min_ram_gib = 0              # unified-memory floor to run this model acceptably; drives the "recommended for your Mac" hint

    @classmethod
    def is_available(cls) -> bool:
        """True if this backend's code dependencies are importable. Unavailable backends are
        skipped at registration so the app still runs with whatever is installed."""
        return True

    def meta(self) -> dict:
        return {"id": self.id, "label": self.label, "info": self.info,
                "prompt_note": self.prompt_note, "variants": self.variants, "params": self.params}

    def generate(self, *, prompt: str, variant: str, params: dict, step_callback):
        """Return a list of PIL.Image. `params` carries the resolved settings (width, height,
        steps, seed, num_images, and any model-specific keys). Call step_callback(step, total)
        per denoising step so the UI can show live progress."""
        raise NotImplementedError

    def will_load(self, variant: str) -> bool:
        """True if the next generate() will construct/load this variant's model into memory (it isn't
        the one already cached on this backend) — a multi-second wait the UI should explain. The very
        first time a model is used it also downloads its weights; later loads come from disk cache.
        Safe to consult off the GPU thread: generation is single-flight (serialized under _LOCK + the
        one GPU thread), so the state can't change before _get() runs. Default False (no load step)."""
        return False

    # --- model management (override for downloadable backends) ---
    def is_installed(self, variant: str) -> bool:
        return True

    def download(self, variant: str, progress) -> None:
        """Download `variant`'s weights, calling progress(done_bytes, total_bytes) as it goes."""
        raise NotImplementedError

    def delete(self, variant: str) -> None:
        raise NotImplementedError
