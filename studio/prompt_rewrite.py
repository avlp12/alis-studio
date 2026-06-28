"""Optional local prompt enhancer — translate a non-English prompt to English and lightly enrich
it into a vivid image caption, using a small on-device MLX LLM (Qwen3-4B-Instruct-2507, 4-bit).

The mlx-lm model runs in a SEPARATE worker process (studio/_enhance_worker.py). That isolation is
required, not cosmetic: running mlx-lm in the same process as the image model corrupts MLX's GPU
stream state ("There is no Stream(gpu, 0) in current thread"). The worker loads the model once and
stays resident, so only the first call pays the load cost.

OPTIONAL: mlx-lm is checked lazily; the dependency-light server runs fine without it. Enable with
`pip install "alis-studio[rewrite]"`. Everything is on-device — no prompt ever leaves the machine.
The first call downloads the model (~2.3 GB) from Hugging Face.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading

MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

_LOCK = threading.Lock()       # serialize access to the single worker
_PROC: subprocess.Popen | None = None
_CACHE: dict[str, str] = {}    # prompt -> rewritten; greedy + cached == deterministic & reproducible


def is_available() -> bool:
    """True if mlx-lm is importable (the optional `rewrite` extra is installed)."""
    return importlib.util.find_spec("mlx_lm") is not None


def _worker() -> subprocess.Popen:
    global _PROC
    if _PROC is None or _PROC.poll() is not None:
        _PROC = subprocess.Popen(
            [sys.executable, "-m", "studio._enhance_worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
    return _PROC


def enhance(prompt: str) -> str:
    """Translate-to-English + lightly enrich `prompt` via the isolated worker. Deterministic
    (greedy) and cached, so the same input always yields the same English — preserving seed
    reproducibility. Raises RuntimeError with an actionable message if mlx-lm isn't installed."""
    prompt = (prompt or "").strip()
    if not prompt:
        return prompt
    if prompt in _CACHE:
        return _CACHE[prompt]
    if not is_available():
        raise RuntimeError('Prompt enhancer needs mlx-lm — install with: pip install "alis-studio[rewrite]"')
    with _LOCK:
        if prompt in _CACHE:        # filled while waiting on the lock
            return _CACHE[prompt]
        p = _worker()
        try:
            p.stdin.write(json.dumps({"prompt": prompt}) + "\n")
            p.stdin.flush()
            line = p.stdout.readline()
        except (BrokenPipeError, OSError):
            line = ""
        if not line:
            raise RuntimeError("the prompt-enhancer worker exited unexpectedly")
        resp = json.loads(line)
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        out = (resp.get("rewritten") or "").strip() or prompt
        _CACHE[prompt] = out
        return out
