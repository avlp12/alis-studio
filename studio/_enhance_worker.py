"""Subprocess worker for the prompt enhancer.

Runs mlx-lm (Qwen3) in its OWN process so its MLX/GPU stream state stays fully isolated from the
image-generation process. Mixing mlx-lm and the image model's MLX in one process corrupts MLX's
per-thread GPU stream and raises "There is no Stream(gpu, 0) in current thread".

Protocol (one JSON object per line): reads {"prompt": "..."} from stdin, writes {"rewritten": "..."}
or {"error": "..."} to stdout. All library noise (download bars, load logs) is sent to stderr so it
can never corrupt the stdout protocol.
"""

import json
import os
import sys

MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

_SYSTEM = (
    "You rewrite text-to-image prompts. Given the user's prompt in any language:\n"
    "- If it is not English, translate it into natural English.\n"
    "- Lightly enrich it into a vivid, concrete image caption (subject, setting, lighting, mood, "
    "style) without inventing unrelated content or changing the user's intent.\n"
    "- Reply with ONLY the rewritten English prompt — no quotes, no preface, no explanation."
)


def main():
    # Reserve a clean channel for the protocol: duplicate the real stdout, then point fd 1 at
    # stderr so anything the libraries print can't corrupt our JSON line protocol.
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model = tok = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            prompt = (json.loads(line).get("prompt") or "").strip()
            if not prompt:
                out.write(json.dumps({"rewritten": ""}) + "\n"); out.flush(); continue
            if model is None:
                model, tok = load(MODEL)
            text = tok.apply_chat_template(
                [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
                add_generation_prompt=True,
            )
            res = generate(model, tok, text, max_tokens=256, sampler=make_sampler(temp=0.0), verbose=False)
            res = (res or "").strip()
            if "</think>" in res:            # defensive: drop any reasoning block
                res = res.rsplit("</think>", 1)[-1].strip()
            res = res.strip('"').strip()
            out.write(json.dumps({"rewritten": res or prompt}) + "\n"); out.flush()
        except Exception as e:
            out.write(json.dumps({"error": str(e)}) + "\n"); out.flush()


if __name__ == "__main__":
    main()
