"""HTTP-bridge file downloader with a progress callback.

Mirrors the krea2 package's resilient downloader (plain HTTP to the HF CDN, resumable, integrity-
checked) but reports progress to a callback instead of printing — so the model manager can show a
live bar. We bypass huggingface_hub's Xet client on purpose (it can hang behind some firewalls).
"""

from __future__ import annotations

import os


def _head_size(url: str) -> int:
    import requests
    try:
        return int(requests.head(url, allow_redirects=True, timeout=30).headers.get("content-length") or 0)
    except Exception:
        return 0


def _download_one(url: str, dest: str, total: int, on_bytes) -> None:
    import requests
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    pos = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if total and pos == total:        # complete .part left by a crash before the rename
        os.replace(tmp, dest)
        on_bytes(total)
        return
    if total and pos > total:         # stale/corrupt leftover
        os.remove(tmp)
        pos = 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120), allow_redirects=True) as r:
        r.raise_for_status()
        resume = bool(pos) and r.status_code == 206
        pos = pos if resume else 0
        total = total or (pos + int(r.headers.get("content-length") or 0))
        done = pos
        with open(tmp, "ab" if resume else "wb") as f:
            for chunk in r.iter_content(4 << 20):
                f.write(chunk)
                done += len(chunk)
                on_bytes(done)
    if total and done != total:
        raise OSError(f"incomplete download of {os.path.basename(dest)} ({done}/{total} bytes)")
    os.replace(tmp, dest)


def download_files(specs, progress) -> None:
    """specs: list of (url, dest). progress(done_total_bytes, grand_total_bytes) is called as bytes land."""
    sizes = [_head_size(u) for u, _ in specs]
    grand = sum(sizes)
    base = 0
    for (url, dest), sz in zip(specs, sizes):
        if os.path.exists(dest) and sz and os.path.getsize(dest) == sz:
            base += sz
            progress(base, grand)
            continue
        start = base
        _download_one(url, dest, sz, lambda d: progress(start + d, grand))
        base += sz
        progress(base, grand)
