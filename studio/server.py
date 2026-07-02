"""Alis Studio — local HTTP server.

Dependency-free (Python standard library): serves web/index.html, exposes the registered
models at /api/models, and streams generation progress as NDJSON from /api/generate. Binds to
127.0.0.1 by default — set ALIS_HOST=0.0.0.0 to expose on your LAN, ALIS_PORT to change the port.
"""

from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__
from .registry import Registry

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEB = os.path.join(ROOT, "web")
HOST = os.environ.get("ALIS_HOST", "127.0.0.1")
PORT = int(os.environ.get("ALIS_PORT") or os.environ.get("PORT") or "7860")

_REGISTRY: Registry | None = None
_LOCK = threading.Lock()    # one GPU — serialize generation
_DLLOCK = threading.Lock()  # serialize downloads (shared .part files)
_CANCEL = threading.Event()  # set by POST /api/cancel — the in-flight generation stops at its next step

# MLX's default GPU stream is per-thread with a global index, and krea2 hard-references stream
# index 0. The stdlib ThreadingHTTPServer hands each request a fresh thread, so generation on a
# request thread otherwise dies with "There is no Stream(gpu, 0) in current thread". Fix: run
# EVERY MLX op (generation, NSFW filter, the model-cache scan) on ONE dedicated thread, and warm
# that thread up FIRST (in start_http, before anything else imports mlx) so it claims index 0.
# (The prompt enhancer's mlx-lm runs in a separate process entirely — see studio/prompt_rewrite.py.)
_GPU = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="alis-gpu")


def _gpu(fn, *args, **kwargs):
    """Run an MLX-touching callable on the one GPU thread and return its result."""
    return _GPU.submit(fn, *args, **kwargs).result()


def _warmup_gpu():
    import mlx.core as mx
    mx.eval(mx.zeros(1))   # claim GPU stream 0 for this thread before any other import touches mlx


class _Cancelled(Exception):
    """Raised from the step callback when the user clicks Stop."""


def _registry() -> Registry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = Registry()
    return _REGISTRY


_UPSCALER = None


def _upscaler():
    global _UPSCALER
    if _UPSCALER is None:
        from .backends.upscale import Upscaler
        _UPSCALER = Upscaler()
    return _UPSCALER


def _png_datauri(im) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _apply_safety(images, enabled):
    """App-level NSFW filter (reuses krea2's pure-MLX classifier). Passes through if unavailable."""
    if not enabled:
        return images, 0
    try:
        from krea2 import safety
        return safety.apply(images, enabled=True)
    except Exception:
        return list(images), 0


def _datauri_to_temp(data_uri, prefix):
    """Decode a base64 data: URI to a temp PNG file and return its path (or None). Caller removes it."""
    if not isinstance(data_uri, str) or not data_uri.startswith("data:"):
        return None
    try:
        import tempfile
        raw = base64.b64decode(data_uri.split(",", 1)[1])
        fd, path = tempfile.mkstemp(prefix=prefix, suffix=".png")
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        return path
    except Exception:
        return None


def _stash_input_image(params):
    """For img2img: decode an uploaded image (data URI in params['init_image']) to a temp PNG and set
    params['image_path'] (what the backends' generate expects). Returns the temp path, or None.
    An attached image that fails to decode raises — silently generating WITHOUT the image the user
    attached would be far more confusing than an error."""
    value = params.pop("init_image", None)
    path = _datauri_to_temp(value, "alis_in_")
    if path:
        params["image_path"] = path
    elif value:
        raise ValueError("Couldn't decode the attached image — remove it and attach it again.")
    return path


def _disk_gb() -> float:
    """Total size of the local model cache, in GB."""
    from krea2.pipeline import _CACHE
    total = 0
    for root, _, files in os.walk(_CACHE):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return round(total / 1e9, 1)


ENHANCE_RAM_GIB = 24            # at/below this, the prompt enhancer + image model may strain memory
_RAM_GIB: float | None = None


def _ram_gib() -> float:
    """Total physical RAM in GiB (cached)."""
    global _RAM_GIB
    if _RAM_GIB is None:
        try:  # canonical on macOS
            import subprocess
            out = subprocess.run(["/usr/sbin/sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True, timeout=2)
            _RAM_GIB = int(out.stdout.strip()) / (1024 ** 3)
        except Exception:
            try:
                _RAM_GIB = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
            except Exception:
                _RAM_GIB = 0.0
    return _RAM_GIB


# Quality-preference order for the "recommended for your Mac" hint: flagship first. The pick is the
# first of these that's installed AND whose RAM floor (Backend.min_ram_gib) this machine clears.
# Text-to-image models only — "qwen-image-edit" is intentionally omitted: it needs an input image,
# so it's never a sensible default pick even on a large Mac.
_RECOMMEND_ORDER = ["krea2-turbo", "qwen-image", "flux-dev", "z-image-turbo", "flux-schnell"]


def _recommended_model(ram: float):
    """The most capable model that fits this Mac's RAM (by _RECOMMEND_ORDER), falling back to the
    lightest available model if nothing's floor is met. Returns (id, label); ('', '') if no models."""
    backs = _registry().backends
    for bid in _RECOMMEND_ORDER:
        b = backs.get(bid)
        if b is not None and getattr(b, "min_ram_gib", 0) <= ram:
            return b.id, b.label
    if backs:
        b = min(backs.values(), key=lambda x: getattr(x, "min_ram_gib", 0))
        return b.id, b.label
    return "", ""


def _system() -> dict:
    """Capabilities the UI needs: gate the optional prompt enhancer, and recommend a model for this Mac."""
    from . import prompt_rewrite
    from .backends.upscale import Upscaler
    ram = _ram_gib()
    rec_id, rec_label = _recommended_model(ram)
    return {"ram_gib": round(ram, 1),
            "constrained": 0 < ram <= ENHANCE_RAM_GIB,
            "enhance_available": prompt_rewrite.is_available(),
            "enhance_model": prompt_rewrite.MODEL,
            "recommended": rec_id, "recommended_label": rec_label,
            # gate upscale on RAM too: SeedVR2-3B + a 2–4 MP target on top of a cached model needs room
            "upscale_available": Upscaler.is_available() and ram >= Upscaler.min_ram_gib}


def _catalog() -> dict:
    backs = []
    for b in _registry().backends.values():
        entries = [{**c, "backend": b.id, "installed": b.is_installed(c["variant"])}
                   for c in getattr(b, "catalog", [])]
        if entries:
            backs.append({"id": b.id, "label": b.label, "entries": entries})
    try:
        disk = _gpu(_disk_gb)   # imports krea2.pipeline (touches mlx) — keep it on the GPU thread
    except Exception:
        disk = 0.0
    return {"disk_gb": disk, "backends": backs}


def _gallery_dir() -> str:
    d = os.path.join(os.path.expanduser("~/Library/Application Support/Alis Studio"), "gallery")
    os.makedirs(d, exist_ok=True)
    return d


def _safe_id(gid: str) -> bool:
    return bool(gid) and all(c.isalnum() or c in "-_" for c in gid)


def _gallery_write(im, gid, ts, prompt, meta) -> str:
    """The one writer: persist a single image + its metadata json to the gallery. Returns the id."""
    d = _gallery_dir()
    im.save(os.path.join(d, gid + ".png"))
    with open(os.path.join(d, gid + ".json"), "w") as f:
        json.dump({"id": gid, "prompt": prompt, "ts": ts, **meta}, f)
    return gid


def _gallery_save(images, prompt, meta) -> None:
    """Persist each generated image + its metadata to the on-disk gallery (best-effort)."""
    base = int(time.time() * 1000)
    for i, im in enumerate(images):
        _gallery_write(im, f"{base}-{int(meta.get('seed', 0))}-{i}", base, prompt, meta)


def _gallery_save_one(im, prompt, meta) -> str:
    """Persist a single image and return its id (used by upscale)."""
    ts = int(time.time() * 1000)
    return _gallery_write(im, f"{ts}-{int(meta.get('seed', 0))}-0", ts, prompt, meta)


def _gallery_list() -> list:
    """All saved generations, newest first."""
    d = _gallery_dir()
    items = []
    for fn in os.listdir(d):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(d, fn)) as f:
                    items.append(json.load(f))
            except Exception:
                pass
    items.sort(key=lambda r: r.get("ts", 0), reverse=True)
    return items


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body=b""):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if body:
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            try:
                with open(os.path.join(WEB, "index.html"), "rb") as f:
                    html = f.read().replace(b"__ALIS_VERSION__", __version__.encode())
                self._send(200, "text/html; charset=utf-8", html)
            except OSError:
                self._send(500, "text/plain", b"web/index.html is missing")
        elif path == "/api/version":
            self._send(200, "application/json", json.dumps({"version": __version__}).encode())
        elif path == "/api/system":
            self._send(200, "application/json", json.dumps(_system()).encode())
        elif path == "/api/models":
            self._send(200, "application/json", json.dumps({"backends": _registry().models()}).encode())
        elif path == "/api/catalog":
            self._send(200, "application/json", json.dumps(_catalog()).encode())
        elif path == "/api/gallery":
            self._send(200, "application/json", json.dumps({"items": _gallery_list()}).encode())
        elif path.startswith("/api/gallery/") and path.endswith(".png"):
            gid = path[len("/api/gallery/"):-4]
            if _safe_id(gid):
                try:
                    with open(os.path.join(_gallery_dir(), gid + ".png"), "rb") as f:
                        self._send(200, "image/png", f.read())
                except OSError:
                    self._send(404, "text/plain", b"not found")
            else:
                self._send(404, "text/plain", b"not found")
        elif path == "/favicon.ico":
            self._send(204, "text/plain")
        else:
            self._send(404, "text/plain", b"not found")

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(obj):
            self.wfile.write((json.dumps(obj) + "\n").encode())
            self.wfile.flush()
        return emit

    def do_POST(self):
        if self.path == "/api/cancel":           # no body — just flag the running generation to stop
            _CANCEL.set()
            self._send(200, "application/json", b'{"ok":true}')
            return
        if self.path not in ("/api/generate", "/api/upscale", "/api/download", "/api/delete", "/api/enhance", "/api/gallery/delete"):
            self._send(404, "text/plain", b"not found")
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 128 * 1024 * 1024:   # cap the body (uploaded images are base64) — don't let a huge POST OOM us
                self._send(413, "text/plain", b"request too large")
                return
            req = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send(400, "text/plain", b"bad request")
            return
        if self.path == "/api/generate":
            self._generate(req)
        elif self.path == "/api/upscale":
            self._upscale(req)
        elif self.path == "/api/enhance":
            self._enhance(req)
        elif self.path == "/api/download":
            self._download(req)
        elif self.path == "/api/gallery/delete":
            self._gallery_delete(req)
        else:
            self._delete(req)

    def _gallery_delete(self, req):
        gid = str(req.get("id", ""))
        if not _safe_id(gid):
            self._send(400, "application/json", b'{"ok":false}')
            return
        d = _gallery_dir()
        for ext in (".png", ".json"):
            try:
                os.remove(os.path.join(d, gid + ext))
            except OSError:
                pass
        self._send(200, "application/json", b'{"ok":true}')

    def _generate(self, req):
        emit = self._stream()

        def safe_emit(obj):           # the client may have gone away (Stop / closed tab)
            try:
                emit(obj)
            except OSError:
                pass

        with _LOCK:
            _CANCEL.clear()           # fresh generation — forget any earlier Stop request
            img_tmp = None            # temp file for an uploaded img2img input; removed in finally

            def step(s, total):       # called once per denoise step; the Stop hook lives here
                if _CANCEL.is_set():
                    raise _Cancelled()
                emit({"type": "step", "step": s, "total": total})

            try:
                backend, variant = _registry().resolve(req.get("model", ""))
                # Hard RAM gate: refuse a build whose memory floor this Mac can't meet, BEFORE mflux
                # fetches gigabytes of weights only to OOM at load. The UI shows a confirm dialog and
                # sends allow_low_ram=true to override; power users can set ALIS_ALLOW_LOW_RAM. A failed
                # RAM probe (ram==0) is treated as "unknown" and allowed rather than blocking blindly.
                ram = _ram_gib()
                floor = backend.min_ram_for(variant)
                allow_low = req.get("allow_low_ram") or os.environ.get("ALIS_ALLOW_LOW_RAM")
                if ram and floor and floor > ram and not allow_low:
                    raise ValueError(
                        f"{backend.label} needs about {floor} GB of memory, but this Mac has "
                        f"{ram:.0f} GB. It would download several GB and then run out of memory. "
                        f"Run it on a Mac with more unified memory.")
                params = dict(req.get("params") or {})
                w = int(params.get("width") or params.get("size") or 1024)
                h = int(params.get("height") or params.get("size") or 1024)
                params["width"], params["height"] = w, h
                img_tmp = _stash_input_image(params)   # img2img: decode uploaded image → params["image_path"]
                t0 = time.time()

                def _job():           # all MLX work (load, denoise, NSFW filter) on the one GPU thread
                    if backend.will_load(variant):   # model not in memory → load (first ever use also downloads)
                        safe_emit({"type": "status",
                                   "message": f"Loading {backend.label}… (first use may download weights)"})
                    out = backend.generate(prompt=str(req.get("prompt", "")), variant=variant,
                                           params=params, step_callback=step)
                    if _CANCEL.is_set():  # stopped after the last step, or by a backend that doesn't tick
                        raise _Cancelled()
                    return _apply_safety(out, req.get("safety", True))

                imgs, flagged = _gpu(_job)
                # report the ACTUAL output size — edit models normalize to ~1 MP and others may round
                # to a multiple of 16, so the produced image can differ from the requested w/h
                ow, oh = (imgs[0].width, imgs[0].height) if imgs else (w, h)
                meta = {"model": backend.label, "width": ow, "height": oh,
                        "steps": int(params.get("steps", 8)), "seed": int(params.get("seed", 0)),
                        "seconds": round(time.time() - t0, 1)}
                try:
                    _gallery_save(imgs, str(req.get("prompt", "")), meta)   # best-effort history
                except Exception:
                    pass
                emit({"type": "done", "flagged": flagged,
                      "images": [_png_datauri(im) for im in imgs], "meta": meta})
            except _Cancelled:
                safe_emit({"type": "cancelled"})
            except (BrokenPipeError, ConnectionResetError):
                return                # client disconnected mid-stream — nothing left to send
            except ValueError as e:
                safe_emit({"type": "error", "message": str(e)})
            except Exception as e:
                if _CANCEL.is_set():  # a backend that wrapped the _Cancelled into its own error
                    safe_emit({"type": "cancelled"})
                    return
                m = str(e).lower()
                if any(k in m for k in ("memory", "alloc", "metal")):
                    safe_emit({"type": "error", "message": "Out of memory — try a smaller size, fewer "
                               "images, or a lighter model build."})
                else:
                    safe_emit({"type": "error", "message": f"Generation failed: {e}"})
            finally:
                if img_tmp:
                    try:
                        os.remove(img_tmp)
                    except OSError:
                        pass

    def _upscale(self, req):
        emit = self._stream()

        def safe_emit(obj):
            try:
                emit(obj)
            except OSError:
                pass

        scale = 3 if int(req.get("scale", 2) or 2) == 3 else 2
        # resolve the source image to a local path: a gallery id, or an uploaded data URI
        src_path, src_tmp, orig_prompt = None, None, str(req.get("prompt", "") or "")
        gid = str(req.get("id", ""))
        if gid and _safe_id(gid):
            p = os.path.join(_gallery_dir(), gid + ".png")
            if os.path.exists(p):
                src_path = p
                try:
                    with open(os.path.join(_gallery_dir(), gid + ".json")) as f:
                        orig_prompt = json.load(f).get("prompt", orig_prompt)
                except Exception:
                    pass
        if src_path is None:
            src_tmp = _datauri_to_temp(req.get("image"), "alis_up_")
            src_path = src_tmp
        if not src_path:
            safe_emit({"type": "error", "message": "No image to upscale."})
            return

        up = _upscaler()
        with _LOCK:
            t0 = time.time()
            try:
                def _job():
                    if up.will_load():
                        safe_emit({"type": "status", "message": "Loading the SeedVR2 upscaler… (first use downloads ~7 GB)"})
                    safe_emit({"type": "status", "message": f"Upscaling {scale}×…"})
                    return _apply_safety([up.upscale(src_path, scale)], req.get("safety", True))

                imgs, flagged = _gpu(_job)
                im = imgs[0]
                meta = {"model": f"SeedVR2 · {scale}× upscale", "width": im.width, "height": im.height,
                        "steps": 0, "seed": 0, "seconds": round(time.time() - t0, 1)}
                newid = None
                try:
                    newid = _gallery_save_one(im, orig_prompt, meta)
                except Exception:
                    pass
                emit({"type": "done", "flagged": flagged, "image": _png_datauri(im), "id": newid, "meta": meta})
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as e:
                m = str(e).lower()
                if any(k in m for k in ("memory", "alloc", "metal")):
                    safe_emit({"type": "error", "message": "Out of memory upscaling — try 2× instead of 3×, or a smaller image."})
                else:
                    safe_emit({"type": "error", "message": f"Upscale failed: {e}"})
            finally:
                if src_tmp:
                    try:
                        os.remove(src_tmp)
                    except OSError:
                        pass

    def _enhance(self, req):
        from . import prompt_rewrite
        prompt = str(req.get("prompt", ""))
        if not prompt.strip():
            self._send(400, "application/json", b'{"error":"empty prompt"}')
            return
        with _LOCK:   # one GPU — don't enhance while a generation is running (first call also loads ~2.3 GB)
            try:
                rewritten = prompt_rewrite.enhance(prompt)   # runs in an isolated worker process (own MLX state)
                self._send(200, "application/json", json.dumps({"rewritten": rewritten}).encode())
            except RuntimeError as e:   # mlx-lm not installed
                self._send(503, "application/json", json.dumps({"error": str(e)}).encode())
            except Exception as e:
                self._send(500, "application/json", json.dumps({"error": f"Enhance failed: {e}"}).encode())

    def _download(self, req):
        emit = self._stream()
        backend = _registry().backends.get(req.get("backend"))
        if backend is None:
            emit({"type": "error", "message": "unknown backend"})
            return
        state = {"last": -1}

        def prog(done, total):
            if done - state["last"] >= 33554432 or done == total:  # throttle to ~32 MB
                state["last"] = done
                emit({"type": "progress", "done": done, "total": total})

        with _DLLOCK:
            try:
                backend.download(req.get("variant", ""), prog)
                emit({"type": "done"})
            except Exception as e:
                emit({"type": "error", "message": f"Download failed: {e}"})

    def _delete(self, req):
        backend = _registry().backends.get(req.get("backend"))
        if backend is None:
            self._send(404, "application/json", b'{"ok":false,"error":"unknown backend"}')
            return
        try:
            with _DLLOCK, _LOCK:  # exclusive with an in-flight download (file) and generate (pipe ref)
                backend.delete(req.get("variant", ""))
            self._send(200, "application/json", b'{"ok":true}')
        except Exception as e:
            self._send(500, "application/json", json.dumps({"ok": False, "error": str(e)}).encode())


def start_http(host=HOST, port=PORT):
    """Build the registry and run the HTTP server in a background daemon thread.
    Returns (server, bound_port). Pass port=0 to bind a free port (the native window does this)."""
    _gpu(_warmup_gpu)   # claim a GPU stream on our dedicated thread first
    _gpu(_registry)     # build the registry (imports mflux/krea2 → mlx) on that SAME thread, so every
                        # mlx import and op shares one thread's stream; also makes startup fail loudly
    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def serve():
    server, port = start_http(HOST, PORT)
    url = f"http://{'localhost' if HOST in ('127.0.0.1', '0.0.0.0') else HOST}:{port}"
    print(f"Alis Studio  →  {url}")
    print("First run downloads the model (a few minutes). Press Ctrl+C to stop.")
    if HOST in ("127.0.0.1", "0.0.0.0"):
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nstopped.")
        server.shutdown()
