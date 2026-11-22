"""Astra (gpt-6-astra) client over the Sign-in-with-ChatGPT subscription route.

One streaming Responses request per call, strict JSON-schema output, bounded
retries on transient transport errors (HTTP 429/5xx / network), full request
and response logging with images stored as files instead of base64.
"""
from __future__ import annotations

import base64
import concurrent.futures as cf
import hashlib
import json
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

import cv2
import numpy as np

from .chatgpt_auth import SubscriptionCredentials

ENDPOINT = "https://api.openai.com/v1/responses"
MODEL = "gpt-6-astra"


class AstraError(RuntimeError):
    def __init__(self, msg, status=None, code=None, retryable=False):
        super().__init__(msg)
        self.status, self.code, self.retryable = status, code, retryable


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise AstraError("endpoint redirected; stopped", code="redirect")


def text_part(s):
    return {"type": "input_text", "text": s}


def image_part(rgb, detail="high"):
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(rgb[:, :, ::-1]))
    assert ok
    return {"type": "input_image", "detail": detail,
            "image_url": "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()}


def _sse_events(resp):
    name, data = None, []
    for raw in resp:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield name, "\n".join(data)
            name, data = None, []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            name = value
        elif field == "data":
            data.append(value)
    if data:
        raise AstraError("stream ended inside an event", code="stream_truncated", retryable=True)


def _read_stream(resp):
    done_items = {}
    first_out = None
    t0 = time.time()
    for _name, raw in _sse_events(resp):
        if raw == "[DONE]":
            break
        ev = json.loads(raw)
        kind = ev.get("type", "")
        if kind in ("response.failed", "response.incomplete", "error", "response.error"):
            r = ev.get("response") or {}
            err = r.get("error") if isinstance(r, dict) else None
            err = err or ev.get("error") or {}
            code = err.get("code") if isinstance(err, dict) else None
            raise AstraError(f"stream {kind}: {code}", code=code, retryable=kind != "response.incomplete")
        if kind == "response.output_text.delta" and first_out is None:
            first_out = time.time() - t0
        if kind == "response.output_item.done":
            done_items[int(ev.get("output_index", len(done_items)))] = ev.get("item")
        if kind == "response.completed":
            r = ev["response"]
            out = r.get("output") or [done_items[k] for k in sorted(done_items)]
            r["output"] = out
            r["_first_text_s"] = first_out
            return r
    raise AstraError("stream ended without response.completed", code="stream_incomplete", retryable=True)


def _extract_text(response):
    texts = []
    for item in response.get("output", []):
        if isinstance(item, dict) and item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    texts.append(part["text"])
                elif part.get("type") == "refusal":
                    raise AstraError("model refusal", code="refusal")
    if not texts:
        raise AstraError("no output text", code="empty_output", retryable=True)
    return "".join(texts)


class AstraClient:
    def __init__(self, log_dir, store_dir=None, effort="medium", max_calls=12, timeout_s=300,
                 model=MODEL, retries=2):
        self.creds = SubscriptionCredentials(store_dir)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.effort = effort
        self.max_calls = max_calls
        self.timeout_s = timeout_s
        self.model = model
        self.retries = retries
        self.calls = []
        self._lock = threading.Lock()
        self._pool = cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="astra")

    # ------------------------------------------------------------------ logging
    def _strip_images(self, content, tag):
        out = []
        for i, part in enumerate(content):
            if part.get("type") == "input_image":
                url = part["image_url"]
                raw = base64.b64decode(url.split(",", 1)[1])
                h = hashlib.sha1(raw).hexdigest()[:12]
                path = self.log_dir / f"{tag}_img{i}_{h}.png"
                path.write_bytes(raw)
                out.append({"type": "input_image", "file": path.name, "detail": part.get("detail")})
            else:
                out.append(part)
        return out

    # --------------------------------------------------------------------- call
    def call(self, kind, instructions, content, schema, effort=None):
        with self._lock:
            n = len(self.calls) + 1
            if n > self.max_calls:
                raise AstraError("model call budget exhausted", code="budget")
            rec = {"n": n, "kind": kind, "effort": effort or self.effort, "t_start": time.time()}
            self.calls.append(rec)
        tag = f"{n:02d}_{kind}"
        payload = {"model": self.model, "service_tier": "default", "store": False, "stream": True,
                   "instructions": instructions,
                   "input": [{"role": "user", "content": content}],
                   "text": {"format": {"type": "json_schema", "name": kind, "strict": True, "schema": schema}},
                   "reasoning": {"effort": effort or self.effort}}
        (self.log_dir / f"{tag}.request.json").write_text(json.dumps(
            {**payload, "input": [{"role": "user", "content": self._strip_images(content, tag)}]}, indent=1))
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        attempt = 0
        while True:
            attempt += 1
            t0 = time.time()
            try:
                token = self.creds.access_token()
                req = Request(ENDPOINT, data=body, method="POST", headers={
                    "Authorization": f"Bearer {token}", "Content-Type": "application/json",
                    "Accept": "text/event-stream"})
                try:
                    resp = build_opener(_NoRedirect()).open(req, timeout=self.timeout_s)
                except HTTPError as e:
                    code = None
                    try:
                        code = (json.loads(e.read().decode()).get("error") or {}).get("code")
                    except Exception:
                        pass
                    raise AstraError(f"HTTP {e.code} {code}", status=e.code, code=code,
                                     retryable=e.code in (408, 409, 429, 500, 502, 503, 504)) from None
                except (URLError, TimeoutError, OSError) as e:
                    raise AstraError(f"network {type(e).__name__}", code="network", retryable=True) from None
                with resp:
                    response = _read_stream(resp)
                text = _extract_text(response)
                data = json.loads(text)
                usage = response.get("usage") or {}
                rec.update({"status": "ok", "attempts": attempt, "latency_s": round(time.time() - t0, 2),
                            "first_text_s": response.get("_first_text_s"),
                            "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                            "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens")})
                (self.log_dir / f"{tag}.response.json").write_text(json.dumps(
                    {"output": data, "usage": usage, "model": response.get("model"), "record": rec}, indent=1))
                return data, rec
            except AstraError as e:
                rec.setdefault("errors", []).append({"attempt": attempt, "error": str(e), "status": e.status,
                                                     "t": round(time.time() - t0, 2)})
                if e.retryable and attempt <= self.retries:
                    time.sleep(4.0 * attempt)
                    continue
                rec["status"] = "error"
                (self.log_dir / f"{tag}.response.json").write_text(json.dumps({"record": rec}, indent=1))
                raise
            except json.JSONDecodeError:
                rec.setdefault("errors", []).append({"attempt": attempt, "error": "invalid json output"})
                if attempt <= self.retries:
                    continue
                rec["status"] = "error"
                raise AstraError("invalid JSON output", code="json")

    def submit(self, kind, instructions, content, schema, effort=None):
        return self._pool.submit(self.call, kind, instructions, content, schema, effort)

    def stats(self):
        ok = [c for c in self.calls if c.get("status") == "ok"]
        return {"calls": len(self.calls), "ok": len(ok),
                "latency_s": [c.get("latency_s") for c in self.calls],
                "kinds": [c["kind"] for c in self.calls],
                "output_tokens": sum((c.get("output_tokens") or 0) for c in ok),
                "reasoning_tokens": sum((c.get("reasoning_tokens") or 0) for c in ok)}
