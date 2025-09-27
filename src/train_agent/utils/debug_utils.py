# @title Debug utilities

import json
import time
import traceback
from typing import Any

DEBUG_LOG = True  # flip to False to silence logs
LOG_JSON_MAX = 2000  # cap large JSON prints


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str, **kv):
    if not DEBUG_LOG:
        return
    parts = [f"[{_ts()}] {msg}"]
    if kv:
        kv_str = " ".join(f"{k}={repr(v)}" for k, v in kv.items())
        parts.append("| " + kv_str)
    print(" ".join(parts))


def log_json(title: str, payload: Any, max_len: int = LOG_JSON_MAX):
    if not DEBUG_LOG:
        return
    try:
        s = json.dumps(payload, indent=2, default=str)
    except Exception:
        s = str(payload)
    if len(s) > max_len:
        s = s[:max_len] + "\n... (truncated)"
    print(f"[{_ts()}] {title}:\n{s}")