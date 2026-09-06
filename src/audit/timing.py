"""
Misurazione dei tempi per singola fase, per verificare i budget di
performance del brief: <2s dalla domanda al primo token prodotto dall'LLM,
<8s per la risposta completa, su laptop moderno.
"""
from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

from config import settings

_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def new_request_id() -> str:
    """Genera un nuovo request_id"""
    rid = uuid.uuid4().hex[:8]
    _request_id_var.set(rid)
    return rid


def current_request_id() -> str | None:
    return _request_id_var.get()


def log_elapsed(label: str, elapsed_s: float) -> None:
    if settings.debug_timing_logs:
        rid = current_request_id() or "--------"
        print(f"[TIMING][{rid}] {label}: {elapsed_s * 1000:.0f} ms")


@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        log_elapsed(label, time.perf_counter() - t0)
