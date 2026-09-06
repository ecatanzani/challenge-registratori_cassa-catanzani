"""
Confronto e identificazione delle versioni dei manuali
"""
from __future__ import annotations

import re


def parse_version(version: str | None) -> tuple[int, ...]:
    """Versione come tupla di interi, per confronti d'ordine corretti.

    I segmenti non numerici (es. il suffisso in "3.0-rc1") vengono ignorati:
    l'ordine resta cosi' totale e deterministico, che e' quello che serve a un
    tie-break. Una versione assente o senza cifre vale (0,), cioe' "la piu'
    vecchia possibile".
    """
    if not version:
        return (0,)
    parts = []
    for segment in str(version).replace("-", ".").split("."):
        digits = "".join(ch for ch in segment if ch.isdigit())
        if digits:
            parts.append(int(digits))
    return tuple(parts) or (0,)


def is_older(version: str | None, than: str | None) -> bool:
    return parse_version(version) < parse_version(than)


def latest_of(versions) -> str | None:
    """La piu' recente fra le versioni date, o None se la sequenza e' vuota."""
    versions = [v for v in versions if v is not None]
    if not versions:
        return None
    return max(versions, key=parse_version)


def version_slug(version: str) -> str:
    """Versione in forma utilizzabile dentro un identificatore ('0.3' -> '0-3')."""
    return re.sub(r"[^A-Za-z0-9]+", "-", str(version)).strip("-")


def make_chunk_id(doc_id: str, version: str, counter: int) -> str:
    """Identificatore stabile di un chunk, che include la versione del manuale: `printf_f_v0-3_c0012`.
    """
    return f"{doc_id}_v{version_slug(version)}_c{counter:04d}"
