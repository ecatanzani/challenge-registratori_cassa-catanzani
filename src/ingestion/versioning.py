"""
Confronto e identificazione delle versioni dei manuali
"""

import re


def parse_version(version: str | None) -> tuple[int, ...]:
    """Versione come tupla di interi, per confronti d'ordine corretti.
    """
    if not version:
        return (0,)
    parts = []
    for segment in str(version).replace("-", ".").split("."):
        digits = "".join(ch for ch in segment if ch.isdigit())
        if digits:
            parts.append(int(digits))
    return tuple(parts) or (0,)


def version_slug(version: str) -> str:
    """Versione in forma utilizzabile dentro un identificatore ('0.3' -> '0-3')."""
    return re.sub(r"[^A-Za-z0-9]+", "-", str(version)).strip("-")


def make_chunk_id(doc_id: str, version: str, counter: int) -> str:
    """Identificatore stabile di un chunk, che include la versione del manuale: `printf_f_v03_c0012`.
    """
    return f"{doc_id}_{version_slug(version)}_c{counter:04d}"
