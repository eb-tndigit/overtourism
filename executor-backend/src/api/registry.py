# SPDX-License-Identifier: Apache-2.0
"""Model-key -> Backend wiring for the REST API.

`FazzonBackend` and `MolvenoBackend` share an identical public contract
(`parameter_schema()`, `evaluate()`) by convention even though there is no
shared base class (see `overtourism/BACKEND_DESIGN.md` §Layer 3) — this
registry is what lets `src.api.routes` expose one generic
route set parameterized by `model_key` instead of duplicating routes per
model.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path

from src.model.fazzon.fazzon_backend import FazzonBackend
from src.model.fazzon.fazzon_data import download_fazzon_data, ensure_fazzon_data
from src.model.molveno.molveno_backend import MolvenoBackend
from src.model.molveno.molveno_data import download_molveno_data, ensure_molveno_data

BACKEND_REGISTRY: dict[str, type[FazzonBackend | MolvenoBackend]] = {
    "fazzon": FazzonBackend,
    "molveno": MolvenoBackend,
}

MODEL_TITLES: dict[str, str] = {
    "fazzon": "Altipiano di Fazzon",
    "molveno": "Molveno",
}

# Models whose data files come from the platform: startup check and on-demand download
# (``overwrite=True``).  Both return the data folder and raise on failure.
DATA_STARTUP: dict[str, Callable[[], Path]] = {
    "fazzon": ensure_fazzon_data,
    "molveno": ensure_molveno_data,
}
DATA_DOWNLOADERS: dict[str, Callable[..., Path]] = {
    "fazzon": download_fazzon_data,
    "molveno": download_molveno_data,
}


@functools.cache
def get_backend(model_key: str) -> FazzonBackend | MolvenoBackend:
    """Return the (process-wide, lazily-built) backend instance for `model_key`.

    Mirrors the `st.cache_resource`-memoized backend construction the
    Streamlit dashboards already use — each backend is expensive enough to
    build once per process, not per-request.

    Raises
    ------
    KeyError
        If `model_key` is not in `BACKEND_REGISTRY`.
    """
    backend_cls = BACKEND_REGISTRY[model_key]
    return backend_cls()
