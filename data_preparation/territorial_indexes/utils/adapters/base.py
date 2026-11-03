# SPDX-License-Identifier: Apache-2.0
"""Adapter registry.

An *adapter* turns a raw yearly source (whatever layout the provider used that year) into the
"aligned" layout expected by the standardization step of its *kind*. When a new year arrives with
a different layout, write a new adapter function, decorate it with @register_adapter and reference
it by name in the update config: nothing else in the pipeline changes.

Adapter signature:  func(raw: RawData, **params) -> pandas.DataFrame
"""

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

# kind -> columns the aligned dataframe must contain (input contract of the standardization step)
KIND_SCHEMAS = {
    "popolazione": ["comune", "popolazione", "anno"],
    "strutture": [
        "comune",
        "anno",
        "alberghieri strutture",
        "alberghieri posti_letto",
        "extra alb. Strutture",
        "extra alb. Posti_letto",
        "tot convenzionali strutture",
        "tot convenzionali posti_letto",
        "all. privati numero",
        "all. privati posti_letto",
        "all.disposizione numero",
        "all. disposizione posti_letto",
    ],
    "vodafone": ["locId", "date", "value", "userProfile", "locType"],
    "meteo_raw": ["date", "comune_id", "t_min", "t_max"],  # weather, used as downloaded (not normalized)
    "presenze_raw": ["mese"],  # ISPAT monthly arrivals / presences, used as downloaded (not normalized)
    # Vodafone OD co-visits (flows between areas)
    "vodafone_covisits": [
        "date",
        "locType",
        "sourceId",
        "sourceAmbito",
        "targetId",
        "targetAmbito",
        "covisitType",
        "userCountry",
        "userProfile",
        "value",
    ],
    # energy category files (standard layout of read_energy_category)
    "energia": [
        "DATA",
        "LOCATION",
        "codice_utilizzo",
        "gennaio",
        "febbraio",
        "marzo",
        "aprile",
        "maggio",
        "giugno",
        "luglio",
        "agosto",
        "settembre",
        "ottobre",
        "novembre",
        "dicembre",
    ],
}

# kind -> datasets it can feed
KIND_DATASETS = {
    "popolazione": {"popolazione"},
    "strutture": {"strutture"},
    "vodafone": {"vodafone", "vodafone_provenienza", "vodafone_arrivi", "vodafone_tempi"},
    "presenze_raw": {"presenze_alb", "presenze_extralb"},
    "meteo_raw": {"meteo"},
    "vodafone_covisits": {"vodafone_covisits"},
    "energia": {"ee_prelievi", "gas_prelievi", "ee_totale", "gas_totale"},
}


class SchemaMismatchError(ValueError):
    """The aligned dataframe does not respect the schema of its kind."""


@dataclass
class RawData:
    """Output of a reader: the dataframe plus optional side information (e.g. APT names)."""

    df: pd.DataFrame
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Adapter:
    name: str
    kind: str
    func: Callable
    description: str = ""
    # True: stands in for a source whose update format is not known yet. It is refused when the
    # update config is validated, i.e. before anything is downloaded.
    placeholder: bool = False


_REGISTRY: dict[str, Adapter] = {}


def register_adapter(name: str, *, kind: str, placeholder: bool = False):
    if kind not in KIND_SCHEMAS:
        raise ValueError(f"Unknown kind {kind!r}. Valid: {sorted(KIND_SCHEMAS)}")

    def decorator(func):
        if name in _REGISTRY:
            raise ValueError(f"Adapter {name!r} already registered")
        doc = (inspect.getdoc(func) or "").splitlines()
        _REGISTRY[name] = Adapter(
            name, kind, func, doc[0] if doc else "", placeholder=placeholder
        )
        return func

    return decorator


def get_adapter(name: str) -> Adapter:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown adapter {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_adapters(kind: str | None = None) -> list[str]:
    return sorted(a.name for a in _REGISTRY.values() if kind is None or a.kind == kind)


def check_adapter_params(adapter: Adapter, params: dict):
    """Raises TypeError if `params` do not fit the adapter signature (missing / unexpected)."""
    inspect.signature(adapter.func).bind(None, **params)


def validate_schema(df: pd.DataFrame, kind: str, adapter_name: str, strict: bool = True) -> bool:
    """Checks that the aligned df contains all the columns required by its kind."""
    missing = sorted(set(KIND_SCHEMAS[kind]) - set(df.columns))
    if not missing:
        return True
    msg = (
        f"Adapter '{adapter_name}' (kind '{kind}') produced a dataframe with missing "
        f"columns: {missing}. Check that the adapter matches the layout of the source file."
    )
    if strict:
        raise SchemaMismatchError(msg)
    logger.warning(msg)
    return False
