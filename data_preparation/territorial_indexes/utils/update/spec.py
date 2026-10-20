# SPDX-License-Identifier: Apache-2.0
"""Declarative description of a yearly update (which files, how to read them, which adapter).

Which datasets are updated is decided here: list only the ones you want, or set `enabled: false` on a
dataset to skip it without deleting its sources.

Example (YAML):

    datasets:
      popolazione:
        enabled: true                # optional, default true
        sources:
          - file: popolazione_2027_ISPAT.csv
            reader: csv
            adapter: popolazione_ispat_1jan
            adapter_kwargs: {year: 2026}
      strutture:
        sources:
          - file: strutture_2026.xlsx
            reader: excel
            reader_kwargs: {header: [0, 1]}
            adapter: strutture_annuario # default: 2025 layout; use a dedicated adapter for other layouts
                adapter_kwargs: {year: 2026}
            label: "2026"            # optional: keeps this source separate until merge
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from utils.adapters import (
    KIND_DATASETS,
    check_adapter_params,
    get_adapter,
)
from utils.config import DATA_START, STD_SUFFIX, UPDATE_PART_SUFFIX, UPDATE_WINDOW
from utils.datasets import ALL_DATASETS
from utils.window import Window
from utils.update.sources import READERS

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SourceSpec:
    file: str
    reader: str
    adapter: str
    reader_kwargs: dict = field(default_factory=dict)
    adapter_kwargs: dict = field(default_factory=dict)
    # If set, this source remains separate until its processed frame is combined into the merge.
    label: str | None = None

    def part_name(self, dataset: str) -> str:
        """Internal name for a source part during standardization and processing."""
        if self.label:
            return f"{dataset}_{self.label}{UPDATE_PART_SUFFIX}"
        return f"{dataset}{UPDATE_PART_SUFFIX}"


@dataclass(frozen=True)
class UpdateConfig:
    sources: dict  # dataset name -> list[SourceSpec]
    # period of the update (optional `period: {start, end}` in the yaml); None = `period.update_*` of settings.yaml
    period: tuple | None = None

    @property
    def window(self) -> Window:
        """Time window of the update: only the data inside it are processed and appended (never before
        the global `period.start`)."""
        if self.period is None:
            return UPDATE_WINDOW
        start, end = (pd.Timestamp(str(x)) for x in self.period)
        return Window(max(start, DATA_START), end)

    @property
    def datasets(self) -> list:
        return [d for d in ALL_DATASETS if d in self.sources]


def _load_mapping(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        return yaml.safe_load(text)
    return json.loads(text)


def parse_config(raw: dict) -> UpdateConfig:
    if not isinstance(raw, dict) or "datasets" not in raw:
        raise ConfigError("Config must be a mapping with a top-level 'datasets' key")
    sources = {}
    for dataset, body in raw["datasets"].items():
        if dataset not in ALL_DATASETS:
            raise ConfigError(
                f"Unknown dataset {dataset!r}. Available datasets: {list(ALL_DATASETS)}"
            )
        body = body or {}
        unknown_body = set(body) - {"sources", "enabled"}
        if unknown_body:
            raise ConfigError(
                f"Dataset {dataset!r} has unknown keys {sorted(unknown_body)}"
            )
        if not body.get("enabled", True):
            logger.info("Dataset %s is disabled in the update config: skipped", dataset)
            continue
        items = body.get("sources") or []
        if not items:
            raise ConfigError(f"Dataset {dataset!r} has no sources")
        sources[dataset] = []
        for i, item in enumerate(items):
            missing = {"file", "reader", "adapter"} - set(item)
            if missing:
                raise ConfigError(
                    f"{dataset}.sources[{i}] is missing {sorted(missing)}"
                )
            unknown = set(item) - {
                "file",
                "reader",
                "adapter",
                "reader_kwargs",
                "adapter_kwargs",
                "label",
            }
            if unknown:
                raise ConfigError(
                    f"{dataset}.sources[{i}] has unknown keys {sorted(unknown)}"
                )
            label = item.get("label")
            if label is not None:
                label = str(label)
                if not re.fullmatch(r"[A-Za-z0-9_]+", label):
                    raise ConfigError(
                        f"{dataset}.sources[{i}]: label {label!r} must be letters/digits/_"
                    )
            sources[dataset].append(
                SourceSpec(
                    file=item["file"],
                    reader=item["reader"],
                    adapter=item["adapter"],
                    reader_kwargs=dict(item.get("reader_kwargs") or {}),
                    adapter_kwargs=dict(item.get("adapter_kwargs") or {}),
                    label=label,
                )
            )
    if not sources:
        raise ConfigError("No dataset is enabled in the update config")
    if "references" in raw:
        raise ConfigError(
            "Shared references belong in config/settings.yaml, not the update config"
        )
    period = None
    if raw.get("period") is not None:
        p = raw["period"]
        if not isinstance(p, dict) or set(p) != {"start", "end"}:
            raise ConfigError("`period` must be a mapping with `start` and `end`")
        try:
            period = (pd.Timestamp(str(p["start"])), pd.Timestamp(str(p["end"])))
        except ValueError as e:
            raise ConfigError(f"`period`: {e}") from None
        if period[0] > period[1]:
            raise ConfigError("`period`: start is after end")
    return UpdateConfig(sources=sources, period=period)


def validate_config(config: UpdateConfig) -> UpdateConfig:
    """Offline validation: readers/adapters exist, adapter kind fits the dataset, kwargs fit the signature."""
    for dataset, sources in config.sources.items():
        for src in sources:
            where = f"{dataset}:{src.file}"
            if src.reader not in READERS:
                raise ConfigError(
                    f"{where}: unknown reader {src.reader!r}. Available: {sorted(READERS)}"
                )
            try:
                adapter = get_adapter(src.adapter)
            except KeyError as e:
                raise ConfigError(f"{where}: {e.args[0]}") from None
            if adapter.placeholder:
                raise ConfigError(
                    f"{where}: adapter {src.adapter!r} is a placeholder, the format of "
                    f"the update data of {dataset!r} is not defined yet"
                )
            if dataset not in KIND_DATASETS[adapter.kind]:
                raise ConfigError(
                    f"{where}: adapter {src.adapter!r} (kind {adapter.kind!r}) cannot feed {dataset!r}"
                )
            try:
                check_adapter_params(adapter, src.adapter_kwargs)
            except TypeError as e:
                raise ConfigError(
                    f"{where}: bad adapter_kwargs for {src.adapter!r}: {e}"
                ) from None
    return config


def load_config(config) -> UpdateConfig:
    """`config`: path to a .yaml/.yml/.json file, a dict, or an UpdateConfig."""
    if isinstance(config, UpdateConfig):
        return validate_config(config)
    if isinstance(config, (str, Path)):
        config = _load_mapping(Path(config))
    return validate_config(parse_config(config))


# ------------------------------------------------------------------ parts
@dataclass(frozen=True)
class UpdatePart:
    """Set of sources that end up in the same files: <name>_std (normalized), then stacked per dataset
    into <dataset>_update_pr (processed)."""

    name: str  # base name: <name>_std
    dataset: str
    kind: str

    @property
    def std_name(self) -> str:
        return f"{self.name}{STD_SUFFIX}"


def plan_parts(config: UpdateConfig) -> list:
    """The parts of the update, in config order. Sources with the same part name are stacked."""
    parts = {}
    for dataset in config.datasets:
        for src in config.sources[dataset]:
            kind = get_adapter(src.adapter).kind
            part = UpdatePart(src.part_name(dataset), dataset, kind)
            if parts.setdefault(part.name, part) != part:
                raise ValueError(
                    f"Sources of '{part.name}' have different kind/dataset: "
                    "give them a different `label`"
                )
    return list(parts.values())


def config_from_argv(description: str, argv=None) -> Path:
    """`--config` argument of the update steps run as scripts."""
    import argparse

    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--config", required=True, type=Path, help="update config (.yaml/.yml/.json)"
    )
    return p.parse_args(argv).config
