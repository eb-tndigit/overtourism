"""
Build the indicator registry from the YAML files of ``api/indicators/`` (see ``api/indicators/README.md``).

An indicator is one YAML file, ``definitions/<key>.yaml`` (the file name is the key used by the API). It is
computed in one of three ways, exactly one of which must be present:

``formula``      an arithmetic expression over the phenomena (``beds / population``), see ``formula.py``;
``combinator``   the name of a Python function registered in ``combinators.py`` (+ ``params``);
``class``        a Python ``Indicator`` subclass registered with :func:`indicator_class`
                 (+ ``sources`` / ``params`` for its constructor), for the indicators that need more than a function.

Everything is validated when the files are loaded (:func:`load_registry`): a mistake in a definition stops the
start-up with a message that names the file and the problem, instead of failing on the first request. Building the
``Indicator`` objects is lazy and does not read any data (the phenomena load their files on first use).
"""

from __future__ import annotations

import copy
import importlib
import inspect
import re
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from overtourism.overtourism.backend_extension.api.models import trenitno_phenomena
from overtourism.overtourism.backend_extension.api.models.combinators import (
    COMBINATORS,
)
from overtourism.overtourism.backend_extension.api.models.formula import (
    Formula,
    FormulaError,
)
from overtourism.overtourism.backend_extension.api.models.indicator import Indicator
from overtourism.overtourism.backend_extension.api.models.phenomenon import Phenomenon


class IndicatorDefinitionError(ValueError):
    """One or more YAML definitions are not valid. ``problems`` lists them all, one string per problem."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__(
            f"{len(self.problems)} problem(s) in the indicator definitions:\n"
            + "\n".join(f"  - {p}" for p in self.problems)
        )


# ---------------------------------------------------------------------------
# Python classes usable from YAML (``class: <Name>``)
# ---------------------------------------------------------------------------

_INDICATOR_CLASSES: dict[str, type[Indicator]] = {}


def indicator_class(cls: type[Indicator]) -> type[Indicator]:
    """Class decorator: make an ``Indicator`` subclass usable from YAML as ``class: <ClassName>``."""
    _INDICATOR_CLASSES[cls.__name__] = cls
    return cls


# ---------------------------------------------------------------------------
# YAML reading
# ---------------------------------------------------------------------------


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses a key repeated in the same mapping (PyYAML would silently keep the last)."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable):
                continue  # the standard constructor reports it (with line and column)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None,
                    None,
                    f"duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _read_yaml(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.load(
                fh, Loader=_UniqueKeyLoader
            )  # noqa: S506 (safe loader subclass)
    except yaml.YAMLError as exc:
        raise IndicatorDefinitionError([f"{path.name}: invalid YAML: {exc}"]) from exc


def _read_string_mapping(path: Path, what: str) -> dict[str, str]:
    data = _read_yaml(path)
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        raise IndicatorDefinitionError(
            [f"{path.name}: must be a mapping of {what} (name: text)"]
        )
    return data


def load_sources(path: Path, data_root: Path) -> dict[str, Path]:
    """``sources.yaml``: data file name -> path (relative to ``data_root``)."""
    return {
        name: data_root / relative
        for name, relative in _read_string_mapping(path, "data sources").items()
    }


def load_snippets(path: Path | None) -> dict[str, str]:
    """``snippets.yaml``: reusable pieces of description text, used in a description as ``{{name}}``."""
    if path is None or not Path(path).exists():
        return {}
    return _read_string_mapping(Path(path), "text snippets")


# ---------------------------------------------------------------------------
# Specification (what a YAML file says, validated)
# ---------------------------------------------------------------------------

# YAML key -> (Indicator attribute, expected type, human-readable type)
_METADATA: dict[str, tuple[str, type | tuple, str]] = {
    "name": ("name", str, "text"),
    "category": ("category", str, "text"),
    "order": ("order", int, "integer"),
    "description": ("description", str, "text"),
    "index_value_unit_description": ("index_value_unit_description", str, "text"),
    "year_only": ("year_only", bool, "true/false"),
    "available_for_variation": ("availableForVariation", bool, "true/false"),
    "internal_only": ("internal_only", bool, "true/false"),
    "extra_fields": ("extraFields", list, "list of names"),
    "custom_legend": ("custom_legend", dict, "mapping"),
    "column_aggs": ("column_aggs", dict, "mapping column: aggregation"),
}
# required for an indicator built from a formula / a combinator (a class may provide its own)
_REQUIRED_METADATA = (
    "name",
    "category",
    "order",
    "description",
    "index_value_unit_description",
)
_STRUCTURE_KEYS = {
    "phenomena",
    "formula",
    "combinator",
    "class",
    "params",
    "sources",
}
_RESERVED_COLUMNS = {"ID_COMUNE", "DATA", "INDICE"}
_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SNIPPET_PATTERN = re.compile(r"\{\{\s*([A-Za-z0-9_-]+)\s*\}\}")
_RUNTIME_ARGUMENTS = {"df", "filtered_data", "start_date", "end_date"}


@dataclass(frozen=True)
class PhenomenonSpec:
    name: str
    cls: type[Phenomenon]
    source: Path
    hidden: bool
    kwargs: dict[str, Any]  # every other key of the YAML entry (col, measure, ...)


@dataclass(frozen=True)
class IndicatorSpec:
    key: str
    file: str
    metadata: dict[str, Any]  # Indicator attribute -> value (only what the YAML sets)
    phenomena: tuple[PhenomenonSpec, ...] = ()
    formula: Formula | None = None
    combinator: Callable | None = None
    combinator_name: str = ""
    indicator_cls: type[Indicator] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    class_sources: dict[str, Path] = field(default_factory=dict)

    @property
    def order(self):
        value = self.metadata.get("order")
        if value is None and self.indicator_cls is not None:
            value = getattr(self.indicator_cls, "order", None)
        return 9999 if value is None else value


def _import_object(path: str, what: str) -> Any:
    """Import ``package.module.Name`` or ``package.module:Name``."""
    module_name, _, attribute = path.replace(":", ".").rpartition(".")
    if not module_name:
        raise ValueError(f"{what} {path!r} is not a registered name or a dotted path")
    try:
        return getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"{what} {path!r} cannot be imported: {exc}") from exc


def _bind_check(callable_: Callable, label: str, *args, **kwargs) -> None:
    try:
        inspect.signature(callable_).bind(*args, **kwargs)
    except TypeError as exc:
        raise ValueError(f"{label}: {exc}") from exc


def _parse_phenomenon(
    entry: Any, index: int, sources: dict[str, Path]
) -> PhenomenonSpec:
    where = f"phenomena[{index}]"
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: must be a mapping (name, type, source, ...)")
    entry = dict(entry)
    for required in ("name", "type", "source"):
        if required not in entry:
            raise ValueError(f"{where}: missing '{required}'")
    name, type_name, source_name = (
        entry.pop("name"),
        entry.pop("type"),
        entry.pop("source"),
    )
    hidden = entry.pop("hidden", False)
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(
            f"{where}: name {name!r} must be a plain identifier (letters, digits, '_'): "
            "it is the column name used by the formula"
        )
    where = f"phenomena[{name}]"
    if name in _RESERVED_COLUMNS:
        raise ValueError(f"{where}: {name!r} is a reserved column name")
    if not isinstance(hidden, bool):
        raise ValueError(f"{where}: 'hidden' must be true or false")
    if source_name not in sources:
        raise ValueError(
            f"{where}: unknown source {source_name!r}. Known sources (sources.yaml): {sorted(sources)}"
        )

    if not isinstance(type_name, str):
        raise ValueError(f"{where}: 'type' must be a phenomenon class name")
    if "." in type_name:
        cls = _import_object(type_name, f"{where}: type")
    else:
        cls = getattr(trenitno_phenomena, type_name, None)
    if not (inspect.isclass(cls) and issubclass(cls, Phenomenon)):
        known = sorted(
            n
            for n, c in vars(trenitno_phenomena).items()
            if inspect.isclass(c) and issubclass(c, Phenomenon) and c is not Phenomenon
        )
        raise ValueError(
            f"{where}: unknown phenomenon type {type_name!r}. Available: {known}"
        )

    spec = PhenomenonSpec(
        name=name,
        cls=cls,
        source=sources[source_name],
        hidden=hidden,
        kwargs=entry,
    )
    _phenomenon_call(spec, where)  # signature check, nothing is built
    return spec


def _phenomenon_call(spec: PhenomenonSpec, where: str) -> tuple[dict, bool]:
    """
    The keyword arguments to build the phenomenon and whether its name has to be set afterwards.

    ``name`` is passed to the constructor when it takes one (otherwise it is set as an attribute: every Phenomenon
    only reads ``self.name`` when it resolves). ``col`` defaults to ``name`` for the types that require one.
    """
    parameters = inspect.signature(spec.cls).parameters
    kwargs = dict(spec.kwargs)
    accepts_name = "name" in parameters
    if accepts_name:
        kwargs["name"] = spec.name
    col = parameters.get("col")
    if (
        col is not None
        and col.default is inspect.Parameter.empty
        and "col" not in kwargs
    ):
        kwargs["col"] = spec.name
    _bind_check(spec.cls, f"{where} ({spec.cls.__name__})", spec.source, **kwargs)
    return kwargs, not accepts_name


def _check_combinator_params(
    fn: Callable, params: dict, extra_fields: list[str], label: str
) -> None:
    """Every ``params`` key must be a named argument; every required one must be given (or be a run-time value)."""
    parameters = inspect.signature(fn).parameters
    if not any(p.kind is p.VAR_KEYWORD for p in parameters.values()):
        raise ValueError(
            f"{label}: the function must accept **_extra (the Indicator also passes "
            "filtered_data, start_date, end_date and the request's extra fields)"
        )
    named = {
        n
        for n, p in parameters.items()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    unknown = sorted(set(params) - named)
    if unknown:
        raise ValueError(
            f"{label}: unknown param(s) {unknown}. The function takes: {sorted(named - {'df'})}"
        )
    clash = sorted(set(params) & (set(extra_fields) | _RUNTIME_ARGUMENTS))
    if clash:
        raise ValueError(
            f"{label}: {clash} are run-time values and cannot also be params"
        )
    first = next(iter(parameters), None)
    missing = [
        n
        for n, p in parameters.items()
        if n != first
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        and p.default is inspect.Parameter.empty
        and n not in params
        and n not in extra_fields
        and n not in _RUNTIME_ARGUMENTS
    ]
    if missing:
        raise ValueError(f"{label}: missing param(s) {missing}")


def _expand_snippets(text: str, snippets: dict[str, str]) -> str:
    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in snippets:
            raise ValueError(
                f"description uses the unknown snippet {{{{{name}}}}}. Known snippets: {sorted(snippets)}"
            )
        return snippets[name]

    return _SNIPPET_PATTERN.sub(replace, text)


def parse_definition(
    path: Path, sources: dict[str, Path], snippets: dict[str, str]
) -> IndicatorSpec:
    """Read and validate one definition file. Raises ``ValueError`` with the first problem found."""
    key = path.stem
    if not _KEY_PATTERN.match(key):
        raise ValueError(
            f"the file name is the indicator key and must match {_KEY_PATTERN.pattern}"
        )
    data = _read_yaml(path)
    if not isinstance(data, dict):
        raise ValueError(
            "the file must contain a mapping (name: ..., formula: ..., ...)"
        )

    unknown = sorted(set(data) - set(_METADATA) - _STRUCTURE_KEYS)
    if unknown:
        raise ValueError(
            f"unknown key(s) {unknown}. Allowed: {sorted(set(_METADATA) | _STRUCTURE_KEYS)}"
        )

    # -- how it is computed: exactly one of formula / combinator / class ------
    modes = [m for m in ("formula", "combinator", "class") if m in data]
    if len(modes) != 1:
        raise ValueError(
            "exactly one of 'formula', 'combinator' or 'class' is required"
            + (f", found {modes}" if modes else "")
        )
    mode = modes[0]

    # -- metadata -------------------------------------------------------------
    metadata: dict[str, Any] = {}
    for yaml_key, (attribute, expected, label) in _METADATA.items():
        if yaml_key not in data:
            continue
        value = data[yaml_key]
        if (
            not isinstance(value, expected)
            or (expected is int and isinstance(value, bool))
            or (expected is str and not value.strip())
        ):
            hint = (
                " (a text starting with '{' is read by YAML as a mapping: "
                "write it as a block starting with '>-' or put it in quotes)"
                if isinstance(value, dict) and expected is str
                else ""
            )
            raise ValueError(f"'{yaml_key}' must be {label}, got {value!r}{hint}")
        metadata[attribute] = value
    if "extraFields" in metadata and not all(
        isinstance(f, str) for f in metadata["extraFields"]
    ):
        raise ValueError("'extra_fields' must be a list of names")
    if "column_aggs" in metadata and not all(
        isinstance(k, str) and isinstance(v, str)
        for k, v in metadata["column_aggs"].items()
    ):
        raise ValueError("'column_aggs' must map a column name to an aggregation name")
    if "description" in metadata:
        metadata["description"] = _expand_snippets(metadata["description"], snippets)

    params = data.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("'params' must be a mapping")
    if mode == "formula" and params:
        raise ValueError("'params' cannot be used with a formula (use a combinator)")
    if mode == "formula" and metadata.get("extraFields"):
        raise ValueError(
            "a formula cannot use 'extra_fields' (use a combinator, which receives them)"
        )

    # -- computed with a class ------------------------------------------------
    if mode == "class":
        if "phenomena" in data:
            raise ValueError(
                "'phenomena' cannot be used with 'class': the class builds its own phenomena"
            )
        name = data["class"]
        if not isinstance(name, str):
            raise ValueError("'class' must be a class name")
        cls = _INDICATOR_CLASSES.get(name) or (
            _import_object(name, "class") if "." in name else None
        )
        if not (inspect.isclass(cls) and issubclass(cls, Indicator)):
            raise ValueError(
                f"unknown indicator class {name!r}. Registered with @indicator_class: {sorted(_INDICATOR_CLASSES)}"
            )
        raw_sources = data.get("sources", {})
        if not isinstance(raw_sources, dict):
            raise ValueError(
                "'sources' must map a constructor argument to a source name"
            )
        class_sources = {}
        for argument, source_name in raw_sources.items():
            if source_name not in sources:
                raise ValueError(
                    f"sources.{argument}: unknown source {source_name!r}. Known: {sorted(sources)}"
                )
            class_sources[argument] = sources[source_name]
        overlap = sorted(set(class_sources) & set(params))
        if overlap:
            raise ValueError(f"{overlap} are given both in 'sources' and in 'params'")
        _bind_check(cls, f"class {name}", **class_sources, **params)
        return IndicatorSpec(
            key=key,
            file=path.name,
            metadata=metadata,
            indicator_cls=cls,
            params=dict(params),
            class_sources=class_sources,
        )

    # -- computed with a formula or a combinator ------------------------------
    if "sources" in data:
        raise ValueError(
            "'sources' is only for 'class' (a phenomenon names its own source)"
        )
    missing = [k for k in _REQUIRED_METADATA if k not in data]
    if missing:
        raise ValueError(f"missing required key(s) {missing}")
    raw_phenomena = data.get("phenomena")
    if not isinstance(raw_phenomena, list) or not raw_phenomena:
        raise ValueError("'phenomena' must be a non-empty list")

    phenomena = tuple(
        _parse_phenomenon(entry, i, sources) for i, entry in enumerate(raw_phenomena)
    )
    names = [p.name for p in phenomena]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f"phenomenon name(s) used more than once: {repeated}")

    if mode == "formula":
        try:
            formula = Formula(data["formula"])
        except FormulaError as exc:
            raise ValueError(f"formula: {exc}") from exc
        unknown_names = sorted(formula.columns - set(names))
        if unknown_names:
            raise ValueError(
                f"formula {formula.text!r} uses {unknown_names}, which are not phenomena of this "
                f"indicator. Its phenomena: {names}"
            )
        return IndicatorSpec(
            key=key,
            file=path.name,
            metadata=metadata,
            phenomena=phenomena,
            formula=formula,
        )

    name = data["combinator"]
    fn = COMBINATORS.get(name) if isinstance(name, str) else None
    if fn is None and isinstance(name, str) and "." in name:
        fn = _import_object(name, "combinator")
    if not callable(fn):
        raise ValueError(
            f"unknown combinator {name!r}. Registered in combinators.py: {sorted(COMBINATORS)}"
        )
    _check_combinator_params(
        fn, params, metadata.get("extraFields", []), f"combinator {name}"
    )
    return IndicatorSpec(
        key=key,
        file=path.name,
        metadata=metadata,
        phenomena=phenomena,
        combinator=fn,
        combinator_name=name,
        params=dict(params),
    )


# ---------------------------------------------------------------------------
# Building the Indicator objects
# ---------------------------------------------------------------------------


class DeclarativeIndicator(Indicator):
    """An indicator whose phenomena and INDICE are declared in YAML (``formula`` or ``combinator``)."""

    def __init__(
        self,
        spec: IndicatorSpec,
        phenomena: list[Phenomenon],
        combinator: Callable,
    ):
        super().__init__(phenomena=phenomena, combinator=combinator)
        self.key = spec.key
        self._hidden_columns = [p.name for p in spec.phenomena if p.hidden]

    def _compute_indicator(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        **extra,
    ) -> pd.DataFrame | None:
        result = super()._compute_indicator(start_date, end_date, **extra)
        if result is None or not self._hidden_columns:
            return result
        # helper phenomena (e.g. a numerator) are used by the INDICE but are not returned next to it
        return result.drop(columns=self._hidden_columns)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} key={self.key!r} phenomena={self.phenomenon_names()}>"


def _build_phenomenon(spec: PhenomenonSpec, where: str) -> Phenomenon:
    kwargs, set_name = _phenomenon_call(spec, where)
    phenomenon = spec.cls(spec.source, **kwargs)
    if set_name:
        phenomenon.name = spec.name
    return phenomenon


def _apply_metadata(indicator: Indicator, spec: IndicatorSpec) -> None:
    """Set the attributes the YAML declares on the instance (they take precedence over the class defaults)."""
    for attribute, value in spec.metadata.items():
        if isinstance(value, (list, dict)):
            value = copy.deepcopy(
                value
            )  # never shared between indicators (or with the spec)
        setattr(indicator, attribute, value)


def build_indicator(spec: IndicatorSpec) -> Indicator:
    """Instantiate the indicator of a validated spec. Reads no data."""
    if spec.indicator_cls is not None:
        indicator = spec.indicator_cls(**spec.class_sources, **spec.params)
        _apply_metadata(indicator, spec)
        return indicator

    phenomena = [
        _build_phenomenon(p, f"{spec.file}: phenomena[{p.name}]")
        for p in spec.phenomena
    ]
    for built, declared in zip(phenomena, spec.phenomena):
        if built.name != declared.name:  # a phenomenon type that renames itself
            raise ValueError(
                f"{spec.file}: phenomenon {declared.name!r} ({declared.cls.__name__}) "
                f"ended up named {built.name!r}"
            )

    if spec.formula is not None:
        by_name = {p.name: p for p in phenomena}
        text_columns = sorted(
            n for n in spec.formula.columns if not getattr(by_name[n], "numeric", True)
        )
        if text_columns:
            raise ValueError(
                f"{spec.file}: formula {spec.formula.text!r} uses text column(s) {text_columns}"
            )
        combinator = spec.formula.as_combinator()
    else:
        bound = partial(spec.combinator, **spec.params)

        def combinator(df, **extra):
            return bound(df, **extra)

        combinator.__name__ = f"{spec.combinator_name}({spec.params})"

    indicator = DeclarativeIndicator(spec, phenomena, combinator)
    _apply_metadata(indicator, spec)
    return indicator


def load_registry(
    definitions_dir: Path,
    *,
    sources: dict[str, Path],
    snippets: dict[str, str] | None = None,
) -> dict[str, Callable[[], Indicator]]:
    """
    Read and validate every ``*.yaml`` of ``definitions_dir`` and return ``{key: factory}``, where calling the
    factory builds the ``Indicator`` (the registry of ``trentino_indicators.get_indicator``).

    The entries are sorted by ``order`` and then by key, so the list the API returns does not depend on the file
    system. All the problems of all the files are reported together in one ``IndicatorDefinitionError``.
    """
    definitions_dir = Path(definitions_dir)
    snippets = snippets or {}
    files = sorted(definitions_dir.glob("*.yaml"))
    if not files:
        raise IndicatorDefinitionError([f"no *.yaml definitions in {definitions_dir}"])

    specs: list[IndicatorSpec] = []
    problems: list[str] = []
    for path in files:
        try:
            specs.append(parse_definition(path, sources, snippets))
        except IndicatorDefinitionError as exc:
            problems.extend(exc.problems)
        except ValueError as exc:
            problems.append(f"{path.name}: {exc}")
    if problems:
        raise IndicatorDefinitionError(problems)

    specs.sort(key=lambda s: (s.order, s.key))
    return {spec.key: (lambda spec=spec: build_indicator(spec)) for spec in specs}
