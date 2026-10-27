# Tourism & Mobility Indicators Framework

This package computes territorial indicators (indici) for tourism and
mobility phenomena. It is designed to work in different territories, with
each region or area plugging in its own data sources and indicator definitions.

This document explains the core logic (`Phenomenon` and `Indicator`), how a
region plugs into that engine, and how to add a new indicator, phenomenon,
or region.

---

## 1. Mental model

The idea of the framework is to be able to insert arbitrary source data
into that one canonical shape as cheaply as possible, once, and then
treating every query as a cheap slice of it.
Data can later parsed to answer different needs of visualization: a value
on a map, a time series in a chart. 

```
 raw source file            Phenomenon                    Indicator
 (CSV / Parquet,     ──►  daily × municipality panel   ──►   combine N phenomena
  any native grain)        (ID_municipality, DATA,             into one INDICE
                            <phenomenon name>)            per municipality / date
```


Two classes do all the work:

- **`Phenomenon`** (`phenomenon.py`) — wraps *one* raw data source (e.g.
  "beds per municipality per year", "mobile-network presences per municipality per
  day") and knows how to resolve it to the canonical daily × municipality grid.
  An `Indicator` can also be used as a "phenomenon" for another
  `Indicator` — see §3.3 below — to build macro-indicators out of several
  sub-indicators.
- **`Indicator`** (`indicator.py`) — owns a list of `Phenomenon`s plus a
  *combinator* function that turns their aligned values into a single
  `INDICE` (e.g. beds ÷ population). It also owns all the query-time logic:
  filtering by date, aggregating, computing time series with rolling
  variation, and caching.

Everything territorial (which comuni are visible, municipality vs. macro-area
or province grouping) is handled by `TerritorialConfig`, deliberately *outside* both
classes — an `Indicator` never knows about permissions or spatial
granularity, it only ever returns "one row per municipality."

---

## 2. `Phenomenon`

### 2.1 Two independent axes

A `Phenomenon` declares its **native resolution** on two axes, independently:

| Axis | Attribute | Values | Meaning |
|---|---|---|---|
| Temporal | `temporal_resolution` | `daily` / `monthly` / `yearly` | grain of the source data |
| Temporal | `temporal_strategy` | `identity` / `constant` / `weighted` | how to bring it to daily |
| Spatial | `spatial_resolution` | `municipality` / `macro-area` / `province` | grain of the source data |
| Spatial | `spatial_strategy` | `identity` / `constant` / `weighted` | how to bring it to municipality |

The **strategy** tells the class *how* to disaggregate:

- **`identity`** — already at target grain, nothing to do.
- **`constant`** — broadcast a coarser value uniformly (e.g. a yearly bed
  count is repeated for every day of that year; a provincial total is
  repeated for every municipality in the province). Cheap, vectorised, fine to do
  at request time.
- **`weighted`** — distributes a coarser value across the finer grain in
  proportion to a *reference* (`support_temporal` / `support_spatial`)
  phenomenon's own profile (e.g. spread a monthly tourist total across days
  in proportion to daily mobile-network presences). This is the expensive
  path — see the callout below.

> **Performance note on `weighted`:** this strategy loops per period/municipality
> in Python. It's provided for convenience on small datasets, but for
> anything non-trivial it should be run **offline** as a preprocessing
> script that writes a daily Parquet file, which is then loaded with
> `temporal_resolution="daily"` / `temporal_strategy="identity"` at
> runtime. Don't let `weighted` run inside a request path in production.

### 2.2 The resolved panel

Calling `phenom.resolve()` reads the source file, applies both axis strategies,
and stores the result in `self._panel`:

```
ID_municipality (str, zero-padded ISTAT code)  |  DATA (datetime64)  |  <name> (float)
```

one row per `(municipality, day)`. `Indicator` calls `resolve()` on every
phenomenon before doing anything else, so by the time a query runs,
`_filter_by_date()` is just a boolean mask over `self._panel` — no
disaggregation ever happens at query time.

### 2.3 Constructing a Phenomenon

```python
class BedsPhenomenon(Phenomenon):
    name = "beds"                       # column name in the resolved panel
    temporal_resolution = "yearly"      # source is one row per municipality per year
    temporal_strategy = "constant"      # broadcast that value to every day
    spatial_resolution = "municipality"       # source is already per-municipality
    spatial_strategy = "identity"

    def __init__(self, source, col="tot_postiletto"):
        super().__init__(source, col, agg="mean")
```

- `source`: path to a CSV or Parquet file. Files are cached process-wide by
  path (`_FILE_CACHE` in `phenomenon.py`) so multiple `Phenomenon`s reading
  the same file only hit disk once.
- `col`: the physical column name in the source file; it gets renamed to
  `self.name` internally so every resolved panel uses the phenomenon's
  logical name, not the raw file's column name.
- `agg`: how to collapse multiple rows into one value when needed —
  `"sum"`, `"mean"`, `"max"`, or `"min"` (e.g. `sum` for presences,
  `mean` for a yearly bed count that shouldn't be summed across days).

For `weighted` strategies, pass `support_temporal` / `support_spatial`
(another, already-resolvable `Phenomenon`) and optionally
`assign_temporal_fn` / `assign_spatial_fn` if proportional splitting isn't
the right rule for that phenomenon.

---

## 3. `Indicator`

### 3.1 Combinators

An `Indicator` is a list of resolved phenomena plus a **combinator**: a
pure function `fn(df, *, filtered_data, start_date, end_date, **extra) ->
pd.Series` that takes a DataFrame with one column per phenomenon (already
aligned by `ID_municipality`) and returns the `INDICE` values.

The common case — a straight ratio — has a built-in helper:

```python
class AccommodationCapacityIndicator(Indicator):
    name = "Indice di ricettività"

    def __init__(self, beds_source, population_source):
        super().__init__(
            phenomena=[BedsPhenomenon(beds_source), PopulationPhenomenon(population_source)],
            combinator=self.divide("beds", "population"),
        )
```

For anything more custom, write your own method and pass it as the
combinator.

### 3.2 Query surface

- **`get_indicator(start_date, end_date, **extra)`** → one row per municipality
  with an `INDICE` column, or `None` if any underlying phenomenon has no
  data in that window. Result is cached by `(start_date, end_date, extra)`.
- **`get_temporal_variation(start_date, end_date, granularity, region_id,
  **extra)`** → a list of `{label, data, std}` series, one per municipality plus
  one region-wide aggregate, bucketed by `granularity`
  (`giornaliero` / `settimanale` / `mensile` / `annuale`). Weekly buckets are
  labelled by their Monday date and include data through Sunday, clipped to
  the requested date range. Also cached.
- **`years_range`** → `{min_year, max_year}`, the full-calendar-year span
  for which *every* phenomenon in the indicator has data — i.e. the
  intersection of each phenomenon's `[min_date, max_date]`, restricted to
  complete years. Used to populate date pickers in the frontend so users
  can't pick a range with partial/missing data.

Both caches are unbounded, keyed by whatever arguments the frontend passes.
This is fine for a small number of distinct queries but is worth revisiting
(LRU/TTL) if usage patterns turn out to hit many distinct date ranges.

### 3.3 Composability: an Indicator as a phenomenon (macro-indicators)

`Indicator` implements the same duck-typed interface `Phenomenon` does —
`.name`, `.agg`, `.resolve()`, `._filter_by_date()`, `.aggregate()`,
`.date_bounds()` — so an `Indicator` can be passed straight into another
`Indicator`'s `phenomena=[...]` list, no adapter needed:

```python
ricettivita = AccommodationCapacityIndicator(BEDS_SOURCE, POPULATION_SOURCE)
ricettivita.name = "ricettivita"

densita = DensityIndicator(PRESENCES_SOURCE)
densita.name = "densita"

class MacroIndicator(Indicator):
    name = "Macro indicator name"

    def __init__(self):
        super().__init__(
            phenomena=[ricettivita, densita],
            combinator=self.divide("ricettivita", "densita"),  # or any custom fn
        )
```

`get_indicator()`, `get_temporal_variation()`, and `years_range` all work
unmodified on the macro-indicator — `_ensure_resolved()` etc. just call
`.resolve()` on each item in `phenomena`, and a sub-`Indicator`'s
`resolve()` builds its own `[ID_municipality, DATA, name]` daily panel by merging
its own phenomena and applying its own combinator, exactly like a real
`Phenomenon` would. Nesting is not limited to one level — an `Indicator`
built from other composed `Indicator`s works the same way.

**Important constraint on combinators used this way:** an `Indicator`'s
`resolve()` panel is built *once*, covering the full range its underlying
phenomena have data for, with `start_date=None`, `end_date=None`, and no
`**extra`. There's no single query these could come from — the panel has
to be valid for every future date-range slice a parent `Indicator` might
request. So a combinator that will be used on a sub-indicator must be a
pure function of the row's phenomenon values alone (like the built-in
`divide()`); it must not branch on `start_date`/`end_date`/`filtered_data`/
`**extra`, or the composed values will be silently wrong.

Each sub-`Indicator` used as a phenomenon also picks up a class attribute,
`agg` (default `"mean"`), controlling how its own daily `INDICE` values
collapse to one row per municipality when a *parent* `Indicator` queries it over
a date range — overridable per instance if summing rather than averaging
makes sense for a given composition.

---

## 4. Region plug-in convention

Everything above is region/territory-agnostic. A region is a **pair of modules**
following a fixed naming convention, currently:

```
{region}_indicators.py     e.g. trentino_indicators.py, liguria_indicators.py
utils_{region}.py          e.g. utils_trentino.py,      utils_liguria.py
```

`routes.py` loads them dynamically by name:

```python
region = "trentino"   # <-- currently hardcoded; see §5
indicators = import_module(f".{region}_indicators", package=__package__)
utils = import_module(f".utils_{region}", package=__package__)
```

### 4.1 What `{region}_indicators.py` must expose

| Name | Type | Purpose |
|---|---|---|
| `_REGISTRY` | `dict[str, Callable[[], Indicator]]` | maps a URL-facing indicator key (e.g. `"indice-turisticita"`) to a zero-arg factory that builds it |
| `get_indicator(key)` | function | looks up `_REGISTRY`, memoises the built `Indicator` in a module-level cache, raises `KeyError` on unknown keys |
| `CODICI_COMUNI_FILE` | `Path` | JSON/CSV mapping comuni → macro-areas |
| `MACRO_AREAS_FILE` | `Path` | source for `get_macro_areas` |
| `MAP_SHAPEFILE` | `Path` | shapefile for the region's municipality boundaries |

Concrete `Indicator` and `Phenomenon` subclasses for the region's own
tourism/mobility phenomena live in this same file (see
`trentino_indicators.py` for the current worked example: beds, population,
presences, facility counts, and the ratios/combinators built from them).

### 4.2 What `utils_{region}.py` must expose

| Name | Purpose |
|---|---|
| `get_list_comuni(file)` | all comuni as `[{code, name}]`, "Provincia" (`-1`) first |
| `get_macro_areas(file)` | list of `MacroArea(name, comuni)` for the region's province/APT groupings |
| `get_map_geometry(file)` | cached, reprojected (EPSG:4326) `GeoDataFrame` of municipality boundaries |
| `_build_macro_area_geodataframe(result, codici_comuni_file, geometry_file)` | joins computed values onto pre-unioned macro-area polygons |

Expensive one-time work (shapefile reprojection, macro-area polygon
unioning) is wrapped in `@lru_cache(maxsize=1)` here, since it only needs
to happen once per process regardless of how many requests come in.

### 4.3 Region-specific quirks stay in the region module

Anything genuinely region-specific belongs in `{region}_indicators.py` /
`utils_{region}.py`, not in the shared engine:

- `utils_trentino.py` currently hardcodes `gdf[gdf.COD_PROV == 22]` to
  select Trentino's province code from a national shapefile — a Liguria
  equivalent would filter on its own `COD_PROV`.
- Source file layouts, column names, and the specific `Indicator`/
  `Phenomenon` subclasses (e.g. Liguria's `DensityIndicator`,
  `RatioPresencesBedsIndicator`) are free to differ entirely between
  regions — they don't share a class hierarchy beyond `Indicator` and
  `Phenomenon` themselves.

---

## 5. Usage

The framework is composed by a Python backend and a FastAPI frontend.

We recommend using [uv](https://astral.sh/uv) for managing the Python
version, the virtual environment, and the dependencies. Please, refer
to `uv` documentation regarding how to install it for your operating system.

Once `uv` is installed, use these commands:

```bash
cd overtourism-liguria
uv venv
source .venv/bin/activate
uv sync --dev
```

To run a sample use

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Access the front-end by opening your browser and navigating to:

```
http://localhost:8000
```

## Dependencies

See [pyproject.toml](pyproject.toml) for the full list of dependencies.
Dependencies are anyway automatically installed by `uv sync --dev`.

---

## 6. Glossary

| Term | Meaning |
|---|---|
| **municipality** | Italian municipality — the finest spatial grain the engine works at |
| **macro_area** | a named group of comuni (such as an APT / tourism district) |
| **province** | all comuni aggregated into the province-wide unit |
| **INDICE** | the final computed value of an indicator, one row per municipality (or per municipality × date for variation queries) |
| **panel** | the resolved `Phenomenon` DataFrame: one row per `(ID_municipality, DATA)` |
| **combinator** | the pure function that turns aligned phenomenon columns into `INDICE` |

---

## 7. License

```
SPDX-License-Identifier: Apache-2.0
```

## 8. Notes/TODO

Disaggreatations will be performed during data preparation since they are computational heavy.

INDICI DI CAPACITÀ --> manca tasso di variazione arrivi turistici, ma è ricavabile da variazione turisticità

TODO:
- indicatore ricettività, giuste presenze vodafone? Non c'è una fonte migliore
- TURISMO SOMMERSO --> TODO
- FLUSSI --> TODO
- LIVELLO DI AFFOLLAMENTO TURISTICO --> TODO
- RIDISTRIBUZIONE DEI TURISTI --> TODO
- indicare confidenza con disaggregazioni