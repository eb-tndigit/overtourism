# How to define indicators

Every indicator of the API is **one YAML file** in `definitions/`. The file name is the indicator key used by the API
(`definitions/ricettivita.yaml` → `/…/ricettivita`). Nothing has to be registered anywhere else: add the file, restart,
and the indicator is in the list.

```
api/
├── indicators/
│   ├── README.md              ← this file
│   ├── sources.yaml           ← the data files, by name
│   ├── snippets.yaml          ← pieces of description text shared by several indicators
│   └── definitions/*.yaml     ← ONE FILE PER INDICATOR
└── models/
    ├── formula.py             ← the engine of `formula:`
    ├── combinators.py         ← Python functions usable from YAML (`combinator:`)
    ├── trentino_indicators.py ← Python classes usable from YAML (`class:`)
    └── indicator_loader.py    ← reads and validates the YAML
```

## Choose how the index is computed

An indicator is computed in exactly one of three ways. Use the first one that is enough.

| | You write | Use it when |
|---|---|---|
| **1. `formula`** | an expression in the YAML | the index is arithmetic on the phenomena: a ratio, a share, a sum, a density |
| **2. `combinator`** | a Python *function* + YAML for the rest | the maths needs more (conditions, ranking, rounding options…) but the indicator is otherwise ordinary |
| **3. `class`** | a Python *class* + YAML for the metadata | the indicator must add its own columns, return extra data (e.g. map arrows), or combine other indicators |

Of the 25 current indicators, 18 are level 1, 2 are level 2 (`pagerank-discretizzato`,
`distanza-media-turisti`), 5 are level 3.

---

## 1. An indicator with a formula (no Python)

Example: beds per km². Save as `definitions/densita-ricettiva.yaml`:

```yaml
name: Densità ricettiva                       # shown to the user
category: Pressione turistica                 # groups the indicators in the UI
order: 18                                     # position in the list (see "Order" below)
index_value_unit_description: Posti letto per km²
description: >-
  Il <strong>numero di posti letto</strong> per chilometro quadrato di superficie del comune.

phenomena:                                    # the input data: one entry per column used
  - {name: beds,       type: BedsPhenomenon,    source: structures, col: tot_postiletto}
  - {name: superficie, type: SurfacePhenomenon, source: surface,    col: Shape_Area}

formula: beds / superficie                    # INDICE, written with the `name`s above
```

That is the whole indicator.

### Fields

| Key | Required | Meaning |
|---|---|---|
| `name` | yes | Title of the indicator |
| `category` | yes | Group in the UI |
| `order` | yes | Integer; the list is sorted by it (lowest first) |
| `description` | yes | Text for the user. HTML is allowed (`<strong>`…). Can contain `{{snippet}}` (see below) |
| `index_value_unit_description` | yes | Unit / reading of the value |
| `phenomena` | yes | The input columns (below) |
| `formula` / `combinator` / `class` | exactly one | How INDICE is computed |
| `year_only` | no, `false` | `true` if the data is yearly: only whole years can be requested |
| `available_for_variation` | no, `true` | `false` to hide the indicator from the time series |
| `internal_only` | no, `false` | `true` for infrastructure indicators that are not listed to the user |
| `extra_fields` | no | Request parameters passed to the calculation (combinators and classes only) |
| `custom_legend`, `column_aggs` | no | Legend definition / how the extra columns are aggregated on macro areas |

`custom_legend` supports `type: cluster-value` for cluster-based colors and
`type: diverging` for a zero-centered red/blue scale (negative red, positive blue).

Unknown keys, repeated keys and wrong types stop the start-up with a message that names the file.

### Phenomena

A phenomenon is one column of one data file, brought to a common daily grain per comune. Each entry has:

| Key | Meaning |
|---|---|
| `name` | **Required.** The name of the column in the calculation: what the formula uses, and the column name in the API response. A plain identifier (`beds`, `presenze_vodafone`). Unique inside the indicator |
| `type` | **Required.** The phenomenon class (list below) |
| `source` | **Required.** A data file from `sources.yaml` |
| `col` | The column of the file. Required by some types, a default for others (list below) |
| `hidden` | `true`: used by the calculation but not returned next to INDICE (a helper, e.g. a numerator) |
| others | The type's own options (`measure`, `max_minutes`, `min_total`, `text`, …) |

A column named in `phenomena` is returned by the API next to `INDICE`. To give it a readable label in the UI, add `name: "label"`
to `PHENOMENON_LABELS_IT` in `models/trenitno_phenomena.py`.

Available types (the classes of `models/trenitno_phenomena.py`; a class you add there is usable immediately by its name):

| `type` | Grain of the file | Options (default) |
|---|---|---|
| `PresencesPhenomenon` | daily | `col` (`presenze`) |
| `BedsPhenomenon` | yearly | `col` (`tot_postiletto`) |
| `ExtraBedsPhenomenon` | yearly | `col` (`tot_postiletto_non_conv`) |
| `FacilitiesTotalPhenomenon` | yearly | `col` (`tot_strutture`) |
| `ExtraFacilitiesPhenomenon` | yearly | `col` (`tot_strutture_non_conv`) |
| `PopulationPhenomenon` | yearly | `col` (`popolazione`) |
| `SurfacePhenomenon` | static | `col` (`Shape_Area`), `province_code` (`22`) |
| `ConsumptionPhenomenon` | daily | `col` (**required**) |
| `WeatherPhenomenon` | daily | `col` (`hdd`) |
| `FlowIndicatorPhenomenon` | yearly | `col` (**required**) |
| `ProvenienzaPhenomenon` | yearly | `col` (`hhi_provenienza`) |
| `ProvenienzaVolumePhenomenon` | yearly | `col` (**required**), `min_total` (`0`) |
| `TravelTimeArrivalsPhenomenon` | daily | `measure` (`arrivals` / `arrival_minutes` / `arrivals_within`), `max_minutes` (only with `arrivals_within`) |
| `ElectricityTourismSharePhenomenon` | yearly | `col` (**required**), `text` (`false`; `true` for a text column) |
| `RedistributionPhenomenon` | daily | `col` (**required**) |

For the types whose `col` is required, leaving it out means `col = name`. A type from another module can be used with its
dotted path (`type: my.module.MyPhenomenon`). The parameters are checked when the files are loaded: a wrong option is an error.

### The formula language

A formula is an arithmetic expression over the phenomenon names. It is parsed, **never executed as Python**: only the
syntax below is accepted.

* numbers, phenomenon names, parentheses, `+ - * /` and `**`
* functions:

| Function | Result |
|---|---|
| `sum(a, b, …)` | sum of the values that are present; null only if **all** are missing |
| `mean(a, b, …)`, `min(a, b, …)`, `max(a, b, …)` | same, for the mean / minimum / maximum |
| `coalesce(a, b, …)` | the first value that is present |
| `clip(x, lower, upper)` | `x` limited to `[lower, upper]` |
| `abs(x)`, `sqrt(x)`, `log(x)` | `sqrt` of a negative number and `log` of a value ≤ 0 are null |

**Nulls.** A null means "no value" and is never filled in silently:

* `+ - * **` with a null operand give a null;
* **`/` is zero-safe**: dividing by 0 gives a null, never `inf`. You do not need to guard divisions;
* to give a missing input an explicit value, say so: `coalesce(ee_services, 0) / ee_total`.

Real examples from the current definitions:

```yaml
formula: sum(presenze_vodafone, presenze_escursionisti_vodafone) / superficie       # densita-turistica
formula: strutture_alloggi_turistici / sum(strutture_alloggi_turistici, strutture_alb, strutture_exalb)
formula: coalesce(presenze_vodafone / (presenze_alb + presenze_xalb), presenze_vodafone)   # fallback if no official presences
formula: quota_ingresso_combined                                                      # the index is a column as it is
```

Comparisons, conditions (`if`), loops, attribute access and unknown functions are rejected. If you need them, use level 2.

---

## 2. A formula is not enough: a Python combinator

Keep the YAML (metadata, phenomena) and replace `formula` with `combinator` + `params`:

```yaml
phenomena:
  - {name: pagerank_combined, type: FlowIndicatorPhenomenon, source: flows_od, col: pagerank_combined}
  - {name: pagerank_tourist,  type: FlowIndicatorPhenomenon, source: flows_od, col: pagerank_tourist}
  - {name: pagerank_visitor,  type: FlowIndicatorPhenomenon, source: flows_od, col: pagerank_visitor}

combinator: quantile_class
params: {column: pagerank_combined, n_classes: 5}
```

The function lives in `models/combinators.py` and is registered with a decorator:

```python
@combinator("quantile_class")
def quantile_class(df, *, column, n_classes=5, **_extra):
    """..."""
    ...
    return classes            # a pandas Series, one value per row of df
```

Rules (also in the docstring of `combinators.py`):

* `df` has one row per comune (or per comune and day) and one column per phenomenon, by `name`.
* Return a `pandas.Series` aligned with `df`. A null stays a null: the base class does not fill it.
* Every keyword-only argument without a default is a **param**: it must be in the YAML `params:`. A param that is
  misspelled or missing stops the start-up.
* The function must accept `**_extra`: the indicator also passes `filtered_data`, `start_date`, `end_date` and the
  request parameters listed in the YAML `extra_fields` (for instance `average_travel_time` receives
  `arrotonda_30_minuti`).
* It must not depend on anything but its inputs.

A function in any other module can be used with a dotted path: `combinator: my.module:my_function`.

`hidden: true` on a phenomenon is the usual companion: `distanza-media-turisti` needs `arrival_minutes` to compute the
index, but only `tourist_arrivals` is returned.

---

## 3. A class (the exception)

Use a class only when the indicator needs more than "a value per comune": extra columns computed by code
(`gini-stagionalita`: the seasonality cluster), data besides the values (`squilibrio-ridistribuzione`: the arrows on the
map), or a composition of other indicators (`sostenibilita-territoriale`, `tasso-variazione`).

1. In `models/trentino_indicators.py` write an `Indicator` subclass and register it:

```python
@indicator_class
class MyIndicator(Indicator):
    def __init__(self, source_file, threshold=10):
        super().__init__(
            phenomena=[PresencesPhenomenon(source_file, "presenze_vodafone")],
            combinator=self._combine,
        )
        ...
```

2. Point a YAML file to it. `sources` maps a constructor argument to a name of `sources.yaml` (the loader passes the
   path); `params` are literal arguments:

```yaml
name: ...
category: ...
order: ...
index_value_unit_description: ...
description: >-
  ...

class: MyIndicator
sources: {source_file: presences}
params: {threshold: 20}
```

For a class, `phenomena` and `formula` are not allowed (the class builds its own phenomena) and every metadata key
is optional: what the YAML omits is taken from the class. The usual split, as in the existing classes: the YAML holds
what a person may want to edit (name, category, order, text, unit, `year_only`, …); the class keeps what is tied to the
columns its code produces (`column_aggs`, `custom_legend`, `extraFields`).

---

## Shared files

**`sources.yaml`** names the data files, relative to `index_data_path`:

```yaml
presences: phen_presenze.parquet
```

A phenomenon uses it as `source: presences`. To read a new file, add a line here.

**`snippets.yaml`** holds text used by several descriptions. In a description write `{{name}}`:

```yaml
description: >-
  La quota misura ... {{flow_common}}
```

Unknown snippet names stop the start-up.

## Order

`order` sorts the list the user sees (lowest first). Values are unique on purpose, so the position of every indicator is
visible in the files: the current ones are in groups (10s: pressure, 30s: structures, 50s: mobility, 100s: flows and origin,
150s: consumption). If two indicators have the same `order`, they are sorted by key. To put a new indicator between two
others, renumber a neighbour: there is no hidden order.

## Checks and tests

All the definitions are read and validated **when the API starts**: a mistake in any file (unknown phenomenon type or
source, a name in the formula that is not a phenomenon, a bad formula, a missing key, a repeated key, a misspelled
param…) stops it with one message that lists every problem and the file it is in, instead of failing on the first
request. Building the indicators does not read any data.

Run the tests after adding an indicator:

```
pytest api/tests
```

`test_indicator_definitions.py` loads every definition and builds every indicator. To check the numbers of a new formula
indicator, add it to `REFERENCE` in that file: a small function that computes the expected value, tried on rows full of
zeros and missing values.

## Checklist for a new indicator

1. Is the data in a file that is in `sources.yaml`? If not, add it.
2. Is there a phenomenon type for it? If not, add a class to `models/trenitno_phenomena.py` (and the labels of its
   columns to `PHENOMENON_LABELS_IT`).
3. Create `definitions/<key>.yaml`: a formula if possible, otherwise a combinator, otherwise a class.
4. Pick an `order`.
5. Start the API (or `pytest api/tests`): any mistake is reported with the file name.

## Pitfalls

* **Quote text that starts with `{` or contains `: `.** Write long descriptions as a block (`description: >-` and the
  text indented below it), which needs no quoting. A bare `description: {{x}}` is read by YAML as a mapping.
* **`: ` and ` #` inside a one-line value** change its meaning in YAML (a new key, a comment): quote the value
  (`index_value_unit_description: 'Classe di PageRank (quantili: 1 = più basso)'`).
* **The `name` of a phenomenon is the API column name.** Renaming it changes the response.
* **Numbers that appear in two places.** A description may say "5 ore" while `max_minutes: 300` sets it: there is a comment
  next to the two in the existing files (`pagerank-discretizzato`, `dipendenza-domanda-prossimita`); keep them equal.
* **Deployment.** `indicators/**/*.yaml` must be shipped with the package (package data / Docker `COPY`), and `PyYAML` must
  be a dependency.
