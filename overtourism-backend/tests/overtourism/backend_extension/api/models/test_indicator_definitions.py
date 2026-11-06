"""
Tests of the YAML indicator definitions (``api/indicators``) and of the loader.

* the real definitions: all load, all can be built, the formula-based ones compute what the hand-written combinators
  they replaced computed (checked on frames full of zeros and missing values);
* the loader: a mistake in a definition is reported with the file and the problem.
"""

import numpy as np
import pandas as pd
import pytest
from overtourism.overtourism.backend_extension.api.models import (
    trentino_indicators as t,
)
from overtourism.overtourism.backend_extension.api.models.indicator_loader import (
    IndicatorDefinitionError,
    load_registry,
)

nan = np.nan


# ---------------------------------------------------------------------------
# The real definitions
# ---------------------------------------------------------------------------


def test_every_definition_builds_without_reading_data():
    assert len(t._REGISTRY) >= 25
    for key in t._REGISTRY:
        indicator = t.get_indicator(key)
        assert indicator.name and indicator.category, key
        assert isinstance(indicator.order, int), key


def test_order_values_are_unique_so_the_list_order_is_explicit():
    orders = [t.get_indicator(k).order for k in t._REGISTRY]
    assert len(orders) == len(set(orders))


def test_registry_is_sorted_by_order():
    orders = [t.get_indicator(k).order for k in t._REGISTRY]
    assert orders == sorted(orders)


def _edge_frame(columns, rows=300, seed=0):
    """Random values with plenty of zeros and missing values."""
    rng = np.random.default_rng(seed)
    pool = np.array([nan, 0.0, 0.0, 0.5, 1.0, 3.0, 10.0, 250.0])
    return pd.DataFrame({c: rng.choice(pool, rows) for c in columns})


def _sum(frame, columns):
    return frame[list(columns)].sum(axis=1, min_count=1)


def _share(frame, numerator, columns):
    return frame[numerator].div(frame[list(columns)].sum(axis=1).replace(0, nan))


# the hand-written combinators the formulas replaced: key -> function(df) -> Series
REFERENCE = {
    "ricettivita": lambda d: d["beds"].div(d["population"].replace(0, nan)),
    "turisticita": lambda d: d["presences"].div(d["population"].replace(0, nan)),
    "densita-turistica": lambda d: _sum(
        d, ["presenze_vodafone", "presenze_escursionisti_vodafone"]
    ).div(d["superficie"].replace(0, nan)),
    "densita-antropica": lambda d: (
        _sum(d, ["presenze_vodafone", "presenze_escursionisti_vodafone"])
        + d["population"]
    ).div(d["superficie"].replace(0, nan)),
    "incidenza-alloggi-turistici-strutture": lambda d: _share(
        d,
        "strutture_alloggi_turistici",
        ["strutture_alloggi_turistici", "strutture_alb", "strutture_exalb"],
    ),
    "incidenza-alloggi-disposizione-strutture": lambda d: _share(
        d,
        "strutture_alloggi_disp",
        [
            "strutture_alloggi_turistici",
            "strutture_alloggi_disp",
            "strutture_alb",
            "strutture_exalb",
        ],
    ),
    "incidenza-alloggi-turistici-letti": lambda d: _share(
        d,
        "letti_alloggi_turistici",
        ["letti_alloggi_turistici", "letti_alb", "letti_exalb"],
    ),
    "incidenza-alloggi-disposizione-letti": lambda d: _share(
        d,
        "letti_alloggi_disp",
        ["letti_alloggi_turistici", "letti_alloggi_disp", "letti_alb", "letti_exalb"],
    ),
    "turismo-sommerso": lambda d: d["presenze_vodafone"]
    .div((d["presenze_alb"] + d["presenze_xalb"]).where(lambda s: s != 0))
    .fillna(d["presenze_vodafone"]),
    "carico-antropico": lambda d: (
        d["presenze_vodafone"] + d["presenze_escursionisti_vodafone"]
    ).div(d["population"].where(d["population"] != 0)),
    "quota-mobilita-ingresso": lambda d: d["quota_ingresso_combined"],
    "quota-mobilita-uscita": lambda d: d["quota_uscita_combined"],
    "quota-intra-ambito": lambda d: d["quota_intra_ambito_combined"],
    "concentrazione-mercati-provenienza": lambda d: d["hhi_provenienza"],
    "impatto-turismo-consumi-elettrici": lambda d: d["ee_services"]
    .fillna(0)
    .div(d["ee_total"].where(d["ee_total"] != 0)),
    "impatto-turismo-consumi-gas": lambda d: d["gas_services"]
    .fillna(0)
    .div(d["gas_total"].where(d["gas_total"] != 0)),
    "dipendenza-domanda-prossimita": lambda d: d["arrivals_within"].div(
        d["tourist_arrivals"].replace(0, nan)
    ),
}


@pytest.mark.parametrize("key", sorted(REFERENCE))
def test_formula_indicators_match_the_hand_written_combinators(key):
    indicator = t.get_indicator(key)
    frame = _edge_frame(indicator.phenomenon_names())
    pd.testing.assert_series_equal(
        indicator.combinator(frame).astype(float),
        REFERENCE[key](frame).astype(float),
        check_names=False,
    )


def test_quota_consumi_elettrici_is_the_share_column():
    indicator = t.get_indicator("quota-consumi-elettrici-attribuibili-turismo")
    frame = pd.DataFrame(
        {
            "quota_ee_turismo": [0.1, nan],
            "presenze_medie": [1, 2],
            "consumo_medio": [3, 4],
            "affidabilita": ["alta", "bassa"],
        }
    )
    pd.testing.assert_series_equal(
        indicator.combinator(frame), frame["quota_ee_turismo"], check_names=False
    )


def test_pagerank_classes():
    indicator = t.get_indicator("pagerank-discretizzato")
    frame = pd.DataFrame({"pagerank_combined": [0.1, 0.2, 0.3, 0.4, 0.5, nan]})
    classes = indicator.combinator(
        frame, filtered_data={}, start_date=None, end_date=None
    )
    assert classes.tolist() == [1, 2, 3, 4, 5, 0]  # 5 classes, 0 = no value


def test_average_travel_time_in_hours_with_optional_rounding():
    indicator = t.get_indicator("distanza-media-turisti")
    frame = pd.DataFrame(
        {"tourist_arrivals": [10.0, 0.0, 4.0], "arrival_minutes": [2700.0, 0.0, 1000.0]}
    )
    assert indicator.combinator(frame).tolist()[0] == 4.5  # 270 min
    assert np.isnan(indicator.combinator(frame).tolist()[1])  # no arrivals
    rounded = indicator.combinator(frame, arrotonda_30_minuti=True)
    assert rounded.tolist()[2] == 4.0  # 250 min -> 240 min


def test_helper_phenomena_are_not_returned_next_to_the_index():
    indicator = t.get_indicator("distanza-media-turisti")
    assert indicator._hidden_columns == ["arrival_minutes"]


# ---------------------------------------------------------------------------
# The loader
# ---------------------------------------------------------------------------

SOURCES = {"structures": "structures.parquet", "population": "population.parquet"}

GOOD = """\
name: Test
category: Cat
order: 1
index_value_unit_description: unit
description: text
phenomena:
  - {name: beds, type: BedsPhenomenon, source: structures, col: tot_postiletto}
  - {name: population, type: PopulationPhenomenon, source: population, col: popolazione}
formula: beds / population
"""


def load(tmp_path, files, snippets=None):
    folder = tmp_path / "definitions"
    folder.mkdir(exist_ok=True)
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")
    return load_registry(
        folder, sources={k: tmp_path / v for k, v in SOURCES.items()}, snippets=snippets
    )


def problem(tmp_path, text, filename="x.yaml", snippets=None):
    with pytest.raises(IndicatorDefinitionError) as excinfo:
        load(tmp_path, {filename: text}, snippets)
    return "\n".join(excinfo.value.problems)


def test_a_good_definition_builds_an_indicator(tmp_path):
    registry = load(tmp_path, {"x.yaml": GOOD})
    indicator = registry["x"]()
    assert indicator.name == "Test"
    assert indicator.phenomenon_names() == ["beds", "population"]


def test_end_to_end_with_data(tmp_path):
    pd.DataFrame(
        {
            "ID_COMUNE": ["1", "2"],
            "DATA": ["2022", "2022"],
            "tot_postiletto": [10.0, 5.0],
        }
    ).to_parquet(tmp_path / "structures.parquet")
    pd.DataFrame(
        {"ID_COMUNE": ["1", "2"], "DATA": ["2022", "2022"], "popolazione": [5.0, 0.0]}
    ).to_parquet(tmp_path / "population.parquet")
    result = load(tmp_path, {"x.yaml": GOOD})["x"]().get_indicator(
        "2022-02-01", "2022-02-10"
    )
    values = result.set_index("ID_COMUNE")["INDICE"]
    assert values["1"] == 2.0
    assert np.isnan(values["2"])  # division by zero is a null


def test_the_description_can_use_snippets(tmp_path):
    registry = load(
        tmp_path,
        {
            "x.yaml": GOOD.replace(
                "description: text", "description: before {{ s }} after"
            )
        },
        {"s": "SNIP"},
    )
    assert registry["x"]().description == "before SNIP after"


def test_entries_are_sorted_by_order_then_key(tmp_path):
    registry = load(
        tmp_path,
        {
            "b.yaml": GOOD,
            "a.yaml": GOOD,
            "c.yaml": GOOD.replace("order: 1", "order: 0"),
        },
    )
    assert list(registry) == ["c", "a", "b"]


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            GOOD.replace("beds / population", "beds / pop"),
            "pop",
        ),  # name that is not a phenomenon
        (GOOD.replace("beds / population", "beds > population"), "unsupported syntax"),
        (
            GOOD.replace("type: BedsPhenomenon", "type: NopePhenomenon"),
            "unknown phenomenon type",
        ),
        (GOOD.replace("source: structures", "source: nowhere"), "unknown source"),
        (GOOD.replace("name: Test\n", ""), "missing required"),
        (GOOD + "descripton: typo\n", "unknown key"),
        (GOOD + "combinator: quantile_class\n", "exactly one of"),
        (GOOD.replace("formula: beds / population", ""), "exactly one of"),
        (GOOD + "params: {a: 1}\n", "cannot be used with a formula"),
        (GOOD.replace("order: 1", "order: first"), "'order' must be"),
        (GOOD.replace("name: population", "name: beds"), "more than once"),
        (GOOD.replace("name: population", "name: INDICE"), "reserved"),
        (
            GOOD.replace("col: popolazione", "colonna: popolazione"),
            "PopulationPhenomenon",
        ),
        (GOOD.replace("text\n", "'{{ nope }}'\n"), "unknown snippet"),
        (GOOD.replace("text\n", "{{ nope }}\n"), "invalid YAML"),
        (GOOD + "name: again\n", "duplicate key"),
        ("- not\n- a mapping\n", "must contain a mapping"),
        (
            GOOD.replace("formula: beds / population", "combinator: nope"),
            "unknown combinator",
        ),
        (
            GOOD.replace(
                "formula: beds / population",
                "combinator: quantile_class\nparams: {column: beds, n_clases: 3}",
            ),
            "unknown param",
        ),
        (
            GOOD.replace("formula: beds / population", "combinator: quantile_class"),
            "missing param",
        ),
        (
            GOOD.split("phenomena:")[0] + "class: NopeIndicator\n",
            "unknown indicator class",
        ),
        (
            GOOD.replace("formula: beds / population", "class: NopeIndicator"),
            "cannot be used with 'class'",
        ),
        (
            GOOD.replace(
                "formula: beds / population",
                "formula: beds / population\nextra_fields: [x]",
            ),
            "extra_fields",
        ),
    ],
)
def test_mistakes_are_reported_with_file_and_problem(tmp_path, text, expected):
    message = problem(tmp_path, text)
    assert "x.yaml" in message
    assert expected in message


def test_all_the_problems_of_all_the_files_are_reported_together(tmp_path):
    with pytest.raises(IndicatorDefinitionError) as excinfo:
        load(
            tmp_path,
            {
                "a.yaml": GOOD.replace("beds /", "nope /"),
                "b.yaml": GOOD.replace("order: 1", "order: x"),
            },
        )
    assert len(excinfo.value.problems) == 2


def test_a_text_phenomenon_cannot_be_used_in_a_formula(tmp_path):
    text = GOOD.replace(
        "  - {name: population, type: PopulationPhenomenon, source: population, col: popolazione}",
        "  - {name: label, type: ElectricityTourismSharePhenomenon, source: population, col: affidabilita, text: true}",
    ).replace("beds / population", "beds / label")
    registry = load(tmp_path, {"x.yaml": text})
    with pytest.raises(ValueError, match="text column"):
        registry["x"]()
