import numpy as np
import pandas as pd
import pytest
from overtourism.overtourism.backend_extension.api.models.trentino_indicators import (
    _REGISTRY,
    AnthropogenicLoadIndicator,
    AvailableAccommodationBedsIndicator,
    AvailableAccommodationFacilitiesIndicator,
    HiddenTourismIndicator,
    MainMarketDependenceIndicator,
    TouristAccommodationBedsIndicator,
    TouristAccommodationFacilitiesIndicator,
    get_indicator,
)


@pytest.mark.parametrize(
    ("indicator_class", "components", "numerator", "expected"),
    [
        (
            TouristAccommodationFacilitiesIndicator,
            ["strutture_alloggi_turistici", "strutture_alb", "strutture_exalb"],
            "strutture_alloggi_turistici",
            0.2,
        ),
        (
            AvailableAccommodationFacilitiesIndicator,
            [
                "strutture_alloggi_turistici",
                "strutture_alloggi_disp",
                "strutture_alb",
                "strutture_exalb",
            ],
            "strutture_alloggi_disp",
            0.2,
        ),
        (
            TouristAccommodationBedsIndicator,
            ["letti_alloggi_turistici", "letti_alb", "letti_exalb"],
            "letti_alloggi_turistici",
            0.2,
        ),
        (
            AvailableAccommodationBedsIndicator,
            [
                "letti_alloggi_turistici",
                "letti_alloggi_disp",
                "letti_alb",
                "letti_exalb",
            ],
            "letti_alloggi_disp",
            0.2,
        ),
    ],
)
def test_hospitality_indicator_computes_requested_share(
    indicator_class, components, numerator, expected
):
    indicator = indicator_class("unused.parquet")
    values = {
        component: [2.0 if component == numerator else 8.0 / (len(components) - 1)]
        for component in components
    }
    result = indicator.combinator(pd.DataFrame(values))

    assert result.iloc[0] == pytest.approx(expected)
    assert indicator.phenomenon_names() == components


def test_hospitality_indicator_returns_missing_for_zero_total():
    indicator = TouristAccommodationFacilitiesIndicator("unused.parquet")
    values = {column: [0.0] for column in indicator.component_columns}

    result = indicator.combinator(pd.DataFrame(values))

    assert np.isnan(result.iloc[0])


@pytest.mark.parametrize(
    "registry_key",
    [
        "incidenza-alloggi-turistici-strutture",
        "incidenza-alloggi-disposizione-strutture",
        "incidenza-alloggi-turistici-letti",
        "incidenza-alloggi-disposizione-letti",
    ],
)
def test_hospitality_indicator_is_registered(registry_key):
    assert get_indicator(registry_key).name.startswith("Incidenza Alloggi")


def test_main_market_indicator_includes_top_market_names(tmp_path):
    source = tmp_path / "phen_provenienza.csv"
    pd.DataFrame(
        [
            {
                "DATA": "2023",
                "ID_COMUNE": "001",
                "volume_mercato_1": 150,
                "volume_totale": 250,
                "nome_mercato_1": "Italy",
                "nome_mercato_2": "Germany",
                "nome_mercato_3": "France",
            }
        ]
    ).to_csv(source, sep=";", index=False)
    indicator = MainMarketDependenceIndicator(source)

    result = indicator.get_indicator("2023-01-01", "2023-12-31")

    assert result is not None
    assert result.loc[0, "INDICE"] == pytest.approx(0.6)
    assert result.loc[0, "main_markets"] == "Italy, Germany, France"


def test_period_percentage_impact_indicator_is_removed():
    assert "incidenza-periodo" not in _REGISTRY
    assert "gini-stagionalita" in _REGISTRY

    with pytest.raises(KeyError, match="Unknown indicator 'incidenza-periodo'"):
        get_indicator("incidenza-periodo")


def test_hidden_tourism_falls_back_when_official_total_is_zero_or_missing():
    indicator = HiddenTourismIndicator("unused.parquet")
    values = pd.DataFrame(
        {
            "presenze_vodafone": [10.0, 10.0, 10.0, 10.0],
            "presenze_alb": [5.0, 0.0, 5.0, float("nan")],
            "presenze_xalb": [5.0, 0.0, float("nan"), 5.0],
        }
    )

    result = indicator.combinator(values)

    assert result.tolist() == [1.0, 10.0, 10.0, 10.0]


def test_anthropogenic_load_requires_both_presence_profiles():
    indicator = AnthropogenicLoadIndicator("unused.parquet", "unused.parquet")
    values = pd.DataFrame(
        {
            "presenze_vodafone": [2.0, float("nan"), 2.0],
            "presenze_escursionisti_vodafone": [3.0, 3.0, float("nan")],
            "population": [10.0, 10.0, 10.0],
        }
    )

    result = indicator.combinator(values)

    assert result.iloc[0] == 0.5
    assert result.iloc[1:].isna().all()


def test_custom_legend_is_only_configured_for_seasonality_gini():
    assert get_indicator("gini-stagionalita").custom_legend == {
        "type": "cluster-value",
        "cluster_property": "cluster stagionale",
        "clusters": [
            {"id": 0, "label": "annuale", "color": "#4575b4"},
            {"id": 1, "label": "estate_alpina", "color": "#d73027"},
            {"id": 2, "label": "estate_spalle", "color": "#1a9850"},
            {"id": 3, "label": "sci", "color": "#984ea3"},
        ],
    }
    assert get_indicator("densita-turistica").custom_legend is None
