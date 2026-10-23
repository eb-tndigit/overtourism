import numpy as np
import pandas as pd
import pytest

from overtourism.overtourism.backend_extension.api.models.trentino_indicators import (
    AvailableAccommodationBedsIndicator,
    AvailableAccommodationFacilitiesIndicator,
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
                "nome_mercato_1": "Italia",
                "nome_mercato_2": "Germania",
                "nome_mercato_3": "Francia",
            }
        ]
    ).to_csv(source, sep=";", index=False)
    indicator = MainMarketDependenceIndicator(source)

    result = indicator.get_indicator("2023-01-01", "2023-12-31")

    assert result is not None
    assert result.loc[0, "INDICE"] == pytest.approx(0.6)
    assert result.loc[0, "main_markets"] == "Italy, Germany, France"
