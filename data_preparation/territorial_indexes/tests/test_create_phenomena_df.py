import pandas as pd
import pytest

from utils.steps.create_phenomena_df import (
    compute_consumption,
    compute_presenze_trentino,
)


def test_compute_presenze_rejects_duplicate_source_keys():
    duplicate_alb = pd.DataFrame(
        {
            "DATA": ["2024-07-01", "2024-07-01"],
            "ID_COMUNE": ["022001", "022001"],
            "presenze_alb": [10, 10],
        }
    )
    extralb = pd.DataFrame(
        {
            "DATA": ["2024-07-01"],
            "ID_COMUNE": ["022001"],
            "presenze_xalb": [4],
        }
    )
    vodafone = pd.DataFrame(
        {
            "DATA": ["2024-07-01"],
            "ID_COMUNE": ["022001"],
            "presenze": [20],
            "presenze_escursionisti_vodafone": [8],
        }
    )

    with pytest.raises(ValueError, match="presenze alberghiere.*duplicate keys"):
        compute_presenze_trentino(duplicate_alb, extralb, vodafone)


def test_compute_consumption_rejects_duplicate_source_keys():
    duplicate_ee = pd.DataFrame(
        {
            "DATA": ["2024-07-01", "2024-07-01"],
            "ID_COMUNE": ["022001", "022001"],
            "ee_services": [10.0, 10.0],
        }
    )
    gas_services = pd.DataFrame(
        columns=["DATA", "ID_COMUNE", "gas_services"]
    )
    ee_total = pd.DataFrame(
        columns=["DATA", "ID_COMUNE", "ee_total"]
    )
    gas_total = pd.DataFrame(
        columns=["DATA", "ID_COMUNE", "gas_total"]
    )

    with pytest.raises(ValueError, match="consumi elettrici servizi.*duplicate keys"):
        compute_consumption(duplicate_ee, gas_services, ee_total, gas_total)
