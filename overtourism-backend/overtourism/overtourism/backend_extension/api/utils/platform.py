# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import digitalhub as dh  # type: ignore[import-untyped]
from digitalhub.utils.exceptions import StoreError  # type: ignore[import-untyped]

# Platform specific settings
project_name = os.getenv("PROJECT_NAME", "overtourism")


def download_index_data(overwrite: bool = False) -> None:
    """Download the index data from the digitalhub platform."""
    dataitems = [
        "phen_consumption.parquet",
        "phen_ee_tourism_share.parquet",
        "phen_flussi_od.parquet",
        "phen_meteo.parquet",
        "phen_popolazione.parquet",
        "phen_presenze.parquet",
        "phen_provenienza.parquet",
        "phen_strutture.parquet",
        "phen_tempo_medio_turisti.parquet",
    ]
    artifacts = [
        "map_comuni_into_apt.json",
        "mapping_comuni_ISTAT.json",
        "Com01012026_g",
    ]

    project = dh.get_project(project_name)
    download_dir = Path(__file__).parents[3] / "database" / "index_data"

    if download_dir.exists() and not overwrite:
        return  # Skip download if the directory already exists

    download_dir.mkdir(parents=True, exist_ok=True)

    for dataitem in dataitems:
        print(f"Downloading dataitem: {dataitem}")
        try:
            project.get_dataitem(dataitem).download(
                str(download_dir), overwrite=overwrite
            )
        except StoreError as e:
            print(f"Failed to download dataitem {dataitem}: {e}")

    for artifact in artifacts:
        print(f"Downloading artifact: {artifact}")
        try:
            project.get_artifact(artifact).download(
                str(download_dir), overwrite=overwrite
            )
        except StoreError as e:
            print(f"Failed to download artifact {artifact}: {e}")
