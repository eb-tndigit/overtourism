# SPDX-License-Identifier: Apache-2.0
"""Access to the platform (DigitalHub) and to the S3 data lake.

All heavy / optional dependencies (digitalhub, boto3, geopandas) are imported lazily, so that the
rest of the package (and the test-suite) works without them.
"""

import configparser
import io
import json
import logging
import tempfile
from pathlib import Path

from utils.config import (
    CREDENTIALS_FILE,
    PROJECT,
    REFERENCES,
    S3_BUCKET,
    S3_ENV,
    S3_PREFIX,
)

logger = logging.getLogger(__name__)

_s3 = None
_bucket = None


# ---------------------------------------------------------------- platform
def get_dataframe(name: str):
    import digitalhub as dh

    return dh.get_dataitem(name, project=PROJECT).as_df()


def log_dataframe(file_path, name: str, type: str = "parquet"):
    """Uploads an already-saved dataframe file to the platform."""
    import digitalhub as dh

    if type not in {"parquet", "csv"}:
        raise NotImplementedError(f"Unsupported type: {type}.")
    project = dh.get_or_create_project(PROJECT)
    return project.log_table(f"{name}.{type}", source=str(file_path), file_format=type)


# ---------------------------------------------------------------- S3
def _s3_credentials(env: str) -> dict:
    """Credentials from the file `platform.credentials_file` of settings.yaml, in the home folder (DHCLI)."""
    config = configparser.ConfigParser()
    config.read(Path.home() / CREDENTIALS_FILE)
    section = config[env]
    return {
        "endpoint_url": section["aws_endpoint_url"],
        "aws_access_key_id": section["aws_access_key_id"],
        "aws_secret_access_key": section["aws_secret_access_key"],
        "aws_session_token": section["aws_session_token"],
    }


def init_s3_dhcli(env=S3_ENV):
    """Returns (s3 resource, bucket)."""
    import boto3

    s3 = boto3.resource("s3", **_s3_credentials(env))
    return s3, s3.Bucket(S3_BUCKET)


def init_s3(force=False):
    global _s3, _bucket
    if _s3 is None or _bucket is None or force:
        _s3, _bucket = init_s3_dhcli()
    return _s3, _bucket


def get_s3(name: str) -> io.BytesIO:
    """Downloads <S3_PREFIX><name> into an in-memory buffer (pointer at 0)."""
    s3, _ = init_s3()
    obj = s3.Object(S3_BUCKET, S3_PREFIX + name)
    buffer = io.BytesIO()
    obj.download_fileobj(buffer)
    buffer.seek(0)
    return buffer


def get_json_s3(name: str) -> dict:
    """Downloads and parses a JSON by data-lake key (see `references` in settings.yaml)."""
    return json.load(get_s3(name))


def get_mapping(reference: str) -> dict:
    """JSON mapping by reference key of settings.yaml, e.g. get_mapping("mapping_comuni")."""
    return get_json_s3(REFERENCES[reference])


def read_shapefile_s3(base_path: str):
    """Downloads shapefile files (.shp, .shx, .dbf, .prj) and returns a GeoDataFrame."""
    import geopandas as gpd

    _, bucket = init_s3()

    if base_path.endswith(".shp"):
        base_path = base_path[:-4]

    s3_prefix = S3_PREFIX + base_path

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        downloaded_shp = None

        for obj in bucket.objects.filter(Prefix=s3_prefix):
            file_name = Path(obj.key).name
            local_file_path = temp_path / file_name
            bucket.download_file(obj.key, str(local_file_path))
            if file_name.endswith(".shp"):
                downloaded_shp = local_file_path

        if downloaded_shp is None:
            raise FileNotFoundError(f"No .shp file found for prefix: {s3_prefix}")

        return gpd.read_file(downloaded_shp)
