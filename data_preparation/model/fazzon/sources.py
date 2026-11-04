# SPDX-License-Identifier: Apache-2.0
"""
STEP 0 - Raw sources of the Fazzon pipeline.

Input : data lake (settings.yaml platform.s3_prefix + fazzon.yaml keys), or fazzon.yaml local_source_dir
Output: paths of the raw files, as they arrive (no renaming of the content, no normalization)

With a local source dir the files are read in place (no copy: the road counts alone are ~2.3 GB).
From the data lake they are streamed to <raw>/fazzon/. See model/common/sources.py.
"""

from pathlib import Path

from model.common import sources
from model.fazzon.config import FAZZON_RAW_DIR


def resolve_sources(config: sources.SourcesConfig, refresh: bool = False, raw_dir=FAZZON_RAW_DIR) -> dict[str, Path | list[Path]]:
    """Local path of every source in fazzon.yaml, downloading it from the data lake if needed."""
    return sources.resolve_sources(config, raw_dir, refresh=refresh)
