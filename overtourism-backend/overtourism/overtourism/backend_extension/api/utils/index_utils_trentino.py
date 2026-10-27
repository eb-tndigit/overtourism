from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import geopandas as gpd

# ---------------------------------------------------------------------------
# Comuni list
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_list_comuni(mapping_comuni_file: Path) -> list[dict]:
    """
    Return the list of all comuni. Cached for the process lifetime —
    the JSON mapping never changes while the server is running.
    """
    comuni = [{"code": "-1", "name": "Provincia"}]
    with open(mapping_comuni_file, mode="r", encoding="utf-8") as fh:
        mapping = json.load(fh)
    for name, id_comune in mapping.items():
        comuni.append({"code": str(id_comune).zfill(6), "name": name})
    return comuni


# ---------------------------------------------------------------------------
# Comuni - areas mapping
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_macro_areas(codici_comuni_file: Path) -> list[any]:
    """
    Read a JSON file mapping macro-area names to lists of comune IDs.

    Returns a list of MacroArea objects.

    Result is cached for the process lifetime.
    """
    from ..models.territorial_config import MacroArea

    with open(codici_comuni_file, "r", encoding="utf-8") as fh:
        data: dict[str, list[str]] = json.load(fh)

    for name, comuni in data.items():
        if not isinstance(comuni, list):
            raise TypeError(
                f"Expected list for {name!r}, got {type(comuni)}: {comuni!r}"
            )

    return [
        MacroArea(
            name=name,
            comuni=[str(c).zfill(6) for c in comuni],
        )
        for name, comuni in data.items()
        if comuni
    ]


# ---------------------------------------------------------------------------
# Comuni - seasonality cluster mapping
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_comuni_clusters(clusters_file: Path) -> dict[str, dict]:
    """
    Read a JSON file mapping comune IDs to their seasonality cluster:
    {"022001": {"cluster": 2, "tipo_cluster": "estate_spalle"}, ...}
    (cluster = integer code, tipo_cluster = name of the cluster).

    Result is cached for the process lifetime.
    """
    with open(clusters_file, "r", encoding="utf-8") as fh:
        data: dict[str, dict] = json.load(fh)

    return {
        str(comune).zfill(6): {
            "cluster": int(value["cluster"]),
            "tipo_cluster": str(value["tipo_cluster"]),
        }
        for comune, value in data.items()
    }


# ---------------------------------------------------------------------------
# Map geometry
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_map_geometry_cached(map_shapefile: Path) -> gpd.GeoDataFrame:
    """
    Read and reproject the shapefile once.  Reprojection is expensive CPU
    work; doing it on every request was the dominant cost for map endpoints.
    """
    gdf = gpd.read_file(map_shapefile).to_crs(epsg=4326)
    return gdf[gdf.COD_PROV == 22]


@lru_cache(maxsize=1)
def _load_vodafone_map_geometry_cached(geojson_file: Path) -> gpd.GeoDataFrame:
    """
    Read the geometry file (AREA_ID scheme, e.g. 'ITA.04.022.100.127'),
    used by flow-based indicators instead of the shapefile.
    """
    gdf = gpd.read_file(geojson_file).to_crs(epsg=4326)
    return gdf[gdf["AREA_ID"].str.startswith("ITA.04.022")]


def get_map_geometry(map_shapefile: Path):
    """
    Return a copy of the cached geometry so callers that mutate the frame
    (e.g. _empty_map_response) don't corrupt the shared cache object.
    """
    return _load_map_geometry_cached(map_shapefile).copy()


def get_vodafone_map_geometry(geojson_file: Path):
    """Return a copy of the cached Vodafone geometry (mutation-safe)."""
    return _load_vodafone_map_geometry_cached(geojson_file).copy()


# ---------------------------------------------------------------------------
# Macro-area geometry cache
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _build_macro_area_geometries_cached(
    codici_comuni_file: Path, geometry_file: Path
) -> gpd.GeoDataFrame:
    """
    Pre-compute the unioned polygon for each of the four province-level
    macro-areas.  Expensive (shapely union_all), so done once and cached.

    Returns a GeoDataFrame with columns [area_name, geometry].
    """

    gdf_base = _load_map_geometry_cached(geometry_file)
    rows = []
    for area in get_macro_areas(codici_comuni_file):
        mask = gdf_base["PRO_COM_T"].isin(area.comuni)
        geoms = gdf_base.loc[mask, "geometry"]
        if geoms.empty:
            continue
        union_geom = geoms.union_all()
        rows.append({"area_name": area.name, "geometry": union_geom})

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf_base.crs)


# ---------------------------------------------------------------------------
# Map response helpers
# ---------------------------------------------------------------------------


def _build_macro_area_geodataframe(
    result, codici_comuni_file: Path, geometry_file: Path
) -> gpd.GeoDataFrame:
    gdf_areas = (
        _build_macro_area_geometries_cached(codici_comuni_file, geometry_file)
        .rename(columns={"area_name": "AREA_NAME"})
        .copy()
    )

    merged = gdf_areas.merge(
        result.rename(columns={"ID_COMUNE": "AREA_NAME"}),
        on="AREA_NAME",
        how="left",
    )

    return gpd.GeoDataFrame(merged, geometry="geometry", crs=gdf_areas.crs)


def _build_province_geodataframe(gdf_base, result) -> gpd.GeoDataFrame:
    """Join a province-level result to the union of all municipality shapes."""

    province = gpd.GeoDataFrame(
        {
            "PRO_COM_T": ["-1"],
            "COMUNE": ["Provincia"],
            "geometry": [gdf_base.geometry.union_all()],
        },
        geometry="geometry",
        crs=gdf_base.crs,
    )
    merged = province.merge(
        result,
        left_on="PRO_COM_T",
        right_on="ID_COMUNE",
        how="left",
    ).rename(columns={"COMUNE": "AREA_NAME"})

    return gpd.GeoDataFrame(
        merged.drop(columns=["PRO_COM_T", "ID_COMUNE"]),
        geometry="geometry",
        crs=gdf_base.crs,
    )


def _build_geodataframe_vodafone_ids(gdf_base, result, join_on="ID_COMUNE"):
    """
    Same as _build_geodataframe, with ID_COMUNE reflectig AREA_ID scheme (e.g. 'ITA.04.022.100.127')
    """
    merged = (
        gdf_base[["AREA_ID", "AREA_LABEL", "geometry"]]
        .merge(result, left_on="AREA_ID", right_on=join_on, how="left")
        .rename(columns={"AREA_LABEL": "AREA_NAME"})
        .drop(columns=["AREA_ID", join_on])
    )

    return gpd.GeoDataFrame(merged, geometry="geometry", crs=gdf_base.crs)
