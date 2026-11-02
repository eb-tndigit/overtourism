"""API endpoints for data analysis functions."""

import logging
from datetime import datetime
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from overtourism.backend.auth.identity.authorization import require_global_admin
from overtourism.backend.auth.tokens.dependencies import get_auth_context
from overtourism.backend.utils.config import BASE_ROUTE
from overtourism.overtourism.backend_extension.api.models.territorial_config import (
    TerritorialConfig,
)
from overtourism.overtourism.backend_extension.api.models.trenitno_phenomena import (
    PHENOMENON_LABELS_IT,
)
from overtourism.overtourism.backend_extension.api.models.trentino_indicators import (
    _REGISTRY,
    CODICI_COMUNI_FILE,
    MACRO_AREAS_FILE,
    MAP_SHAPEFILE,
    get_indicator,
)
from overtourism.overtourism.backend_extension.api.utils.index_utils import (
    _build_geodataframe,
    _to_map_response,
    _translate_columns,
    get_chart_labels,
)
from overtourism.overtourism.backend_extension.api.utils.index_utils_trentino import (
    _build_macro_area_geodataframe,
    _build_province_geodataframe,
    get_list_comuni,
    get_macro_areas,
    get_map_geometry,
)
from overtourism.overtourism.backend_extension.api.utils.platform import (
    download_index_data,
)

logger = logging.getLogger(__name__)

indexes_router = APIRouter(
    prefix=f"{BASE_ROUTE}/indexes",
    tags=["Indexes"],
    dependencies=[Depends(get_auth_context)],
)

# from pathlib import Path
# PATH_JSON_MAP = Path(__file__).parent.resolve() / 'filename.json'

# ---------------------------------------------------------------------------
# Endpoint-level permission defaults
# Future: replace with per-user resolution from auth token / session.
# ---------------------------------------------------------------------------

# None = all comuni are visible (maximum permission).
_DEFAULT_ALLOWED_COMUNI: set[str] | None = None

# ---------------------------------------------------------------------------
# Shared param: the only territorial input accepted from the frontend
# ---------------------------------------------------------------------------

# Extra query params that are never forwarded to indicator combinators
_KNOWN_PARAMS = {
    "index",
    "start_date",
    "end_date",
    "start_date_baseline",
    "end_date_baseline",
    "start_date_comparison",
    "end_date_comparison",
    "granularity",  # temporal granularity for charts
    "spatial_granularity",  # territorial granularity (frontend)
}


def _build_tc(
    request: Request,
    *,
    allowed_comuni: set[str] | None = _DEFAULT_ALLOWED_COMUNI,
    macro_area_agg: str = "mean",
    column_aggs: dict[str, str] | None = None,
) -> TerritorialConfig:
    """
    Build a TerritorialConfig for a request.

    ``spatial_granularity`` is the only territorial parameter read from the
    query string.  Everything else is server-controlled.

    Parameters
    ----------
    request
        FastAPI request object.
    allowed_comuni
        Hardcoded per endpoint.  ``None`` = all comuni allowed.
    macro_area_agg
        "mean" for ratio/index indicators, "sum" for count indicators.
        Hardcoded per endpoint (or overridable in the future).
    column_aggs
        Aggregation of specific columns next to INDICE (the indicator's
        ``column_aggs``, e.g. "mode" for a category).
    """
    spatial_granularity = request.query_params.get("spatial_granularity", "comune")
    try:
        return TerritorialConfig.build(
            areas=get_macro_areas(MACRO_AREAS_FILE),
            spatial_granularity=spatial_granularity,
            allowed_comuni=allowed_comuni,
            macro_area_agg=macro_area_agg,
            column_aggs=column_aggs,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _extra_params(request: Request) -> dict:
    """Strip all known/territorial params; return indicator-specific extras."""
    return {k: v for k, v in request.query_params.items() if k not in _KNOWN_PARAMS}


# ---------------------------------------------------------------------------
# Date validation helpers
# ---------------------------------------------------------------------------


def _parse_date(value: str | None, field_name: str) -> datetime | None:
    """Parse a YYYY-MM-DD date string, raising a 422 on bad input."""
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid date format for '{field_name}': {value!r}. "
            "Expected YYYY-MM-DD.",
        ) from exc


def _validate_date_order(
    start_dt: datetime | None,
    end_dt: datetime | None,
    start_field: str = "start_date",
    end_field: str = "end_date",
) -> None:
    """Ensure end_dt is not before start_dt (when both are present)."""
    if start_dt is not None and end_dt is not None and end_dt < start_dt:
        raise HTTPException(
            status_code=422,
            detail=f"'{end_field}' ({end_dt.date()}) cannot be before "
            f"'{start_field}' ({start_dt.date()}).",
        )


# ---------------------------------------------------------------------------
# Map value validation helper
# ---------------------------------------------------------------------------


def _validate_map_values(min_value: Any, max_value: Any, *, context: str) -> None:
    """
    Guard against degenerate / invalid map value ranges.

    Raises instead of letting the response go out when:
      - both min_value and max_value are <= 0, or
      - either min_value or max_value is NaN.
    """
    min_is_nan = min_value is None or (
        isinstance(min_value, float) and np.isnan(min_value)
    )
    max_is_nan = max_value is None or (
        isinstance(max_value, float) and np.isnan(max_value)
    )

    if min_is_nan or max_is_nan:
        raise HTTPException(
            status_code=422,
            detail=f"[{context}] Computed value range is invalid "
            f"(min_value={min_value}, max_value={max_value}).",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@indexes_router.get("/get-index-list", response_model=dict[str, Any])
def get_indicators_list():
    """Get list of all available indices."""
    logger.info("[get-index-list] Fetching list of indices")

    try:
        indicators = []
        additional_labels = []

        registered_indicators = [(key, get_indicator(key)) for key in _REGISTRY]

        # Sort by the indicator's declared order.
        registered_indicators.sort(key=lambda item: getattr(item[1], "order", 9999))

        for key, indicator in registered_indicators:
            if not indicator.internal_only:
                try:
                    years_range = getattr(indicator, "years_range", {}) or {}

                    indicators.append(
                        {
                            "value": key,
                            "label": (
                                f"{getattr(indicator, 'category', 'Altro')} - "
                                f"{indicator.name}"
                            ),
                            "index_description": indicator.description,
                            "index_value_unit_description": (
                                indicator.index_value_unit_description
                            ),
                            "availableForVariation": (indicator.availableForVariation),
                            "extraFields": indicator.extraFields,
                            "years_range": {
                                "min_year": years_range.get("min_year"),
                                "max_year": years_range.get("max_year"),
                            },
                        }
                    )
                except:
                    logger.error(f"Unable to retrieved datas for index {indicator}")
                    continue
            else:
                additional_labels.append(
                    {
                        "value": key,
                        "label": indicator.name or key,
                        "index_description": indicator.description,
                        "index_value_unit_description": (
                            indicator.index_value_unit_description
                        ),
                    }
                )

        return {
            "indicators": indicators,
            "additional_labels": additional_labels,
        }

    except Exception:
        logger.exception("[get-index-list] Error fetching indicators")
        raise HTTPException(
            status_code=500,
            detail="Error fetching indicators list",
        )


@indexes_router.get("/get-spatial-areas", response_model=dict[str, Any])
def get_spatial_areas(request: Request):
    """
    Return the list of spatial units visible to the caller.

    The region aggregate ("-1") is always included first.
    """
    logger.info("[get-comuni] Fetching list of comuni")
    try:
        tc = _build_tc(request)

        area_entries = [{"code": "-1", "name": "Provincia"}] + [
            {"code": a.name, "name": a.name} for a in tc.areas
        ]

        # Comune granularity: filter by allowed_comuni
        all_comuni = get_list_comuni(CODICI_COMUNI_FILE)
        if tc.allowed_comuni is not None:
            comuni = [
                c
                for c in all_comuni
                if c["code"] == "-1" or c["code"] in tc.allowed_comuni
            ]
        else:
            comuni = all_comuni

        return {
            "comuni": comuni,
            "areas": area_entries,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _extra_params(request: Request, indicator) -> dict:
    """Return only the query params declared by the indicator."""
    allowed = set(getattr(indicator, "extraFields", ()))
    return {k: v for k, v in request.query_params.items() if k in allowed}


@indexes_router.get("/get_index_data", response_model=dict[str, Any])
def get_index_data(
    request: Request,
    index: str,
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
):
    try:
        indicator = get_indicator(index)
        tc = _build_tc(request, column_aggs=indicator.column_aggs)
        extra = _extra_params(
            request, indicator
        )  # <-- whitelist driven by this indicator

        logger.info(
            f"[{index}] start_date={start_date}, end_date={end_date}, "
            f"extra={extra}, spatial_granularity={tc.spatial_granularity}"
        )

        start_dt = _parse_date(start_date, "start_date")
        end_dt = _parse_date(end_date, "end_date")
        _validate_date_order(start_dt, end_dt)

        gdf_base = get_map_geometry(MAP_SHAPEFILE)

        computed_index = indicator.get_indicator(
            start_date=start_date,
            end_date=end_date,
            **extra,
        )
        if computed_index is None:
            raise RuntimeError("It was not possible to compute the indicator")

        else:
            result = tc.apply(computed_index)
            if tc.spatial_granularity == "provincia":
                gdf_final = _build_province_geodataframe(gdf_base, result)
            elif tc.spatial_granularity == "macro_area":
                gdf_final = _build_macro_area_geodataframe(
                    result, MACRO_AREAS_FILE, MAP_SHAPEFILE
                )
            else:
                gdf_final = _build_geodataframe(gdf_base, result)

            # Translate phenomenon keywords (e.g. "beds", "population") into
            # Italian before serialising to GeoJSON. "AREA_NAME"/"INDICE" are
            # left untouched since they aren't in PHENOMENON_LABELS_IT.
            gdf_final = _translate_columns(gdf_final, PHENOMENON_LABELS_IT)

            geo_data = _to_map_response(gdf_final)

        _validate_map_values(
            geo_data["min_value"],
            geo_data["max_value"],
            context=f"{index} get_index_data",
        )

        response: dict[str, Any] = {
            "geo_data": {
                "data": geo_data["geojson"],
                "min_value": geo_data["min_value"],
                "max_value": geo_data["max_value"],
            },
        }

        # Optional overlay: indicators that also describe movements between areas (e.g. the redistribution
        # indicator) expose get_flows(); the arrows follow the same window and permissions as the map. Other
        # indicators are untouched. Meaningless for a single province-wide unit, so skipped there.
        if tc.spatial_granularity != "provincia" and hasattr(indicator, "get_flows"):
            flows = indicator.get_flows(
                start_date,
                end_date,
                allowed_comuni=tc.allowed_comuni,
                **extra,
            )
            if flows is not None:
                response["flows"] = flows

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


@indexes_router.post(
    "/download-index-data",
    response_model=dict[str, str],
    dependencies=[Depends(require_global_admin)],
)
def force_download_index_data() -> dict[str, str]:
    """Force a fresh download of the index data from the platform."""
    download_index_data(overwrite=True)
    return {"message": "Index data download completed"}


@indexes_router.get("/get-variation-data", response_model=dict[str, Any])
def get_variation_data(
    request: Request,
    index: str,
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    granularity: str | None = Query(None),
    selected_territories: list[str] | None = Query(
        None,
        description="Optional comune IDs or macro-area IDs to include in the result.",
    ),
):
    """
    Time-series variation for spatial units. When ``selected_territories`` is
    provided, only the matching comune or macro-area series are returned. With
    ``spatial_granularity=provincia``, only the province-wide series is returned
    when no territories are selected.

    ``granularity`` accepts daily/weekly/monthly/yearly values; these are normalized before computing the time series.
    ``spatial_granularity`` (query param) accepts ``comune``, ``macro_area``,
    or ``provincia`` and controls the territorial grain.
    """
    try:
        tc = _build_tc(request)

        logger.info(
            f"[{index} variation] start_date={start_date}, end_date={end_date}, "
            f"temporal={granularity}, spatial={tc.spatial_granularity}, "
            f"selected_territories={selected_territories}"
        )

        start_dt = _parse_date(start_date, "start_date")
        end_dt = _parse_date(end_date, "end_date")
        if start_dt is None or end_dt is None:
            raise HTTPException(
                status_code=422,
                detail="Both 'start_date' and 'end_date' are required.",
            )
        _validate_date_order(start_dt, end_dt)

        labels = get_chart_labels(start_dt, end_dt, granularity)
        indicator = get_indicator(index)
        raw_series = indicator.get_temporal_variation(
            start_date=start_dt,
            end_date=end_dt,
            granularity=granularity,
        )

        series = tc.apply_to_series(raw_series)
        if selected_territories is not None:
            selected_ids = {str(territory) for territory in selected_territories}
            series = [s for s in series if str(s["label"]) in selected_ids]

            # Keep the series contract intact, but show human-readable names
            # instead of IDs when the caller explicitly selected territories.
            names_by_id = {
                str(area["code"]): area["name"]
                for area in get_list_comuni(CODICI_COMUNI_FILE)
            }
            for item in series:
                territory_id = str(item["label"])
                item["label"] = names_by_id.get(territory_id, territory_id)

        return {
            "labels": labels,
            "series": series,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))
