"""Adapters registry. Importing the package registers all the built-in adapters."""

from utils.adapters import (  # noqa: F401
    energia,
    meteo,
    popolazione,
    presenze,
    strutture,
    vodafone,
)
from utils.adapters.base import (  # noqa: F401
    KIND_DATASETS,
    KIND_SCHEMAS,
    Adapter,
    RawData,
    SchemaMismatchError,
    check_adapter_params,
    get_adapter,
    list_adapters,
    register_adapter,
    validate_schema,
)
