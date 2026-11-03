# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from overtourism.backend.auth.identity.sql_repository import SQLUserRepository
from overtourism.backend.auth.identity.user_manager import UserManager
from overtourism.backend.handler import Handler
from overtourism.backend.utils.executor_utils import (
    call_executor,
    list_models,
)
from overtourism.dt_manager.manager.config import BootstrapConfig
from overtourism.dt_manager.manager.manager import Manager
from overtourism.dt_manager.session.config import SessionCleanupConfig
from overtourism.dt_manager.stores.classes.sql.store import SQLStore
from overtourism.dt_manager.stores.config import StoreConfig
from overtourism.dt_manager.utils.exception import EntityDoesNotExist
from overtourism.dt_manager.utils.metadata import ExtrasConfig
from overtourism.overtourism.backend_extension.api.utils.platform import (
    download_index_data,
)

# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────

# Whether execute standalone, with data already prepared
standalone_mode = os.getenv("DT_OVERTURISM_STANDALONE_MODE", "true").lower() == "true"

if not standalone_mode:
    download_index_data()

# ──────────────────────────────────────────────
# Store
# ──────────────────────────────────────────────

data_dir = Path(__file__).parent / "database"
index_data_path = data_dir / "index_data"
database_url = os.getenv(
    "OVERTOURISM_DATABASE", f"sqlite:///{data_dir / 'overtourism.sqlite'}"
)
store_conf = StoreConfig("sql", {"url": database_url})


# ──────────────────────────────────────────────
# Manager
# ──────────────────────────────────────────────

extras_config = ExtrasConfig(
    problem_keys=frozenset(("objective", "links")),
    proposal_keys=frozenset(("resources", "context", "impact")),
    scenario_keys=frozenset("index_diffs"),
)
session_cleanup_config = SessionCleanupConfig()

crud_manager = Manager(
    store_conf,
    extras_config,
    session_cleanup_config=session_cleanup_config,
)
sql_store = SQLStore(database_url)
user_repository = SQLUserRepository(sql_store.engine, sql_store.session_factory)

# ──────────────────────────────────────────────
# Bootstrap entites
# ──────────────────────────────────────────────

with (Path(__file__).parent / "bootstrap_config.json").open(
    encoding="utf-8"
) as config_file:
    bootstrap_config_json: list[dict[str, Any]] = json.load(config_file)

for model in list_models():
    model_config = next(
        (
            config
            for config in bootstrap_config_json
            if config.get("territory") == model["key"]
        ),
        None,
    )
    # Further scenarios of the territory, linked to its base proposal: created when missing,
    # so that an existing platform gets the ones added later.
    extra_scenarios: list[dict[str, Any]] = []
    if model_config is not None:
        model_config = dict(model_config)
        extra_scenarios = model_config.pop("scenarios", [])
    names_cfg = (
        BootstrapConfig(**model_config)
        if model_config is not None
        else BootstrapConfig(model["key"])
    )

    try:
        crud_manager.scenario_manager.read_scenario(names_cfg.scenario_id)
    except EntityDoesNotExist:
        crud_manager.problem_manager.create_problem(
            problem_id=names_cfg.problem_id,
            territory=names_cfg.territory,
            name=names_cfg.problem_name,
            description=names_cfg.problem_description,
        )
        crud_manager.scenario_manager.create_scenario(
            scenario_id=names_cfg.scenario_id,
            territory=names_cfg.territory,
            name=names_cfg.scenario_name,
            description=names_cfg.scenario_description,
            summary=names_cfg.scenario_summary,
            extras=names_cfg.scenario_extras,
        )
        evaluation = crud_manager.create_evaluation(names_cfg.scenario_id)
        try:
            result = call_executor(names_cfg.territory)
        except Exception:
            crud_manager.fail_evaluation(evaluation)
        else:
            crud_manager.complete_evaluation(evaluation, result)
        crud_manager.proposal_manager.create_proposal(
            proposal_id=names_cfg.proposal_id,
            problem_id=names_cfg.problem_id,
            name=names_cfg.proposal_name,
            description=names_cfg.proposal_description,
            status=names_cfg.proposal_status,
            extras=names_cfg.proposal_extras,
        )
        crud_manager.relationship_manager.link_scenario_proposal(
            proposal_id=names_cfg.proposal_id,
            scenario_id=names_cfg.scenario_id,
        )

    for scenario_config in extra_scenarios:
        scenario_id = f"{names_cfg.territory}_{scenario_config['key']}"
        try:
            crud_manager.scenario_manager.read_scenario(scenario_id)
            continue
        except EntityDoesNotExist:
            pass
        crud_manager.scenario_manager.create_scenario(
            scenario_id=scenario_id,
            territory=names_cfg.territory,
            param_overrides=scenario_config["param_overrides"],
            name=scenario_config["name"],
            description=scenario_config["description"],
        )
        evaluation = crud_manager.create_evaluation(scenario_id)
        try:
            result = call_executor(names_cfg.territory, scenario_config["param_overrides"])
        except Exception:
            crud_manager.fail_evaluation(evaluation)
        else:
            crud_manager.complete_evaluation(evaluation, result)
        crud_manager.relationship_manager.link_scenario_proposal(
            proposal_id=names_cfg.proposal_id,
            scenario_id=scenario_id,
        )


# ──────────────────────────────────────────────
# Handler
# ──────────────────────────────────────────────


def build_handler() -> Handler:
    """Build the molveno backend handler and its collaborators."""
    return Handler(
        manager=crud_manager,
        user_manager=UserManager(user_repository),
    )
