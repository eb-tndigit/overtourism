"""Scenari what-if per il modello di overtourism di Fazzon (Lago dei Caprioli, Val di Sole).

Ogni :class:`WhatIfScenario` porta un'etichetta di visualizzazione, una descrizione
(tabella Markdown mostrata sotto la figura del campo) e una ``overrides_fn`` che mappa
da un'istanza live di :class:`FazzonModel` a un dict di override compatibile con Scenario.

Utilizzo::

    from overtourism.model.fazzon.fazzon_scenarios import SCENARIO_BY_KEY
    scenario_def = SCENARIO_BY_KEY["a1_no_cap"]
    overrides = scenario_def.overrides_fn(model)
    scenario = Scenario(model, overrides=overrides)

Per aggiungere un nuovo scenario: definire una costante :class:`WhatIfScenario` e aggiungerla
a :data:`ALL_SCENARIOS`.  La dashboard recepisce la modifica automaticamente.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from scipy import stats

if TYPE_CHECKING:
    from src.model.fazzon.fazzon_model import FazzonModel

# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WhatIfScenario:
    """Scenario what-if con overrides dei parametri per il modello di Fazzon."""

    key: str
    label: str
    category: str
    description: str
    overrides_fn: Callable[[FazzonModel], dict] = field(
        default=lambda m: {}, compare=False, hash=False
    )


# ---------------------------------------------------------------------------
# Distribution helpers (return frozen scipy distributions)
# ---------------------------------------------------------------------------


def _triang(loc: float, scale: float, c: float):
    return stats.triang(c=c, loc=loc, scale=scale)


def _uniform(loc: float, scale: float):
    return stats.uniform(loc=loc, scale=scale)


# ---------------------------------------------------------------------------
# Riferimento — As-Is (regole 2025)
# ---------------------------------------------------------------------------

AS_IS = WhatIfScenario(
    key="as_is",
    label="As-is (regole 2025)",
    category="Riferimento",
    description=(
        "Nessun intervento: le regole della stagione 2025.  Parcheggio a 5 € fino a "
        "esaurimento posti, nessuna navetta.  È anche la condizione in cui è espressa la "
        "domanda sugli assi del grafico."
    ),
)

# ---------------------------------------------------------------------------
# Categoria A — Capacità di parcheggio
#
# La variabile di indice i_c_parking rappresenta il numero massimo di auto
# in parcheggio simultaneo (non giornaliero).  I valori di riferimento
# riflettono il contingentamento 2025 (≈200 auto simultanee).
# ---------------------------------------------------------------------------

A1_NO_CAP = WhatIfScenario(
    key="a1_no_cap",
    label="A1 – Senza cap (baseline storico 2022)",
    category="A — Capacità",
    description=(
        "Scenario pre-intervento 2022: il contingentamento non è ancora in vigore. "
        "La capacità di parcheggio simultaneo sale a 400 auto (stima del massimo "
        "storico prima della regolamentazione).\n\n"
        "| Parametro | Riferimento (2025) | Scenario |\n"
        "| --- | --- | --- |\n"
        "| cap parcheggio simultaneo | triang, moda 200, range [150, 250] auto"
        " | **triang, moda 400, range [350, 450] auto** |"
    ),
    overrides_fn=lambda m: {
        m.i_c_parking: _triang(loc=350.0, scale=100.0, c=0.5),
    },
)

A2_REDUCED_CAP = WhatIfScenario(
    key="a2_reduced_cap",
    label="A2 – Cap ridotto (100 auto simultanee)",
    category="A — Capacità",
    description=(
        "Inasprimento del contingentamento: il numero massimo di auto in parcheggio "
        "simultaneo scende a 100.  Questo scenario testa la resilienza del sistema "
        "a una riduzione ulteriore dell'accesso motorizzato.\n\n"
        "| Parametro | Riferimento (2025) | Scenario |\n"
        "| --- | --- | --- |\n"
        "| cap parcheggio simultaneo | triang, moda 200, range [150, 250] auto"
        " | **triang, moda 100, range [80, 120] auto** |"
    ),
    overrides_fn=lambda m: {
        m.i_c_parking: _triang(loc=80.0, scale=40.0, c=0.5),
    },
)

# ---------------------------------------------------------------------------
# Categoria B — Tariffa e navetta
#
# cv_parking_price: prezzo medio pagato per veicolo, da 5 a 30 € a passi di 1 € (A24, D15); 2025 = 5 €.
# i_shuttle_buses: bus della navetta (2 x 30 posti ogni 30 minuti, 9-19 = Lakebus 2026, A29).
# La risposta alla tariffa e la ripartizione fra navetta / a piedi / altrove sono
# stimate sui dati 2025-2026 (pipeline Fazzon, fazzon_modal_shift; D8).
# ---------------------------------------------------------------------------

B1_SHUTTLE_ONLY = WhatIfScenario(
    key="b1_shuttle_only",
    label="B1 – Navetta senza tariffa",
    category="B — Tariffa e navetta",
    description=(
        "Navetta Lakebus (2 bus da 30 posti ogni 30 minuti, 9-19) con il parcheggio a 5 € come nel 2025.  "
        "Senza un incentivo a lasciare l'auto la navetta non sposta visitatori (come il "
        "Greenbus pilota del 2025): aggiunge solo passaggi sulla strada.\n\n"
        "| Parametro | Riferimento (2025) | Scenario |\n"
        "| --- | --- | --- |\n"
        "| tariffa parcheggio | 5 € | 5 € |\n"
        "| bus navetta (30 posti, ogni 30 min, 9-19) | 0 | **2** |"
    ),
    overrides_fn=lambda m: {
        m.i_shuttle_buses: 2,
    },
)

B2_PRICE_10 = WhatIfScenario(
    key="b2_price_10",
    label="B2 – Tariffa 10 € + navetta",
    category="B — Tariffa e navetta",
    description=(
        "Tariffa intermedia di 10 € per veicolo con la navetta Lakebus.  È fuori dai due "
        "livelli di prezzo osservati (5 € e ~15 €): l'effetto deriva dalla forma "
        "esponenziale della risposta alla tariffa (A23).\n\n"
        "| Parametro | Riferimento (2025) | Scenario |\n"
        "| --- | --- | --- |\n"
        "| tariffa parcheggio | 5 € | **10 €** |\n"
        "| bus navetta (30 posti, ogni 30 min, 9-19) | 0 | **2** |"
    ),
    overrides_fn=lambda m: {
        m.cv_parking_price: "10",
        m.i_shuttle_buses: 2,
    },
)

# ---------------------------------------------------------------------------
# Categoria C — Pacchetti di misure
# ---------------------------------------------------------------------------

C1_PIANO_2026 = WhatIfScenario(
    key="c1_piano_2026",
    label="C1 – Regole 2026 (tariffa + navetta)",
    category="C — Pacchetti",
    description=(
        "Le regole 2026 come osservate: parcheggio a 20 € la giornata e 10 € dopo le 12 "
        "(prezzo medio pagato 15 €), prenotazione e semaforo, navetta Lakebus ogni 30 "
        "minuti.  La risposta alla tariffa è stimata proprio su questo periodo: lo scenario "
        "riproduce il 2026 e serve da riferimento per gli altri.\n\n"
        "| Parametro | Riferimento (2025) | Scenario |\n"
        "| --- | --- | --- |\n"
        "| tariffa parcheggio | 5 € | **15 €** (media pagata) |\n"
        "| bus navetta (30 posti, ogni 30 min, 9-19) | 0 | **2** |"
    ),
    overrides_fn=lambda m: {
        m.cv_parking_price: "15",
        m.i_shuttle_buses: 2,
    },
)

C2_STRONG = WhatIfScenario(
    key="c2_strong",
    label="C2 – Intervento forte (cap 80 + tariffa 20 € + navetta)",
    category="C — Pacchetti",
    description=(
        "Massima restrizione: contingentamento a 80 auto simultanee, tariffa di 20 € per "
        "ogni veicolo (nessuno sconto pomeridiano) e navetta Lakebus.\n\n"
        "| Parametro | Riferimento (2025) | Scenario |\n"
        "| --- | --- | --- |\n"
        "| cap parcheggio simultaneo | triang, moda 200, range [150, 250] auto"
        " | **triang, moda 80, range [60, 100] auto** |\n"
        "| tariffa parcheggio | 5 € | **20 €** |\n"
        "| bus navetta (30 posti, ogni 30 min, 9-19) | 0 | **2** |"
    ),
    overrides_fn=lambda m: {
        m.i_c_parking: _triang(loc=60.0, scale=40.0, c=0.5),
        m.cv_parking_price: "20",
        m.i_shuttle_buses: 2,
    },
)

# ---------------------------------------------------------------------------
# Catalogo ordinato — la dashboard itera questa lista per costruire il menu
# ---------------------------------------------------------------------------

ALL_SCENARIOS: list[WhatIfScenario] = [
    AS_IS,
    A1_NO_CAP,
    A2_REDUCED_CAP,
    B1_SHUTTLE_ONLY,
    B2_PRICE_10,
    C1_PIANO_2026,
    C2_STRONG,
]

SCENARIO_BY_KEY: dict[str, WhatIfScenario] = {s.key: s for s in ALL_SCENARIOS}
