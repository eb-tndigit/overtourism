"""Analisi dei nuovi dati AIxPA 2025 per Fazzon / Alta Val di Sole.

Script semplice con marcatori di cella ``# %%``, eseguibile sia da riga di
comando sia cella-per-cella in un REPL / finestra interattiva stile Jupyter
(VS Code, PyCharm, `jupytext`, ...). Dalla cartella ``data_preparation``::

    python -m model.fazzon.prepare           # una volta: produce i dataset (vedi prepare.py)
    python -m model.fazzon.exploratory_analysis

I dati sono letti dai parquet prodotti dalla pipeline Fazzon
(``<normalized>/fazzon/`` e ``<final>/fazzon/``), non dai file sorgente.

Cosa fa, in ordine:

1. Costruisce la serie giornaliera target di Fazzon: auto (biglietti
   parcheggio, dato misurato) e biglietti navetta bus, per la stagione 2025.
2. Estrae le serie predittive giornaliere per la Val di Sole dai nuovi dati:
   presenze telefoniche Vodafone (granularita' POI/comune/apt/provincia),
   meteo (Meteotrentino, stazione di Male'), contatori bici/pedoni sulla
   ciclabile, utilizzo Guest Card e passaggi di autoveicoli rilevati dalle
   stazioni di conteggio del traffico della Provincia (Servizio Gestione
   Strade) in Val di Sole / Val di Non.
3. Calcola le correlazioni grezze e destagionalizzate (entro-mese) con il
   target, e una matrice di correlazione completa tra tutte le variabili.
4. Verifica l'ipotesi di saturazione del canale auto (appiattimento della
   pendenza ad alta domanda) confrontandolo con il canale bici/pedoni. Contro
   le presenze Vodafone la pendenza cala, contro il traffico misurato a
   Pellizzano (punto 7) no: la saturazione non e' confermata.
5. Esegue l'"esperimento naturale" della navetta: confronta le auto per un
   dato livello di domanda/meteo tra il periodo in cui la navetta era
   realmente attiva (28/06/2025 -> 20/07/2025) e il resto della stagione.
6. Mostra la copertura dei dati della navetta pilota 2025: i biglietti non
   coprono i giorni di domanda più alta della stagione, quindi con i soli dati
   2025 lo spostamento modale non è stimabile (lo è con i dati 2026, punto 8 e
   ``modal_shift.py``).
7. Usa i conteggi stradali (auto *misurate* sulla SS42 a Pellizzano) come
   segnale di domanda alternativo alle presenze Vodafone: ripete l'analisi di
   saturazione e l'esperimento navetta, e confronta il periodo navetta 2025
   con lo stesso periodo del 2023/2024 (differenza-nelle-differenze rispetto a
   stazioni di controllo in valli laterali).
8. Confronta la stagione 2026 (parcheggio a pagamento e a numero chiuso,
   semaforo, navetta "Lakebus") con la stagione 2025 (parcheggio a 5 euro fino a
   esaurimento posti, navetta pilota Greenbus solo 28/6-20/7), a parita' di
   giorno della settimana.
9. Mostra i dataset letti dal modello: i livelli di auto per stagione e meteo
   (``fazzon_presence_stats``) contro i dati osservati, la ripartizione dei
   visitatori che non salgono piu' in auto e la risposta al prezzo del
   parcheggio (``fazzon_modal_shift``).
10. Mostra cosa si puo' misurare in ogni anno (condizioni di accesso osservate e
    dati disponibili giorno per giorno) e l'evidenza delle scelte del modello di
    spostamento modale: il riferimento in condizioni 2025 dalla relazione con il
    traffico (A37), confrontato fuori campione con i livelli stagione x meteo;
    l'effetto anno 2026 (A22), con le sue misure e la sensibilita' delle stime;
    gli indicatori di monitoraggio (D20).

Decisioni e ipotesi (con quelle ancora da validare con gli esperti di dominio)
sono elencate in ``ASSUMPTIONS.md`` (stessa cartella).

Correzioni di qualita' dei dati applicate dalla pipeline: il file dei
biglietti bus 2025 contiene cinque righe sparse in agosto relative a un'altra
area, non Fazzon, e la navetta 2025 non ha funzionato al di fuori
dell'intervallo 28/06/2025 -> 20/07/2025.

Nota sulla scelta delle variabili: dove possibile si usa il numero di auto
(biglietti di parcheggio venduti, dato *misurato*) invece dei "visitatori in
auto" (biglietti x 2.83 persone/auto, dato *inferito*) — vedi
``PERSONS_PER_CAR`` più sotto, calcolato comunque e disponibile nel pannello
per chi necessita della stima in persone.

Output (scritti in ``OUTPUT_DIR``): ``panel_daily.csv``, ``correlations.csv``,
``correlation_matrix.csv``, ``saturation_slopes.csv``, ``shuttle_road_did.csv``,
``measurement_space.csv``, ``reference_cross_validation.csv``, ``year_effect_checks.csv``,
``year_effect_sensitivity.csv`` e i grafici PNG numerati, con etichette in italiano.
"""

# %% Import e configurazione
import sys
from pathlib import Path

if __package__ in (None, ""):  # eseguito come file: rende importabile model.*
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import stats

from model.fazzon.modal_shift import (
    DEMAND_PREDICTOR,
    YEAR_EFFECT_UNCERTAINTY,
    calibration_days,
    estimate_point,
    expected_cars,
    fit_demand,
)
from model.fazzon.monitoring import CARS_PER_MORNING_CAR, CARS_VS_RELATION
from model.fazzon.prepare import read_normalized, read_reference
from model.fazzon.reference_datasets import (
    EXCLUDED_DAYS,
    PRICE_2025,
    REFERENCE_OCCUPANCY,
    REGULATED_PARKING,
    SEASONS,
    SHUTTLE_PERIODS,
    WEATHER_CLASSES,
    occupancy,
)

FAZZON_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = FAZZON_DIR / "data" / "analysis_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PERSONS_PER_CAR = 2.83  # A1: NetMobility 2021 field study

# Regime 2026 (definito in reference_datasets.py): parcheggio a pagamento con
# semaforo e prenotazione dal 4/7 al 31/8, navetta Lakebus dal 4/7 al 30/8.
REGIME_2026_START, REGIME_2026_PARKING_END = REGULATED_PARKING[2026]
REGIME_2026_SHUTTLE_END = SHUTTLE_PERIODS["lakebus"][1]
# Persone per veicolo secondo il gestore (ASSUMPTIONS.md, A2): ~2,5 senza
# regime (2025, e 2026 fuori dal periodo regolato); ~3,5 con il regime 2026,
# perche' con parcheggio a 20 euro "si sale solo con l'auto piena".
OCCUPANCY_NO_REGIME = 2.5  # A2
OCCUPANCY_REGIME_2026 = 3.5  # A2

# Meteo: stazione Meteotrentino di Male' (T0074), misure giornaliere.
RAINY_DAY_MM = 1.0  # A18: soglia climatologica standard di "giorno piovoso"

BIKE_SENSOR_SHORT_NAMES = {  # etichette per la mappa
    "LLWW78M8": "Laghetti S. Leonardo", "GLWB8Q97": "Fucine di Ossana", "TFXV3R06": "Salèt (Pellizzano)",
    "RXSLG442": "Daolasa", "FFXV2R01": "Molini (Caldes)",
}
# Stazioni di controllo per il confronto anno-su-anno: valli laterali non
# attraversate dal traffico verso Fazzon (Val di Rabbi, Val di Non) e Val di
# Peio. Celledizzo non e' un controllo del tutto pulito: gli ospiti della Val
# di Peio diretti a Fazzon ci passano (scendendo verso Fucine/Ossana), ma
# sono al massimo poche decine di auto su ~3.800/giorno (~1%).
ROAD_CONTROL_STATIONS = ["pracorno", "celledizzo", "revo"]

# Etichette in italiano usate nei grafici (non nel codice, per leggibilita' del sorgente)
IT_LABELS = {
    "parking_tickets": "Auto/giorno (biglietti parcheggio)",
    "car_visitors": "Visitatori in auto (stimati)",
    "bus_tickets": "Biglietti navetta/giorno",
    "vf_pres_apt_valdisole": "Presenze Vodafone - Valli di Sole (APT)",
    "vf_pres_provincia_trentino": "Presenze Vodafone - Provincia di Trento",
    "vf_pres_comune_vermiglio": "Presenze Vodafone - Comune di Vermiglio",
    "vf_pres_comune_ossana_pellizzano": "Presenze Vodafone - Ossana+Pellizzano",
    "vf_pres_poi_caprioli": "Presenze Vodafone - Lago dei Caprioli (POI)",
    "bici_valley_total": "Bici/pedoni - intera Val di Sole",
    "bici_near_fazzon": "Bici/pedoni - vicino a Fazzon",
    "tgc_accessi_tonale": "Accessi Guest Card - Passo Tonale",
    "strada_pellizzano_salita_mattina": "Auto SS42 Pellizzano, verso monte 7-12",
    "strada_pellizzano_totale": "Auto SS42 Pellizzano, totale giornaliero",
    "strada_pracorno_totale": "Auto Pracorno (Val di Rabbi), totale",
    "strada_celledizzo_totale": "Auto Celledizzo (Val di Peio), totale",
    "strada_tonale_totale": "Auto Passo del Tonale, totale",
    "strada_revo_totale": "Auto Revò (Val di Non), totale",
    "tmax": "Temperatura massima (C, stazione Male')",
    "tmin": "Temperatura minima (C, stazione Male')",
    "rain_mm": "Pioggia (mm, stazione Male')",
    "tmax_bollettino": "Temperatura massima (C, bollettino)",
    "rain_mm_bollettino": "Pioggia (mm, bollettino)",
    "is_rainy": "Giorno piovoso",
    "is_weekend": "Fine settimana",
}

# --- palette (palette di default dello skill dataviz del progetto) -------
C_BLUE = "#2a78d6"
C_ORANGE = "#eb6834"
C_AQUA = "#1baf7a"
C_VIOLET = "#4a3aa7"
C_RED = "#e34948"  # categorico slot 8, usato anche come polo del diverging
C_RED_STATUS = "#d03b3b"
C_GOOD_STATUS = "#0ca30c"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_NEUTRAL_MID = "#f0efec"
C_SURFACE_ORANGE = "#fbe1d6"  # riempimento chiaro per marcatori arancio secondari
SURFACE = "#fcfcfb"

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "blue_red_diverging", [C_BLUE, C_NEUTRAL_MID, C_RED]
)

WEATHER_COLORS = {"good": C_BLUE, "cool": C_AQUA, "rainy": C_VIOLET}
WEATHER_LABELS = {"good": "Bel tempo", "cool": "Fresco", "rainy": "Pioggia"}
SEASON_LABELS = {"june": "Giugno", "july": "Luglio", "august": "Agosto", "september": "Settembre"}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": C_AXIS,
    "axes.labelcolor": C_INK2,
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.8,
    "text.color": C_INK,
    "xtick.color": C_INK2,
    "ytick.color": C_INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "figure.dpi": 130,
})


# %% 1-3. Dati: dataset della pipeline Fazzon e pannello giornaliero 2025
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """``fazzon_daily`` (dataset di riferimento), validita' giornaliera dei
    contatori stradali, anagrafica delle stazioni e tutti i dataset di
    riferimento, prodotti da ``python -m model.fazzon.prepare``."""
    try:
        reference = read_reference()
        normalized = read_normalized()
    except FileNotFoundError as e:
        raise SystemExit(f"{e}\nEseguire prima, da data_preparation: python -m model.fazzon.prepare") from None
    return reference["fazzon_daily"], normalized["road_counts_daily"], normalized["stations"], reference


def build_panel(daily: pd.DataFrame) -> pd.DataFrame:
    """Stagione Fazzon 2025 (giorni con conteggio auto): target e predittori.

    ``bus_tickets`` = biglietti Greenbus venduti, solo nel periodo della navetta
    pilota; ``shuttle_running`` = navetta pilota attiva.
    """
    p = daily[(daily["year"] == 2025) & daily["cars"].notna()].copy()
    p = p.rename(columns={"cars": "parking_tickets", "greenbus_tickets": "bus_tickets"})
    p["car_visitors"] = p["parking_tickets"] * PERSONS_PER_CAR
    p["shuttle_running"] = p["shuttle"].eq("greenbus_pilot")
    p["is_rainy"] = (p["rain_mm"] >= RAINY_DAY_MM).astype(int)
    keep = ["date", "parking_tickets", "car_visitors", "shuttle_running", "bus_tickets", "dow", "month",
            "is_weekend"]
    predictors = [c for c in p.columns if c.startswith(("vf_pres_", "bici_", "tgc_", "strada_"))]
    weather = ["weather_desc", "tmin", "tmax", "rain_mm", "meteo_quality", "tmin_bollettino", "tmax_bollettino",
               "rain_mm_bollettino", "is_rainy"]
    return p[keep + predictors + weather].reset_index(drop=True)


# %% 4. Tabella di correlazione (grezza + entro-mese) e matrice di correlazione
TARGET_COL = "parking_tickets"  # dato misurato; car_visitors ne è un multiplo costante


def correlation_table(panel: pd.DataFrame) -> pd.DataFrame:
    candidate_cols = [
        c for c in panel.columns
        if c.startswith(("vf_pres_", "bici_", "tgc_", "strada_"))
        or c in ("tmax", "tmin", "rain_mm", "is_rainy", "is_weekend", "tmax_bollettino", "rain_mm_bollettino")
    ]

    def within_month(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        return s - s.groupby(panel["month"]).transform("mean")

    y_raw = panel[TARGET_COL].astype(float)
    y_resid = within_month(panel[TARGET_COL])

    rows = []
    for col in candidate_cols:
        x_raw = panel[col].astype(float)
        mask = x_raw.notna() & y_raw.notna()
        if mask.sum() < 10:
            continue
        r_raw, p_raw = stats.pearsonr(x_raw[mask], y_raw[mask])

        x_resid = within_month(panel[col])
        r_wm, p_wm = stats.pearsonr(x_resid[mask], y_resid[mask])

        rows.append({
            "predictor": col, "n": mask.sum(),
            "raw_r": r_raw, "raw_r2": r_raw ** 2, "raw_p": p_raw,
            "within_month_r": r_wm, "within_month_r2": r_wm ** 2, "within_month_p": p_wm,
        })
    return pd.DataFrame(rows).sort_values("within_month_r2", ascending=False).reset_index(drop=True)


CORRELATION_MATRIX_COLS = [
    "parking_tickets", "bus_tickets",
    "vf_pres_apt_valdisole", "vf_pres_provincia_trentino", "vf_pres_comune_vermiglio",
    "vf_pres_comune_ossana_pellizzano", "vf_pres_poi_caprioli",
    "bici_valley_total", "bici_near_fazzon",
    "strada_pellizzano_salita_mattina", "strada_pellizzano_totale", "strada_pracorno_totale",
    "strada_celledizzo_totale", "strada_revo_totale", "strada_tonale_totale",
    "tgc_accessi_tonale", "tmax", "tmin", "rain_mm", "is_weekend",
]


def correlation_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """Matrice di correlazione di Pearson a coppie complete tra tutte le
    variabili di interesse (utile per relazioni fuori dal target, ad es.
    is_weekend vs bici_*, non catturate dalla tabella di correlazione sopra).
    """
    return panel[CORRELATION_MATRIX_COLS].corr(method="pearson")


def correlation_road_vs_presences(panel: pd.DataFrame) -> float:
    """Correlazione entro-mese tra le due misure di domanda (auto SS42 vs
    presenze Vodafone in valle): quanto si sovrappongono."""
    a = panel["strada_pellizzano_salita_mattina"] - panel.groupby("month")["strada_pellizzano_salita_mattina"].transform("mean")
    b = panel["vf_pres_apt_valdisole"] - panel.groupby("month")["vf_pres_apt_valdisole"].transform("mean")
    return a.corr(b)


# %% 5. Analisi di saturazione: canale auto vs canale bici/pedoni
def _find_optimal_breakpoint(x: pd.Series, y: pd.Series, min_frac: float = 0.2) -> dict:
    """Cerca (per griglia) il punto di rottura che minimizza la somma degli
    errori quadratici di due rette OLS indipendenti (una per lato).

    Non e' una regressione segmentata continua (le due rette non sono
    vincolate a incontrarsi nel breakpoint): stesso impianto "due rette
    indipendenti" usato per la soglia mediana, solo con la soglia scelta per
    minimizzare l'errore totale invece che fissata a priori.

    ``min_frac`` impone almeno questa frazione di punti per lato, per evitare
    rotture agli estremi con rette stimate su pochissimi punti.
    """
    x = x.to_numpy(dtype=float)
    y = y.to_numpy(dtype=float)
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    n = len(xs)
    min_n = max(5, int(np.ceil(min_frac * n)))

    best = None
    for c in np.unique(xs):
        lo_mask, hi_mask = xs <= c, xs > c
        if lo_mask.sum() < min_n or hi_mask.sum() < min_n:
            continue
        fit_lo = stats.linregress(xs[lo_mask], ys[lo_mask])
        fit_hi = stats.linregress(xs[hi_mask], ys[hi_mask])
        sse = (
            np.sum((ys[lo_mask] - (fit_lo.intercept + fit_lo.slope * xs[lo_mask])) ** 2)
            + np.sum((ys[hi_mask] - (fit_hi.intercept + fit_hi.slope * xs[hi_mask])) ** 2)
        )
        if best is None or sse < best["sse"]:
            best = {
                "breakpoint": c, "sse": sse, "fit_lo": fit_lo, "fit_hi": fit_hi,
                "n_lo": int(lo_mask.sum()), "n_hi": int(hi_mask.sum()),
            }

    sst = np.sum((ys - ys.mean()) ** 2)
    single_fit = stats.linregress(xs, ys)
    sse_single = np.sum((ys - (single_fit.intercept + single_fit.slope * xs)) ** 2)
    best["r2_single_line"] = 1 - sse_single / sst
    best["r2_two_segments"] = 1 - best["sse"] / sst
    return best


def saturation_slopes(
    panel: pd.DataFrame, demand_col: str = "vf_pres_apt_valdisole", mode: str = "median", min_frac: float = 0.2,
) -> pd.DataFrame:
    """Confronta la pendenza a domanda bassa/alta per auto e traffico bici/pedoni.

    ``demand_col`` deve essere un segnale di domanda esterno alla capacita'
    fisica di Fazzon stessa (default: presenze Vodafone nell'intera valle),
    perché il confronto non sia circolare.

    ``mode="median"``: soglia fissa alla mediana della domanda (stesso numero
    di giorni per lato, non ottimizzata).
    ``mode="optimal"``: soglia scelta per canale, per griglia, minimizzando
    l'errore quadratico totale delle due rette (vedi ``_find_optimal_breakpoint``).
    Con n=82 e una soglia scelta sui dati stessi, il punto di rottura va letto
    come indicativo, non come stima precisa del limite fisico di capacita'.
    """
    D = panel[demand_col].astype(float)
    rows = []
    for y_col in ["parking_tickets", "bici_near_fazzon", "bici_valley_total"]:
        if mode == "median":
            bp = D.median()
            lo, hi = panel[D <= bp], panel[D > bp]
            fit_lo = stats.linregress(lo[demand_col], lo[y_col])
            fit_hi = stats.linregress(hi[demand_col], hi[y_col])
            r2_single = r2_two = np.nan
        elif mode == "optimal":
            best = _find_optimal_breakpoint(D, panel[y_col], min_frac=min_frac)
            bp, fit_lo, fit_hi = best["breakpoint"], best["fit_lo"], best["fit_hi"]
            r2_single, r2_two = best["r2_single_line"], best["r2_two_segments"]
        else:
            raise ValueError(f"mode sconosciuta: {mode!r}")

        rows.append({
            "channel": y_col, "mode": mode, "breakpoint": bp,
            "slope_low_demand": fit_lo.slope, "slope_high_demand": fit_hi.slope,
            "pct_change": (fit_hi.slope - fit_lo.slope) / fit_lo.slope * 100,
            "r2_single_line": r2_single, "r2_two_segments": r2_two,
        })
    return pd.DataFrame(rows)


# %% 6. "Esperimento naturale" della navetta
def shuttle_natural_experiment(panel: pd.DataFrame, demand_col: str = "vf_pres_apt_valdisole") -> dict:
    """Confronta le auto per un dato livello di domanda/meteo tra il periodo
    in cui la navetta era realmente attiva (28/06/2025 -> 20/07/2025) e il
    resto della stagione, limitandosi all'intervallo di domanda in cui i due
    periodi si sovrappongono davvero.
    """
    era = panel["shuttle_running"]
    lo_overlap = max(panel.loc[era, demand_col].min(), panel.loc[~era, demand_col].min())
    hi_overlap = min(panel.loc[era, demand_col].max(), panel.loc[~era, demand_col].max())
    in_overlap = panel[demand_col].between(lo_overlap, hi_overlap)

    overlap = panel[in_overlap].copy()
    shuttle_days = overlap[overlap["shuttle_running"]]
    no_shuttle_days = overlap[~overlap["shuttle_running"]]

    # Regressione multipla sull'intera stagione: auto ~ domanda + temp.max + pioggia + navetta_attiva
    X = panel[[demand_col, "tmax", "rain_mm"]].astype(float).copy()
    X["shuttle_running"] = panel["shuttle_running"].astype(float)
    X.insert(0, "const", 1.0)
    y = panel[TARGET_COL].astype(float)
    mask = X.notna().all(axis=1) & y.notna()
    coef, residuals, *_ = np.linalg.lstsq(X[mask].values, y[mask].values, rcond=None)
    yhat = X[mask].values @ coef
    resid = y[mask].values - yhat
    n, k = mask.sum(), X.shape[1]
    dof = n - k
    mse = (resid ** 2).sum() / dof
    XtX_inv = np.linalg.inv(X[mask].values.T @ X[mask].values)
    se = np.sqrt(np.diag(mse * XtX_inv))
    coef_names = list(X.columns)
    shuttle_idx = coef_names.index("shuttle_running")
    shuttle_coef = coef[shuttle_idx]
    shuttle_se = se[shuttle_idx]
    t_stat = shuttle_coef / shuttle_se
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), dof))

    return {
        "overlap_demand_range": (lo_overlap, hi_overlap),
        "n_shuttle_days_in_overlap": len(shuttle_days),
        "n_no_shuttle_days_in_overlap": len(no_shuttle_days),
        "mean_parking_tickets_shuttle_running": shuttle_days[TARGET_COL].mean(),
        "mean_parking_tickets_no_shuttle": no_shuttle_days[TARGET_COL].mean(),
        "regression_coef_names": coef_names,
        "regression_coef": coef,
        "shuttle_running_coef": shuttle_coef,
        "shuttle_running_se": shuttle_se,
        "shuttle_running_p": p_val,
        "n_obs_regression": n,
    }


def shuttle_road_did(daily: pd.DataFrame, years=(2023, 2024, 2025)) -> pd.DataFrame:
    """Confronto anno-su-anno sulla SS42 a Pellizzano.

    Per ogni anno: rapporto medio fra auto verso monte al mattino a
    Pellizzano e auto totali nelle stazioni di controllo, nella finestra della
    navetta (28/06-20/07) e nel periodo successivo (21/07-31/08). Se la
    navetta avesse tolto auto dalla strada per Fazzon, nel 2025 il rapporto
    finestra/dopo dovrebbe essere piu' basso che nel 2023/2024. Il 2022 e'
    escluso: la direzione 2 a Pellizzano manca per gran parte dell'estate.
    """
    daily = daily.set_index("date")
    controls = daily[[f"strada_{s}_totale" for s in ROAD_CONTROL_STATIONS]].sum(axis=1, min_count=len(ROAD_CONTROL_STATIONS))
    ratio = daily["strada_pellizzano_salita_mattina"] / controls

    rows = []
    for year in years:
        window = ratio[f"{year}-06-28":f"{year}-07-20"].dropna()
        later = ratio[f"{year}-07-21":f"{year}-08-31"].dropna()
        rows.append({
            "year": year, "shuttle_pilot": year == 2025,
            "ratio_window": window.mean(), "ratio_later": later.mean(),
            "window_over_later": window.mean() / later.mean(),
            "n_window": len(window), "n_later": len(later),
        })
    return pd.DataFrame(rows)


# %% 8. Confronto stagione 2025 vs 2026
def load_parking_2025_2026(daily: pd.DataFrame) -> pd.DataFrame:
    """Stagioni 2025 e 2026 in formato lungo: una riga per giorno, con
    ``year`` e la data 2025 corrispondente (stesso giorno della settimana,
    2026 = 2025 + 364 giorni) in ``date_2025``.

    Le persone sono stimate qui (coefficienti di occupazione del gestore):
    \"N° persone bus\" e' trattato come persone in salita.
    """
    y26 = daily[(daily["year"] == 2026) & (daily["cars"].notna() | daily["event_day"])].copy()
    y26["date_2025"] = y26["date"] - pd.Timedelta(days=364)
    y25 = daily[daily["date"].isin(y26["date_2025"])].copy()
    y25["date_2025"] = y25["date"]
    out = pd.concat([y25, y26], ignore_index=True)
    out["cars_half"] = out["cars_half"].fillna(0)
    out["is_rainy"] = (out["rain_mm"] >= RAINY_DAY_MM).astype(int)
    out["excluded"] = out["date"].isin(EXCLUDED_DAYS)
    in_regime = out["date"].between(REGIME_2026_START, REGIME_2026_PARKING_END)
    out["occupancy"] = np.where(in_regime, OCCUPANCY_REGIME_2026, OCCUPANCY_NO_REGIME)
    out["persons_cars"] = out["cars"] * out["occupancy"]
    out["persons_cars_2_83"] = out["cars"] * PERSONS_PER_CAR  # stessa occupazione nei due anni (sensibilita')
    extra = out["bus_up"].fillna(0) + out["moto_persons"].fillna(0)
    out["persons_total"] = out["persons_cars"] + extra
    out["persons_total_2_83"] = out["persons_cars_2_83"] + extra
    cols = ["date", "year", "cars_full", "cars_half", "bus_up", "bus_down", "moto_persons", "note", "tmax",
            "rain_mm", "date_2025", "is_rainy", "cars", "excluded", "occupancy", "persons_cars",
            "persons_cars_2_83", "persons_total", "persons_total_2_83"]
    return out[cols]


def _period_2026(date_2026: pd.Timestamp) -> str:
    if date_2026 < REGIME_2026_START:
        return "1. fino al 3/7 (2026 senza regime)"
    if date_2026 <= REGIME_2026_SHUTTLE_END:
        return "2. 4/7-30/8 (2026: navetta + parcheggio a pagamento)"
    return "3. dal 31/8 (2026 senza navetta)"


def compare_2025_2026(park: pd.DataFrame) -> pd.DataFrame:
    """Medie giornaliere per periodo, stessi giorni della settimana nei due
    anni, esclusi gli eventi in ``EXCLUDED_DAYS`` (e i corrispondenti
    giorni 2025)."""
    wide = park.pivot(index="date_2025", columns="year")
    excluded = wide[("excluded", 2026)].astype(bool)
    wide = wide[~excluded]
    period = wide[("date", 2026)].map(_period_2026)
    cols = ["cars", "bus_up", "bus_down", "persons_total", "persons_total_2_83", "tmax", "is_rainy"]
    rows = []
    for name, g in wide.groupby(period):
        row = {"period": name, "n_days": len(g)}
        for c in cols:
            for y in (2025, 2026):
                row[f"{c}_{y}"] = g[(c, y)].mean()
        row["cars_change_pct"] = (row["cars_2026"] / row["cars_2025"] - 1) * 100
        rows.append(row)
    return pd.DataFrame(rows)


# %% Funzioni di supporto per i grafici
def _style_ax(ax):
    ax.grid(True, axis="y", zorder=0)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)


def plot_target_overview(panel: pd.DataFrame, path: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                    gridspec_kw={"height_ratios": [2.2, 1]})
    _style_ax(ax1)
    ax1.plot(panel["date"], panel["parking_tickets"], color=C_BLUE, linewidth=1.8, marker="o",
              markersize=3, label="Auto/giorno (biglietti parcheggio)")
    ax1.axvspan(panel["date"].min(), pd.Timestamp("2025-07-20"), color=C_GOOD_STATUS, alpha=0.07,
                 label="Navetta bus attiva")
    ax1.set_ylabel("Auto / giorno")
    ax1.set_title("Stagione Fazzon 2025: auto e copertura dei dati sui biglietti navetta",
                   color=C_INK, loc="left")
    ax1.legend(frameon=False, loc="upper left")

    _style_ax(ax2)
    valid = panel[panel["bus_tickets"].notna()]
    ax2.scatter(valid["date"], valid["bus_tickets"], color=C_GOOD_STATUS, s=28, zorder=3,
                 label="Biglietti navetta (validi, navetta attiva)")
    ax2.axvspan(panel["date"].min(), pd.Timestamp("2025-07-20"), color=C_GOOD_STATUS, alpha=0.07)
    ax2.set_ylabel("Biglietti navetta / giorno")
    ax2.set_xlabel("Data")
    ax2.legend(frameon=False, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_correlation_bars(corr: pd.DataFrame, path: Path):
    corr = corr.sort_values("within_month_r2")
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    _style_ax(ax)
    labels = [IT_LABELS.get(c, c) for c in corr["predictor"]]
    ax.barh(labels, corr["within_month_r2"], color=C_BLUE, height=0.6, zorder=3)
    for y, v in enumerate(corr["within_month_r2"]):
        ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=8, color=C_INK2)
    ax.set_xlabel("R² entro-mese vs auto/giorno (destagionalizzato)")
    ax.set_title("Quali segnali della Val di Sole seguono la domanda giornaliera a Fazzon?",
                  loc="left", color=C_INK)
    ax.set_xlim(0, corr["within_month_r2"].max() * 1.15)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_correlation_matrix(matrix: pd.DataFrame, path: Path):
    labels = [IT_LABELS.get(c, c) for c in matrix.columns]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(0.62 * n + 3, 0.62 * n + 2))
    im = ax.imshow(matrix.values, cmap=DIVERGING_CMAP, vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(n):
        for j in range(n):
            v = matrix.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=C_INK)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Coefficiente di correlazione (r di Pearson)", color=C_INK2)
    cbar.ax.tick_params(labelcolor=C_INK2)

    ax.set_title(
        "Matrice di correlazione tra target e predittori\n"
        "(nota: fine settimana vs bici/pedoni ~ 0 -> nessun effetto di weekend)",
        loc="left", color=C_INK, fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_saturation(
    panel: pd.DataFrame, demand_col: str, path: Path, mode: str = "median", min_frac: float = 0.2,
    title: str = "Auto a Fazzon vs presenze in valle: la pendenza cala ad alta domanda\n"
                 "(non contro il traffico misurato a Pellizzano: saturazione non confermata)",
):
    """``mode="median"``: soglia fissa a meta' campione (Plot 3).
    ``mode="optimal"``: soglia scelta per griglia, minimizzando l'errore
    quadratico totale delle due rette (Plot 3b, 07).
    """
    D = panel[demand_col].astype(float)
    if mode == "median":
        bp = D.median()
        lo, hi = panel[D <= bp], panel[D > bp]
        fit_lo = stats.linregress(lo[demand_col], lo["parking_tickets"])
        fit_hi = stats.linregress(hi[demand_col], hi["parking_tickets"])
        subtitle = f"soglia = mediana della domanda ({bp:,.0f})".replace(",", ".")
    elif mode == "optimal":
        valid = panel[D.notna()]
        best = _find_optimal_breakpoint(valid[demand_col], valid["parking_tickets"], min_frac=min_frac)
        bp, fit_lo, fit_hi = best["breakpoint"], best["fit_lo"], best["fit_hi"]
        lo, hi = panel[D <= bp], panel[D > bp]
        subtitle = (
            f"soglia ottimizzata ({bp:,.0f}, n={best['n_lo']}+{best['n_hi']}) - "
            f"R² con 2 rette={best['r2_two_segments']:.2f} vs R² con 1 retta={best['r2_single_line']:.2f}"
        ).replace(",", ".")
    else:
        raise ValueError(f"mode sconosciuta: {mode!r}")

    fig, ax = plt.subplots(figsize=(8.5, 6.3))
    _style_ax(ax)
    ax.scatter(panel[demand_col], panel["parking_tickets"], color=C_MUTED, s=22, alpha=0.7, zorder=2)
    xs_lo = np.array([lo[demand_col].min(), lo[demand_col].max()])
    xs_hi = np.array([hi[demand_col].min(), hi[demand_col].max()])
    ax.plot(xs_lo, fit_lo.intercept + fit_lo.slope * xs_lo, color=C_BLUE, linewidth=2.4,
             label=f"Retta a domanda bassa (pendenza={fit_lo.slope:.3f})", zorder=3)
    ax.plot(xs_hi, fit_hi.intercept + fit_hi.slope * xs_hi, color=C_ORANGE, linewidth=2.4,
             label=f"Retta a domanda alta (pendenza={fit_hi.slope:.3f}, {(fit_hi.slope/fit_lo.slope-1)*100:+.0f}%)",
             zorder=3)
    ax.axvline(bp, color=C_AXIS, linewidth=1, linestyle="--")
    ax.set_xlabel(IT_LABELS[demand_col])
    ax.set_ylabel(IT_LABELS["parking_tickets"])
    ax.set_title(
        f"{title}\n{subtitle}",
        loc="left", color=C_INK, fontsize=11,
    )
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_car_vs_bici_divergence(
    panel: pd.DataFrame, demand_col: str, path: Path, mode: str = "median", min_frac: float = 0.2,
):
    """Stesso fenomeno di 03/03b (rallentamento auto vs accelerazione
    bici/pedoni), presentato come due pannelli affiancati -- uno per canale,
    ciascuno nella propria unita' di misura -- invece di un'unica curva
    indicizzata: indicizzare a un valore iniziale vicino allo zero (come nel
    caso delle auto) gonfia artificialmente le variazioni percentuali e
    capovolge la lettura del grafico. Qui la stessa codifica colore di 03/03b
    (blu = retta a domanda bassa, arancio = retta a domanda alta) e' condivisa
    fra i due pannelli, cosi' il confronto e' "quale retta si piega, e in che
    verso" invece di un numero indice.
    """
    D = panel[demand_col].astype(float)
    channels = [
        ("parking_tickets", IT_LABELS["parking_tickets"]),
        ("bici_near_fazzon", IT_LABELS["bici_near_fazzon"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (col, ylabel) in zip(axes, channels):
        _style_ax(ax)
        if mode == "median":
            bp = D.median()
            lo, hi = panel[D <= bp], panel[D > bp]
            fit_lo = stats.linregress(lo[demand_col], lo[col])
            fit_hi = stats.linregress(hi[demand_col], hi[col])
        else:
            best = _find_optimal_breakpoint(D, panel[col], min_frac=min_frac)
            bp, fit_lo, fit_hi = best["breakpoint"], best["fit_lo"], best["fit_hi"]

        ax.scatter(panel[demand_col], panel[col], color=C_MUTED, s=20, alpha=0.7, zorder=2)
        xs_lo = np.array([D.min(), bp])
        xs_hi = np.array([bp, D.max()])
        ax.plot(xs_lo, fit_lo.intercept + fit_lo.slope * xs_lo, color=C_BLUE, linewidth=2.4, zorder=3)
        ax.plot(xs_hi, fit_hi.intercept + fit_hi.slope * xs_hi, color=C_ORANGE, linewidth=2.4, zorder=3)
        ax.axvline(bp, color=C_AXIS, linewidth=1, linestyle="--")

        pct = (fit_hi.slope - fit_lo.slope) / fit_lo.slope * 100
        ax.set_title(
            f"{ylabel}\npendenza: {fit_lo.slope:.3g} -> {fit_hi.slope:.3g}  ({pct:+.0f}%),  soglia={bp:,.0f}".replace(",", "."),
            loc="left", color=C_INK, fontsize=10,
        )
        ax.set_xlabel(IT_LABELS[demand_col], fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=9.5)

    handles = [
        Line2D([0], [0], color=C_BLUE, linewidth=2.4, label="Retta a domanda bassa"),
        Line2D([0], [0], color=C_ORANGE, linewidth=2.4, label="Retta a domanda alta"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.94))

    subtitle = (
        "soglia = mediana della domanda (uguale per entrambi i canali)" if mode == "median"
        else "soglia ottimizzata per canale (minimizza l'errore totale delle due rette)"
    )
    fig.suptitle(
        f"Rispetto alle presenze Vodafone le auto rallentano, bici/pedoni no\n"
        f"(rispetto al traffico misurato a Pellizzano le auto non rallentano)\n{subtitle}",
        x=0.02, ha="left", fontsize=12, color=C_INK, y=1.03,
    )
    fig.subplots_adjust(top=0.72, wspace=0.28)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_shuttle_experiment(panel: pd.DataFrame, demand_col: str, result: dict, path: Path):
    lo_overlap, hi_overlap = result["overlap_demand_range"]
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    _style_ax(ax)

    shuttle = panel[panel["shuttle_running"]]
    no_shuttle = panel[~panel["shuttle_running"]]
    ax.scatter(shuttle[demand_col], shuttle["parking_tickets"], color=C_BLUE, s=32, zorder=3,
                label="Navetta attiva (28 giu - 20 lug)")
    ax.scatter(no_shuttle[demand_col], no_shuttle["parking_tickets"], color=C_ORANGE, s=32, zorder=3,
                label="Navetta non attiva (resto della stagione)")
    ax.axvspan(lo_overlap, hi_overlap, color=C_MUTED, alpha=0.08, zorder=1,
                label="Intervallo di domanda in cui i due periodi si sovrappongono")

    ax.set_xlabel(IT_LABELS[demand_col])
    ax.set_ylabel(IT_LABELS["parking_tickets"])
    ax.set_title(f"Giorni con/senza navetta a parità di domanda\n(domanda: {IT_LABELS[demand_col]})",
                 loc="left", color=C_INK, fontsize=11)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)

    coef_txt = (
        f"Regressione: auto ~ domanda + temp.max + pioggia + navetta_attiva\n"
        f"coefficiente navetta_attiva = {result['shuttle_running_coef']:+.1f} auto/giorno "
        f"(p={result['shuttle_running_p']:.2f}, n={result['n_obs_regression']})\n"
        f"Non significativo -- coerente con 'nessuna differenza rilevabile nel numero di auto',\n"
        f"ma il periodo è confuso con il tempo solare (poche osservazioni, identificazione debole)."
    )
    ax.text(0.02, -0.20, coef_txt, transform=ax.transAxes, fontsize=8, color=C_INK2, va="top")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_bus_coverage_gap(panel: pd.DataFrame, path: Path):
    """Il grafico dell'"assenza di prove": mostra che i giorni senza dati
    navetta validi sono esattamente i giorni di domanda più alta (più
    "saturi") della stagione.
    """
    p = panel.sort_values("date").copy()
    saturated_threshold = p["parking_tickets"].quantile(0.75)
    p["saturated_day"] = p["parking_tickets"] >= saturated_threshold

    fig, ax = plt.subplots(figsize=(11, 5.5))
    _style_ax(ax)
    ax.fill_between(p["date"], p["parking_tickets"], color=C_BLUE, alpha=0.15, zorder=1)
    ax.plot(p["date"], p["parking_tickets"], color=C_BLUE, linewidth=1.6, zorder=2, label="Auto / giorno")

    has_bus = p[p["bus_tickets"].notna()]
    no_bus = p[p["bus_tickets"].isna()]
    ax.scatter(has_bus["date"], has_bus["parking_tickets"], color=C_GOOD_STATUS, s=30, zorder=4,
                label="Dati navetta Fazzon validi in questo giorno")
    sat_no_bus = no_bus[no_bus["saturated_day"]]
    other_no_bus = no_bus[~no_bus["saturated_day"]]
    ax.scatter(sat_no_bus["date"], sat_no_bus["parking_tickets"], color=C_RED_STATUS, s=42, marker="X", zorder=5,
                label="Nessun dato navetta E domanda nel quartile superiore (giorno \"saturo\")")
    ax.scatter(other_no_bus["date"], other_no_bus["parking_tickets"], color=C_MUTED, s=16, zorder=3,
                label="Nessun dato navetta, domanda più bassa")

    ax.axhline(saturated_threshold, color=C_AXIS, linewidth=1, linestyle="--")
    ax.text(p["date"].max(), saturated_threshold, " soglia quartile superiore", va="bottom", ha="right",
             fontsize=8, color=C_INK2)
    ax.set_ylabel(IT_LABELS["parking_tickets"])
    ax.set_title(
        "Navetta pilota 2025: nessun dato per i giorni di domanda più alta della stagione",
        loc="left", color=C_INK,
    )
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return saturated_threshold


def plot_shuttle_road_did(did: pd.DataFrame, path: Path):
    """Rapporto finestra navetta / periodo successivo, per anno: barra 2025
    (anno della navetta) evidenziata, 2023/2024 come riferimento."""
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    _style_ax(ax)
    colors = [C_BLUE if pilot else C_MUTED for pilot in did["shuttle_pilot"]]
    x = np.arange(len(did))
    ax.bar(x, did["window_over_later"], color=colors, width=0.55, zorder=3)
    for xi, v in zip(x, did["window_over_later"]):
        ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=C_INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{y}\n(navetta)" if p else str(y) for y, p in zip(did["year"], did["shuttle_pilot"])])
    lo = did["window_over_later"].min()
    ax.set_ylim(lo - 0.08, did["window_over_later"].max() + 0.04)
    ax.set_ylabel("Rapporto 28 giu-20 lug / 21 lug-31 ago")
    ax.set_title(
        "Auto verso monte a Pellizzano (7-12), rispetto alle valli laterali di controllo:\n"
        "nel 2025, con la navetta, la finestra 28/6-20/7 non è più bassa degli anni precedenti",
        loc="left", color=C_INK, fontsize=10.5,
    )
    note = (
        "Rapporto = auto SS42 Pellizzano verso monte 7-12 / auto totali a Pracorno + Celledizzo + Revò,\n"
        "media della finestra navetta diviso media del periodo successivo. Un effetto navetta abbasserebbe la barra 2025.\n"
        "Le auto dirette a Fazzon sono ~10-14% del flusso a Pellizzano: un calo di poche decine di auto/giorno (~2%)\n"
        "e' dello stesso ordine della variabilita' fra un anno e l'altro, quindi non sarebbe distinguibile."
    )
    ax.text(0.0, -0.17, note, transform=ax.transAxes, fontsize=7.8, color=C_INK2, va="top")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_road_data_coverage(road_daily: pd.DataFrame, panel: pd.DataFrame, path: Path):
    """Copertura dei sensori di traffico: giorni validi per stazione e
    direzione, 2022-2025, con la stagione Fazzon 2025 evidenziata."""
    grid = road_daily.pivot_table(index=["station", "direction"], columns="date", values="valid", aggfunc="first")
    all_days = pd.date_range(grid.columns.min(), grid.columns.max(), freq="D")
    grid = grid.reindex(columns=all_days).fillna(False).astype(float)
    labels = [f"{s.capitalize()} dir.{d}" for s, d in grid.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = LinearSegmentedColormap.from_list("valid", [C_RED_STATUS, C_BLUE])
    ax.imshow(grid.values, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
              extent=(0, len(all_days), len(labels), 0))
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_yticklabels(labels, fontsize=8)
    year_starts = [i for i, d in enumerate(all_days) if d.month == 1 and d.day == 1]
    ax.set_xticks(year_starts)
    ax.set_xticklabels([all_days[i].year for i in year_starts])
    ax.grid(False)
    s0 = (panel["date"].min() - all_days[0]).days
    s1 = (panel["date"].max() - all_days[0]).days + 1
    ax.axvline(s0, color=C_INK, linewidth=1)
    ax.axvline(s1, color=C_INK, linewidth=1)
    ax.text((s0 + s1) / 2, -0.3, "stagione Fazzon 2025", ha="center", va="bottom", fontsize=8, color=C_INK)
    handles = [
        Line2D([0], [0], marker="s", linestyle="", color=C_BLUE, markersize=9, label="Dato valido"),
        Line2D([0], [0], marker="s", linestyle="", color=C_RED_STATUS, markersize=9,
               label="Mancante, ore incomplete o sensore a zero"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.06), ncol=2, fontsize=8.5)
    ax.set_title(
        "Copertura dei contatori di traffico PAT: Pellizzano completo dal 2023;\n"
        "Dimaro spento da metà 2024, Tonale nel 2022, Croviana dir.2 quasi sempre a zero",
        loc="left", color=C_INK, fontsize=10.5, pad=16,
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_sensor_map(stations: pd.DataFrame, path: Path):
    """Mappa schematica (senza fondo cartografico) di stazioni di traffico,
    contatori bici/pedoni e Fazzon, colorati per ruolo nell'analisi."""
    road = stations[stations["kind"] == "road"]
    bike = stations[stations["kind"] == "bike"].copy()
    bike["label"] = bike["code"].map(BIKE_SENSOR_SHORT_NAMES)
    fazzon = stations[stations["kind"] == "poi"].iloc[0]

    role = {
        "pellizzano": ("Predittore principale (auto verso monte 7-12)", C_BLUE),
        "pracorno": ("Stazione di controllo", C_AQUA),
        "celledizzo": ("Stazione di controllo", C_AQUA),
        "revo": ("Stazione di controllo", C_AQUA),
        "tonale": ("Predittore", C_VIOLET),
    }
    unused = ("Non usata (sensore guasto o fuori area)", C_MUTED)

    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.grid(False)
    for station, lat, lon in road[["name", "lat", "lon"]].itertuples(index=False):
        _, color = role.get(station, unused)
        ax.scatter(lon, lat, s=90, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=4)
        ax.annotate(station.capitalize(), (lon, lat), xytext=(6, 6), textcoords="offset points",
                    fontsize=8.5, color=C_INK)
    near = bike["role"].eq("near_fazzon")
    ax.scatter(bike["lon"], bike["lat"], marker="^", s=70,
               color=np.where(near, C_ORANGE, C_SURFACE_ORANGE), edgecolor=C_ORANGE, linewidth=1.2, zorder=3)
    for _, r in bike.iterrows():
        ax.annotate(r["label"], (r["lon"], r["lat"]), xytext=(0, -13), textcoords="offset points",
                    ha="center", fontsize=7.5, color=C_INK2)
    ax.scatter(fazzon["lon"], fazzon["lat"], marker="*", s=260, color=C_RED_STATUS, edgecolor=SURFACE, zorder=5)
    ax.annotate("Fazzon / Lago dei Caprioli", (fazzon["lon"], fazzon["lat"]), xytext=(8, -4), textcoords="offset points",
                fontsize=9, color=C_RED_STATUS, fontweight="bold")

    ax.set_aspect(1 / np.cos(np.deg2rad(46.33)))
    ax.set_xlabel("Longitudine")
    ax.set_ylabel("Latitudine")
    handles = [Line2D([0], [0], marker="o", linestyle="", color=c, markersize=8, label=l)
               for l, c in dict(list(role.values()) + [unused]).items()]
    handles += [
        Line2D([0], [0], marker="^", linestyle="", color=C_ORANGE, markersize=8, label="Contatore bici/pedoni vicino a Fazzon"),
        Line2D([0], [0], marker="^", linestyle="", markerfacecolor=C_SURFACE_ORANGE, markeredgecolor=C_ORANGE,
               markersize=8, label="Altro contatore bici/pedoni"),
        Line2D([0], [0], marker="*", linestyle="", color=C_RED_STATUS, markersize=12, label="Fazzon (posizione indicativa)"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8.5)
    ax.set_title("Dove sono i sensori: stazioni di traffico PAT e contatori bici/pedoni in Val di Sole",
                 loc="left", color=C_INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_2025_vs_2026_daily(park: pd.DataFrame, path: Path):
    """Tre pannelli con asse x comune (data 2026; il 2025 e' allineato per
    giorno della settimana): auto, passeggeri navetta, persone totali."""
    wide = park.pivot(index="date_2025", columns="year")
    x = wide[("date", 2026)]
    excluded = wide[("excluded", 2026)].astype(bool)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1, 1.2]})
    for ax in axes:
        _style_ax(ax)
        ax.axvspan(REGIME_2026_START, REGIME_2026_SHUTTLE_END, color=C_GOOD_STATUS, alpha=0.06, zorder=0)
        for d in EXCLUDED_DAYS:
            ax.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12), color=C_MUTED, alpha=0.18, zorder=0)

    def line(ax, col, year, color, label, **kw):
        ax.plot(x, wide[(col, year)].where(~excluded), color=color, linewidth=1.8, label=label, **kw)

    ax = axes[0]
    line(ax, "cars", 2025, C_MUTED, "2025 (5 euro, fino a esaurimento posti)")
    line(ax, "cars", 2026, C_BLUE, "2026 (giornata intera + pomeriggio)")
    ax.set_ylabel("Auto / giorno")
    ax.set_title("Auto al parcheggio di Fazzon", loc="left", color=C_INK, fontsize=10.5)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)

    ax = axes[1]
    line(ax, "bus_up", 2025, C_MUTED, "2025 Greenbus pilota (persone in salita)")
    line(ax, "bus_up", 2026, C_BLUE, "2026 Lakebus, persone in salita")
    line(ax, "bus_down", 2026, C_ORANGE, "2026 Lakebus, persone in discesa (dal 1/8)")
    ax.set_ylabel("Persone / giorno")
    ax.set_title("Passeggeri navetta", loc="left", color=C_INK, fontsize=10.5)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)

    ax = axes[2]
    line(ax, "persons_total", 2025, C_MUTED, "2025: auto x 2,5 + bus")
    line(ax, "persons_total", 2026, C_BLUE, "2026: auto x 3,5 (x 2,5 fuori regime) + bus in salita + moto")
    line(ax, "persons_total_2_83", 2026, C_BLUE, "2026 con 2,83 persone/auto (sensibilità)", linestyle="--", alpha=0.7)
    ax.set_ylabel("Persone / giorno")
    ax.set_title("Persone stimate (coefficienti di occupazione del gestore, da validare)",
                 loc="left", color=C_INK, fontsize=10.5)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.set_xlabel("Data 2026 (2025 allineato per giorno della settimana)")

    fig.suptitle(
        "Fazzon 2025 vs 2026: con navetta e parcheggio a pagamento le auto calano di circa 2/3,\n"
        "le persone stimate restano simili (fascia verde: regime 2026; grigio: giorni esclusi)",
        x=0.01, ha="left", fontsize=12, color=C_INK,
    )
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_2025_vs_2026_periods(comp: pd.DataFrame, path: Path):
    """Medie per periodo: auto e persone stimate, 2025 vs 2026 affiancati."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    labels = [p.split(" (")[0].replace(". ", ".\n") for p in comp["period"]]
    xpos = np.arange(len(comp))
    w = 0.38
    for ax, col, title in [
        (axes[0], "cars", "Auto / giorno"),
        (axes[1], "persons_total", "Persone stimate / giorno"),
    ]:
        _style_ax(ax)
        for off, year, color in [(-w / 2, 2025, C_MUTED), (w / 2, 2026, C_BLUE)]:
            vals = comp[f"{col}_{year}"]
            ax.bar(xpos + off, vals, width=w, color=color, label=str(year), zorder=3)
            for xi, v in zip(xpos + off, vals):
                ax.text(xi, v + vals.max() * 0.01, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5, color=C_INK2)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, loc="left", color=C_INK, fontsize=10.5)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle(
        "Medie per periodo, stessi giorni della settimana\n"
        "1 = fino al 3/7;  2 = 4/7-30/8, regime 2026 (navetta + parcheggio a pagamento);  3 = dal 31/8, senza navetta",
        x=0.01, ha="left", fontsize=11, color=C_INK,
    )
    note = ("Persone: 2025 = auto x 2,5 + bus; 2026 = auto x 3,5 nel regime (x 2,5 fuori) + bus in salita + moto "
            "(coefficienti del gestore, da validare).\nAuto 2026 non conteggiate: ingressi dopo le 17 e parcheggi privati.")
    fig.text(0.01, 0.01, note, fontsize=8, color=C_INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# %% 9. Dataset del modello: livelli per stagione e meteo, spostamento modale, prezzo
def _estimate(estimates: pd.DataFrame, parameter: str, days: str = "") -> pd.Series:
    row = estimates[(estimates["parameter"] == parameter) & (estimates["days"] == days)]
    return row.iloc[0]


def plot_demand_drivers(panel: pd.DataFrame, daily: pd.DataFrame, presence_stats: pd.DataFrame, path: Path):
    """Sinistra: auto a Fazzon vs auto verso monte al mattino a Pellizzano (A17), colorate per
    classe meteo (A19). Destra: auto per stagione e classe meteo, i livelli usati dal modello
    (A21) con le medie osservate."""
    weather = daily.set_index("date")["weather"]
    p = panel.dropna(subset=["strada_pellizzano_salita_mattina"]).copy()
    p["weather"] = p["date"].map(weather)
    x, y = p["strada_pellizzano_salita_mattina"], p["parking_tickets"]
    fit = stats.linregress(x, y)

    def within_month(s):
        return s - s.groupby(p["month"]).transform("mean")

    r2_wm = within_month(x).corr(within_month(y)) ** 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.1, 1]})
    _style_ax(ax1)
    for w, color in WEATHER_COLORS.items():
        d = p[p["weather"] == w]
        ax1.scatter(d["strada_pellizzano_salita_mattina"], d["parking_tickets"], s=34, color=color, alpha=0.85,
                    zorder=3, label=f"{WEATHER_LABELS[w]} ({len(d)} giorni)")
    xs = np.array([x.min(), x.max()])
    ax1.plot(xs, fit.intercept + fit.slope * xs, color=C_INK2, linewidth=1.6, zorder=2,
             label=f"Retta: R² = {fit.rvalue ** 2:.2f} (entro il mese: {r2_wm:.2f})")
    ax1.set_xlabel(IT_LABELS["strada_pellizzano_salita_mattina"])
    ax1.set_ylabel(IT_LABELS["parking_tickets"])
    ax1.set_title("Il traffico in valle al mattino spiega gran parte delle auto a Fazzon",
                  loc="left", color=C_INK, fontsize=10.5)
    ax1.legend(frameon=False, loc="upper left", fontsize=9.5)

    _style_ax(ax2)
    s = presence_stats.copy()
    s["cars"] = s["mean_car_visitors"] / REFERENCE_OCCUPANCY
    observed = (daily[(daily["year"] == 2025) & daily["cars"].notna() & daily["weather"].notna()]
                .assign(season=lambda d: d["month"].map(SEASONS))
                .groupby(["season", "weather"])["cars"].agg(["mean", "size"]))
    seasons = list(SEASON_LABELS)
    xpos = np.arange(len(seasons))
    width = 0.26
    for i, (w, color) in enumerate(WEATHER_COLORS.items()):
        off = (i - 1) * width
        level = s.set_index(["season", "weather"]).loc[[(se, w) for se in seasons], "cars"].to_numpy()
        factor = s.loc[s["weather"] == w, "weather_factor"].iloc[0]
        label = WEATHER_LABELS[w] + ("" if w == "good" else f" (x {factor:.2f})".replace(".", ","))
        ax2.bar(xpos + off, level, width=width, color=color, alpha=0.85, zorder=3, label=label)
        for xi, se in zip(xpos + off, seasons):
            if (se, w) in observed.index:
                m, n = observed.loc[(se, w)]
                ax2.scatter(xi, m, marker="_", s=260, color=C_INK, linewidth=2, zorder=4)
                ax2.text(xi, -12, f"{int(n)}", ha="center", va="top", fontsize=8, color=C_INK2)
            else:
                ax2.text(xi, -12, "0", ha="center", va="top", fontsize=8, color=C_INK2)
    ax2.scatter([], [], marker="_", s=260, color=C_INK, linewidth=2, label="Media osservata 2025")
    ax2.set_xticks(xpos)
    ax2.set_xticklabels([SEASON_LABELS[se] for se in seasons])
    ax2.tick_params(axis="x", pad=16)
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel("Auto / giorno (livello del modello)")
    ax2.set_title("Il meteo: stesso fattore in ogni stagione (A21)", loc="left", color=C_INK, fontsize=10.5)
    ax2.legend(frameon=False, loc="upper left", fontsize=9.5)

    fig.suptitle(f"Cosa muove le auto a Fazzon (stagione 2025, {len(p)} giorni)", x=0.01, ha="left", fontsize=12,
                 color=C_INK)
    note = ("Meteo (A19): pioggia = almeno 5 mm; fresco = senza pioggia e temperatura massima di oltre 2 °C sotto "
            "la media climatologica del mese; bel tempo = il resto. Stazione Meteotrentino di Malè.\n"
            "Destra: barre = livelli del modello (livello della stagione x fattore meteo); trattini = media dei giorni "
            "osservati; numeri sotto le barre = giorni osservati.")
    fig.text(0.01, -0.04, note, fontsize=8.5, color=C_INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_modal_shift_split(estimates: pd.DataFrame, days: pd.DataFrame, path: Path):
    """Visitori in meno in auto/moto al giorno (regime 2026 vs condizioni 2025) e loro ripartizione
    fra navetta, salita a piedi e altrove, con gli intervalli bootstrap al 90 %."""
    regime = days[days["role"] == "regime"]
    n_all = len(regime)
    n_walk = int(regime["bus_down"].notna().sum())
    first_walk = regime.loc[regime["bus_down"].notna(), "date"].min()
    lost = _estimate(estimates, "lost_per_day", "with_walk_up")
    lost_all = _estimate(estimates, "lost_per_day", "all_days")
    shares = [
        ("Navetta\n(persone in salita)", "share_shuttle", C_BLUE),
        ("A piedi\n(scendono in navetta, non salgono)", "share_walk_up", C_AQUA),
        ("Altrove\n(il resto)", "share_elsewhere", C_MUTED),
    ]

    fig, ax = plt.subplots(figsize=(12, 6.2))
    _style_ax(ax)
    xpos = np.arange(len(shares))
    for xi, (label, parameter, color) in zip(xpos, shares):
        e = _estimate(estimates, parameter, "with_walk_up")
        ax.bar(xi, e["estimate"] * 100, width=0.55, color=color, zorder=3)
        ax.errorbar(xi, e["estimate"] * 100, yerr=[[(e["estimate"] - e["low"]) * 100], [(e["high"] - e["estimate"]) * 100]],
                    color=C_INK, capsize=8, linewidth=1.4, zorder=4)
        ax.text(xi + 0.3, e["estimate"] * 100, f"{e['estimate'] * 100:.0f}%\n({e['low'] * 100:.0f}% / {e['high'] * 100:.0f}%)",
                va="center", ha="left", fontsize=9.5, color=C_INK2)
    ax.axhline(0, color=C_AXIS, linewidth=1)
    ax.set_xticks(xpos)
    ax.set_xticklabels([s[0] for s in shares])
    ax.set_xlim(-0.5, len(shares) - 0.2)
    ax.set_ylabel("Quota dei visitori in meno in auto/moto (%)")
    ax.set_title(
        f"Regime 2026: circa {lost['estimate']:.0f} visitori al giorno in meno arrivano in auto o moto "
        f"(intervallo 90%: {lost['low']:.0f}-{lost['high']:.0f}).\n"
        f"Tre su quattro salgono in navetta; la quota che va altrove non è distinguibile da zero",
        loc="left", color=C_INK, fontsize=11,
    )
    note = (
        f"Giorni dal {first_walk:%d/%m/%Y} (persone in discesa registrate): {n_walk}; su tutti i {n_all} giorni del "
        f"regime: {lost_all['estimate']:.0f} visitori in meno al giorno, navetta "
        f"{_estimate(estimates, 'share_shuttle', 'all_days')['estimate'] * 100:.0f}%.\n"
        "Visitori in meno = visitori attesi in condizioni 2025 (relazione 2025 auto-traffico-meteo applicata al traffico "
        "del giorno, A37; effetto anno 1 ± 10%, A22;\n2,5 persone/veicolo) - visitori in auto e moto misurati "
        "(persone/veicolo dal prezzo pagato, A2). "
        "A piedi = persone in discesa - persone in salita in navetta (A6).\n"
        "Altrove = il resto, per differenza. Barre di errore: intervallo bootstrap al 90% sui giorni."
    )
    ax.text(0.0, -0.2, note, transform=ax.transAxes, fontsize=8.5, color=C_INK2, va="top")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_price_response(estimates: pd.DataFrame, path: Path):
    """Visitori e veicoli privati in funzione del prezzo per veicolo, relativi a 5 euro: la
    curva esponenziale del modello (A23), con la banda di beta, e i due livelli osservati."""
    beta = _estimate(estimates, "beta_per_euro")
    price_2026 = _estimate(estimates, "price_paid")
    visitors = _estimate(estimates, "visitor_response")
    cars = _estimate(estimates, "car_response")

    prices = np.linspace(0, 25, 251)

    def response(b):
        return np.exp(-b * np.clip(prices - PRICE_2025, 0, None))  # A27: nessun aumento sotto 5 euro

    vehicles_factor = occupancy(PRICE_2025) / occupancy(prices)
    observed = prices <= price_2026["estimate"]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    _style_ax(ax)
    for values, low, high, color, label in [
        (response(beta["estimate"]), response(beta["high"]), response(beta["low"]), C_BLUE,
         "Visitori in auto/moto"),
        (response(beta["estimate"]) * vehicles_factor, response(beta["high"]) * vehicles_factor,
         response(beta["low"]) * vehicles_factor, C_ORANGE, "Veicoli (con più persone per veicolo, A2)"),
    ]:
        ax.fill_between(prices, low * 100, high * 100, color=color, alpha=0.13, zorder=1)
        ax.plot(prices[observed], values[observed] * 100, color=color, linewidth=2.4, zorder=3, label=label)
        ax.plot(prices[~observed], values[~observed] * 100, color=color, linewidth=2.4, linestyle="--", zorder=3)

    ax.scatter([PRICE_2025], [100], s=70, color=C_INK, zorder=5)
    ax.annotate("2025: 5 €\n(riferimento)", (PRICE_2025, 100), xytext=(8, 6), textcoords="offset points",
                fontsize=9.5, color=C_INK)
    for e, color, name in [(visitors, C_BLUE, "visitori"), (cars, C_ORANGE, "veicoli")]:
        ax.errorbar(price_2026["estimate"], e["estimate"] * 100,
                    yerr=[[(e["estimate"] - e["low"]) * 100], [(e["high"] - e["estimate"]) * 100]],
                    fmt="o", color=color, markersize=8, capsize=6, zorder=5)
        ax.annotate(f"2026 ({name}): {e['estimate'] * 100:.0f}%", (price_2026["estimate"], e["estimate"] * 100),
                    xytext=(-12, -4), textcoords="offset points", ha="right", fontsize=9.5, color=color)
    ax.annotate(f"prezzo medio pagato 2026: {price_2026['estimate']:.1f} €\n(20 € giornata, 10 € pomeriggio, A24)",
                (price_2026["estimate"], 4), ha="center", fontsize=8.5, color=C_INK2)
    ax.axvline(price_2026["estimate"], color=C_AXIS, linewidth=1, linestyle=":")
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 115)
    ax.set_xlabel("Prezzo pagato per veicolo (€)")
    ax.set_ylabel("Rispetto al 2025 a 5 € (%)")
    ax.set_title(
        f"Risposta al prezzo del parcheggio: −{beta['estimate'] * 100:.0f}% di visitori in auto/moto per ogni euro "
        f"in più (circa; β = {beta['estimate']:.3f}, intervallo 90%: {beta['low']:.3f}-{beta['high']:.3f})",
        loc="left", color=C_INK, fontsize=11,
    )
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    caveat = (
        "ATTENZIONE: la curva passa per due soli livelli di prezzo osservati (5 € nel 2025, circa 15 € pagati in media "
        "nel 2026). La forma esponenziale è un'ipotesi (A23), non è\nstimata dai dati: fra i due punti è "
        "un'interpolazione, oltre i 15 € (tratteggio) un'estrapolazione. Tutto l'effetto del pacchetto 2026 (prezzo, "
        "prenotazione, semaforo, navetta) è\nattribuito al prezzo (A25): un cambio di prezzo senza il resto del "
        "pacchetto è un'estrapolazione. Sotto i 5 € si assume nessun aumento (A27). Fasce: intervallo 90% di β."
    )
    ax.text(0.0, -0.13, caveat, transform=ax.transAxes, fontsize=8.8, color=C_INK, va="top",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#fbe1d6", "edgecolor": C_ORANGE, "linewidth": 0.8})
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# %% 10. Cosa si misura in ogni anno; riferimento in condizioni 2025 (A37) ed effetto anno (A22)
# Condizione di accesso del giorno (stagione giugno-settembre): etichetta e colore (stesso ordine nei grafici).
CONDITIONS = {
    "free": ("5 €, senza navetta", C_BLUE),
    "free_shuttle": ("5 €, con navetta (2025: Greenbus pilota)", C_AQUA),
    "regime": ("20 € + prenotazione e semaforo, con navetta (2026: Lakebus)", C_ORANGE),
    "regime_no_shuttle": ("20 € + prenotazione e semaforo, senza navetta", C_VIOLET),
    "excluded": ("Escluso (evento, navetta ferma per maltempo, regole incerte il 31/8)", C_MUTED),
}
SHUTTLE_NAMES = {"greenbus_pilot": "Greenbus pilota", "lakebus": "Lakebus"}
N_BOOT = 2000
BOOT_SEED = 0


def measurement_days(daily: pd.DataFrame) -> pd.DataFrame:
    """Giorni della stagione degli anni con dati Fazzon, con la condizione di accesso e cosa e'
    misurato (auto, traffico valido a Pellizzano, persone in salita / in discesa in navetta)."""
    years = sorted(daily.loc[daily["cars"].notna(), "year"].unique())
    s = daily[daily["month"].isin(list(SEASONS)) & daily["year"].isin(years)].copy()
    s["condition"] = np.select(
        [s["cars"].isna(), s["event_day"] | s["date"].isin(EXCLUDED_DAYS),
         s["regulated_parking"] & s["shuttle"].notna(), s["regulated_parking"], s["shuttle"].notna()],
        ["no_data", "excluded", "regime", "regime_no_shuttle", "free_shuttle"], "free")
    s["has_traffic"] = s[DEMAND_PREDICTOR].notna()
    s["has_bus_up"] = s["bus_up"].notna() | s["greenbus_tickets"].notna()
    s["has_bus_down"] = s["bus_down"].notna()
    return s[["date", "year", "condition", "shuttle", "has_traffic", "has_bus_up", "has_bus_down"]]


def measurement_space(days: pd.DataFrame) -> pd.DataFrame:
    """Giorni per anno e condizione, con quanti hanno traffico valido e persone in discesa."""
    return (days.groupby(["year", "condition"])
            .agg(days=("date", "size"), with_traffic=("has_traffic", "sum"), with_bus_down=("has_bus_down", "sum"),
                 shuttle=("shuttle", lambda s: ", ".join(sorted(s.dropna().unique()))))
            .reset_index())


def _demand_forms() -> dict:
    """Forme della relazione auto-traffico-meteo confrontate (A37): nome -> (design, log)."""
    def weather(x):
        return [(x["weather"] == w).to_numpy(dtype=float) for w in WEATHER_CLASSES[1:]]

    def m(x):
        return x[DEMAND_PREDICTOR].to_numpy(dtype=float)

    def months(x):
        return [(x["month"] == mo).to_numpy(dtype=float) for mo in sorted(SEASONS)[1:]]

    one = lambda x: np.ones(len(x))  # noqa: E731
    return {
        "Traffico + meteo (scelta, A37)": (lambda x: np.column_stack([one(x), m(x), *weather(x)]), False),
        "Traffico + meteo + mese": (lambda x: np.column_stack([one(x), m(x), *weather(x), *months(x)]), False),
        "Proporzionale al traffico, per meteo": (lambda x: np.column_stack([m(x), *[w * m(x) for w in weather(x)]]), False),
        "Log-log + meteo": (lambda x: np.column_stack([one(x), np.log(m(x)), *weather(x)]), True),
    }


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, design, log: bool) -> np.ndarray:
    y = np.log(train["cars"].to_numpy()) if log else train["cars"].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(design(train), y, rcond=None)
    if not log:
        return design(test) @ coef
    smearing = np.mean(np.exp(y - design(train) @ coef))
    return np.exp(design(test) @ coef) * smearing


def reference_cross_validation(daily: pd.DataFrame, presence_stats: pd.DataFrame) -> pd.DataFrame:
    """Errore medio assoluto (auto/giorno) del riferimento a livelli stagione x meteo (A21) e delle
    forme della relazione con il traffico (A37): ``mae_2025`` fuori campione sul 2025 (si lascia
    fuori una settimana alla volta; i livelli sono stimati su tutto il 2025, quindi in campione) e
    ``mae_2026`` sui giorni 2026 a 5 euro senza regole (stima sul 2025: un vero test)."""
    cal = calibration_days(daily)
    cal = cal[cal["cars"] > 0].copy()
    week = cal["date"].dt.isocalendar().week
    test_2026 = daily[(daily["year"] == 2026) & daily["month"].isin(list(SEASONS)) & daily["cars"].notna()
                      & daily[DEMAND_PREDICTOR].notna() & daily["weather"].notna() & ~daily["regulated_parking"]
                      & ~daily["event_day"]]
    levels = presence_stats.set_index(["season", "weather"])["mean_car_visitors"] / REFERENCE_OCCUPANCY

    def level(x):
        return np.array([levels[(SEASONS[mo], w)] for mo, w in zip(x["month"], x["weather"])])

    rows = [{"reference": "Livelli stagione x meteo (A21, prima)", "mae_2025": np.abs(level(cal) - cal["cars"]).mean(),
             "mae_2026": np.abs(level(test_2026) - test_2026["cars"]).mean()}]
    for name, (design, log) in _demand_forms().items():
        pred = pd.Series(np.nan, index=cal.index)
        for w in week.unique():
            test = week == w
            pred[test] = _fit_predict(cal[~test], cal[test], design, log)
        rows.append({"reference": name, "mae_2025": (pred - cal["cars"]).abs().mean(),
                     "mae_2026": np.abs(_fit_predict(cal, test_2026, design, log) - test_2026["cars"]).mean()})
    out = pd.DataFrame(rows)
    out["n_2025"], out["n_2026"] = len(cal), len(test_2026)
    return out


def year_effect_checks(daily: pd.DataFrame, modal_shift: pd.DataFrame, monitoring: pd.DataFrame) -> pd.DataFrame:
    """Le misure dell'effetto anno 2026 (A22) sui giorni a 5 euro senza regole, con intervallo
    bootstrap al 90 %: rapporto grezzo auto / auto del mattino, relazione A37 (sui giorni 2026 e
    rispetto agli stessi giorni 2025), regressione log con mese e meteo su tutta la stagione."""
    rng = np.random.default_rng(BOOT_SEED)
    coef = fit_demand(calibration_days(daily))
    mon = monitoring[(monitoring["year"] == 2026) & (monitoring["period"] == "rest_of_season")].set_index("indicator")
    s = daily[daily["month"].isin(list(SEASONS)) & daily["cars"].notna() & daily[DEMAND_PREDICTOR].notna()
              & daily["weather"].notna() & ~daily["date"].isin(EXCLUDED_DAYS)].copy()
    out_2026 = s[(s["year"] == 2026) & ~s["regulated_parking"]]
    window = s["date"].dt.month * 100 + s["date"].dt.day
    start, end = REGULATED_PARKING[2026]
    out_2025 = s[(s["year"] == 2025) & ~window.between(start.month * 100 + start.day, end.month * 100 + end.day)]

    def boot(stat):
        values = [stat(out_2026.iloc[rng.integers(0, len(out_2026), len(out_2026))],
                       out_2025.iloc[rng.integers(0, len(out_2025), len(out_2025))]) for _ in range(N_BOOT)]
        return np.quantile(values, [0.05, 0.95])

    def raw(a, b):
        return (a["cars"].sum() / a[DEMAND_PREDICTOR].sum()) / (b["cars"].sum() / b[DEMAND_PREDICTOR].sum())

    def relation(a, b):
        return (a["cars"].sum() / expected_cars(a, coef).sum()) / (b["cars"].sum() / expected_cars(b, coef).sum())

    observed = modal_shift.set_index("parameter").loc["year_effect_observed"]
    rows = [
        {"check": "Auto per auto del mattino, 2026 / 2025", "estimate": mon.loc[CARS_PER_MORNING_CAR, "ratio_to_reference"],
         **dict(zip(["low", "high"], boot(raw)))},
        {"check": "Relazione A37 sui giorni 2026", "estimate": observed["estimate"],
         "low": observed["low"], "high": observed["high"]},
        {"check": "Relazione A37, rispetto agli stessi giorni 2025", "estimate": relation(out_2026, out_2025),
         **dict(zip(["low", "high"], boot(relation)))},
    ]
    # regressione su tutta la stagione 2025-2026: log(auto) ~ log(traffico) + mese + meteo + weekend + anno + anno x regole
    s = s[s["cars"] > 0]
    s["regime"] = (s["year"] == 2026) & s["regulated_parking"]

    def design(x):
        cols = [np.ones(len(x)), np.log(x[DEMAND_PREDICTOR]), x["is_weekend"], (x["year"] == 2026).astype(float),
                x["regime"].astype(float)]
        cols += [(x["month"] == mo).astype(float) for mo in sorted(SEASONS)[1:]]
        cols += [(x["weather"] == w).astype(float) for w in WEATHER_CLASSES[1:]]
        return np.column_stack(cols).astype(float)

    def year_coef(x):
        coef_log, *_ = np.linalg.lstsq(design(x), np.log(x["cars"].to_numpy()), rcond=None)
        return np.exp(coef_log[3])

    values = [year_coef(s.iloc[rng.integers(0, len(s), len(s))]) for _ in range(N_BOOT)]
    rows.append({"check": "Regressione log con mese e meteo (2025-2026)", "estimate": year_coef(s),
                 "low": np.quantile(values, 0.05), "high": np.quantile(values, 0.95)})
    return pd.DataFrame(rows)


def year_effect_sensitivity(modal_shift_daily: pd.DataFrame, effects=np.linspace(0.85, 1.15, 31)) -> pd.DataFrame:
    """Stime dello spostamento modale al variare dell'effetto anno (A22), stessi giorni."""
    d = modal_shift_daily.rename(columns={"morning_up": DEMAND_PREDICTOR})
    year, regime = d[d["role"] == "year_effect"], d[d["role"] == "regime"]
    return pd.DataFrame([{"year_effect": e, **estimate_point(year, regime, e)} for e in effects])


def _condition_legend(ax, conditions, **kw):
    handles = [plt.Rectangle((0, 0), 1, 1, color=CONDITIONS[c][1]) for c in conditions]
    ax.legend(handles, [CONDITIONS[c][0] for c in conditions], frameon=False, **kw)


def plot_measurement_space(days: pd.DataFrame, pilot: dict, path: Path):
    """Sinistra: lo spazio delle condizioni di accesso (prezzo x navetta si/no), con i giorni misurati
    per anno e il servizio di navetta. Destra: il calendario di ogni anno, con cosa e' misurato giorno
    per giorno. `pilot`: l'esperimento naturale della navetta 2025 con il traffico come domanda
    (``shuttle_natural_experiment``)."""
    space = measurement_space(days)
    years = sorted(days["year"].unique())
    cells = {  # (riga, colonna) della griglia prezzo x navetta
        "free": (0, 0), "free_shuttle": (0, 1), "regime_no_shuttle": (1, 0), "regime": (1, 1),
    }
    fig = plt.figure(figsize=(15, 6.8))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.9], wspace=0.42)
    ax = fig.add_subplot(grid[0])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.set_xticks([0, 1], ["Senza navetta", "Con navetta"])
    ax.set_yticks([0, 1], ["5 €", "20 € +\nprenotazione\n+ semaforo"])
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for cond, (row, col) in cells.items():
        n = space[space["condition"] == cond].set_index("year")
        measured = n["days"].sum() > 0
        ax.add_patch(plt.Rectangle((col - 0.47, row - 0.45), 0.94, 0.9, color=CONDITIONS[cond][1] if measured else C_NEUTRAL_MID,
                                   alpha=0.22 if measured else 1, zorder=1))
        lines = [f"{y}: {int(n.loc[y, 'days'])} {'giorno' if n.loc[y, 'days'] == 1 else 'giorni'}"
                 + (f"\n{', '.join(SHUTTLE_NAMES.get(s, s) for s in n.loc[y, 'shuttle'].split(', '))}"
                    if n.loc[y, "shuttle"] else "")
                 + (f"\n({int(n.loc[y, 'with_bus_down'])} con discesa)" if n.loc[y, "with_bus_down"] else "")
                 for y in years if y in n.index]
        ax.text(col, row, "\n".join(lines) if lines else "non osservato", ha="center", va="center", fontsize=9.5,
                color=C_INK if lines else C_INK2, zorder=2)
    ax.set_title("Condizioni di accesso osservate (giorni con auto contate)", loc="left", color=C_INK, fontsize=10.5)

    ax = fig.add_subplot(grid[1])
    ax.grid(False)
    rows = []
    for i, year in enumerate(years):
        base = i * 4
        d = days[days["year"] == year]
        x = (d["date"] - pd.to_datetime(d["year"].astype(str) + "-06-01")).dt.days.to_numpy()
        for xi, cond in zip(x, d["condition"]):
            ax.add_patch(plt.Rectangle((xi, base), 1, 1.6, color=CONDITIONS[cond][1] if cond in CONDITIONS
                                       else C_NEUTRAL_MID, linewidth=0))
        for k, (col, label) in enumerate([("has_traffic", "traffico valido a Pellizzano"),
                                          ("has_bus_down", "persone in discesa (a piedi)")]):
            y0 = base + 1.9 + k * 0.55
            ax.add_patch(plt.Rectangle((0, y0), 122, 0.4, color=C_NEUTRAL_MID, linewidth=0))
            for xi in x[d[col].to_numpy()]:
                ax.add_patch(plt.Rectangle((xi, y0), 1, 0.4, color=C_INK2, linewidth=0))
            rows.append((y0 + 0.2, label))
        rows.append((base + 0.8, f"{year}: auto contate"))
    ax.set_xlim(0, 122)
    ax.set_ylim(len(years) * 4 - 0.6, -0.4)
    ax.set_yticks([r[0] for r in rows], [r[1] for r in rows], fontsize=9)
    ax.set_xticks([0, 30, 61, 92], ["1 giu", "1 lug", "1 ago", "1 set"])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Calendario: condizione del giorno e dati disponibili (grigio chiaro: nessun dato)",
                 loc="left", color=C_INK, fontsize=10.5)
    used = [c for c in CONDITIONS if c in set(days["condition"])]
    _condition_legend(ax, used, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=8.5)
    fig.suptitle("Cosa si può misurare: la navetta senza il prezzo solo nel 2025 (pilota, nessun effetto misurabile); "
                 "il prezzo di 20 € e le regole 2026\nsolo insieme alla Lakebus (A25)",
                 x=0.01, ha="left", fontsize=12, color=C_INK)
    coef, se = pilot["shuttle_running_coef"], pilot["shuttle_running_se"]
    note = ("Le due navette sono servizi diversi. Greenbus 2025: pilota 28/6-20/7, 1 €, giro lungo per Fucine e Ossana, "
            f"solo biglietti venduti; nessun calo misurabile delle auto a parità\ndi traffico in valle ({coef:+.0f} "
            f"auto/giorno, intervallo 90% da {coef - 1.645 * se:+.0f} a {coef + 1.645 * se:+.0f}, grafico 05b). "
            "Lakebus 2026: 4/7-30/8, ogni 30 minuti da Pellizzano, persone in salita e in discesa.")
    fig.text(0.01, -0.09, note, fontsize=8.5, color=C_INK2, va="top")
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_reference_choice(cv: pd.DataFrame, modal_shift_daily: pd.DataFrame, modal_shift: pd.DataFrame, path: Path):
    """Sinistra: errore fuori campione sul 2025 dei riferimenti in condizioni 2025. Destra: auto
    osservate / auto attese, giorno per giorno sui giorni 2026 con le regole, per i due riferimenti."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 1.4]})
    _style_ax(ax1)
    ax1.grid(True, axis="x")
    ax1.grid(False, axis="y")
    order = cv.iloc[::-1].reset_index(drop=True)
    ys = np.arange(len(order))
    h = 0.36
    for off, col, color, label in [(h / 2, "mae_2025", C_BLUE, f"2025, una settimana esclusa alla volta "
                                                                  f"({order['n_2025'].iloc[0]} giorni)"),
                                   (-h / 2, "mae_2026", C_ORANGE, f"2026 a 5 € senza regole, stima sul 2025 "
                                                                   f"({order['n_2026'].iloc[0]} giorni)")]:
        ax1.barh(ys + off, order[col], height=h - 0.04, color=color, zorder=3, label=label)
        for yi, v in zip(ys + off, order[col]):
            ax1.text(v + 1, yi, f"{v:.0f}", va="center", fontsize=9, color=C_INK2)
    ax1.set_yticks(ys, order["reference"], fontsize=9)
    for tick in ax1.get_yticklabels():
        if "scelta" in tick.get_text():
            tick.set_fontweight("bold")
    ax1.set_xlabel("Errore medio assoluto (auto/giorno)")
    ax1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.1), fontsize=8.5)
    ax1.set_xlim(0, order[["mae_2025", "mae_2026"]].to_numpy().max() * 1.12)
    ax1.set_title("Errore fuori campione: la scelta è seconda su entrambi i test", loc="left", color=C_INK,
                  fontsize=10.5)

    _style_ax(ax2)
    regime = modal_shift_daily[modal_shift_daily["role"] == "regime"].sort_values("date")
    response = modal_shift.set_index("parameter").loc["car_response", "estimate"]
    for col, color, label in [("cars_2025_levels", C_MUTED, "Livelli stagione x meteo (A21)"),
                              ("cars_2025_conditions", C_BLUE, "Relazione con traffico e meteo (A37)")]:
        ratio = regime["cars"] / regime[col]
        ax2.plot(regime["date"], ratio, color=color, linewidth=1.8, marker="o", markersize=3.5,
                 label=f"{label}: variazione giornaliera {ratio.std() / ratio.mean() * 100:.0f}%, "
                       f"correlazione {regime['cars'].corr(regime[col]):.2f}")
    ax2.axhline(response, color=C_INK2, linewidth=1, linestyle="--",
                label=f"Risposta delle auto (fuori-regole = 1): {response:.2f}")
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel("Auto osservate / auto attese in condizioni 2025")
    ax2.set_title("Giorni 2026 con le regole: con il traffico il rapporto oscilla meno", loc="left",
                  color=C_INK, fontsize=10.5)
    ax2.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax2.set_ylim(0, 0.8)
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")
    fig.suptitle("Riferimento \"auto attese con le condizioni 2025\": la relazione con il traffico del giorno "
                 "segue la domanda giorno per giorno", x=0.01, ha="left", fontsize=12, color=C_INK)
    note = ("Livelli stagione x meteo: stimati su tutto il 2025 (errore in campione, quindi ottimistico). Relazione A37: "
            "auto = a + b x auto in salita a Pellizzano 7-12 + c(meteo), stimata sul 2025.")
    fig.text(0.01, -0.02, note, fontsize=8.5, color=C_INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_year_effect(checks: pd.DataFrame, sensitivity: pd.DataFrame, path: Path):
    """Sinistra: le misure dell'effetto anno 2026 e la banda assunta (A22). Destra: le stime dello
    spostamento modale al variare dell'effetto anno."""
    band = (1 - YEAR_EFFECT_UNCERTAINTY, 1 + YEAR_EFFECT_UNCERTAINTY)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1, 1.2]})
    _style_ax(ax1)
    ax1.grid(True, axis="x")
    ax1.grid(False, axis="y")
    ax1.axvspan(*band, color=C_GOOD_STATUS, alpha=0.08, zorder=0)
    ax1.axvline(1, color=C_INK2, linewidth=1, linestyle="--")
    ys = np.arange(len(checks))[::-1]
    for yi, (_, r) in zip(ys, checks.iterrows()):
        ax1.errorbar(r["estimate"], yi, xerr=[[r["estimate"] - r["low"]], [r["high"] - r["estimate"]]], fmt="o",
                     color=C_BLUE, markersize=8, capsize=5, linewidth=1.6, zorder=3)
        ax1.text(r["high"] + 0.02, yi, f"{r['estimate']:.2f} ({r['low']:.2f}-{r['high']:.2f})", va="center",
                 fontsize=9, color=C_INK2)
    ax1.set_yticks(ys, checks["check"], fontsize=9)
    ax1.set_xlim(0.6, 1.75)
    ax1.set_xlabel("Effetto anno 2026 (1 = come il 2025)")
    ax1.text(band[0] + 0.005, ys.max() + 0.55, f"ipotesi A22: 1 ± {YEAR_EFFECT_UNCERTAINTY * 100:.0f}%",
             fontsize=9, color=C_GOOD_STATUS)
    ax1.set_ylim(-0.6, ys.max() + 0.9)
    ax1.set_title("Le misure: tutte compatibili con 1, nessuna precisa", loc="left", color=C_INK, fontsize=10.5)

    _style_ax(ax2)
    ax2.axvspan(*band, color=C_GOOD_STATUS, alpha=0.08, zorder=0)
    ax2.axhline(0, color=C_AXIS, linewidth=1)
    ax2.axvline(1, color=C_INK2, linewidth=1, linestyle="--")
    series = [("car_response", "Risposta delle auto", C_INK),
              ("with_walk_up__share_shuttle", "Quota in navetta", C_BLUE),
              ("with_walk_up__share_walk_up", "Quota a piedi", C_AQUA),
              ("with_walk_up__share_elsewhere", "Quota altrove", C_ORANGE)]
    for col, label, color in series:
        ax2.plot(sensitivity["year_effect"], sensitivity[col], color=color, linewidth=2, label=label)
        ax2.annotate(label, (sensitivity["year_effect"].iloc[-1], sensitivity[col].iloc[-1]), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=9, color=C_INK2)
    ax2.set_xlim(sensitivity["year_effect"].min(), sensitivity["year_effect"].max() + 0.06)
    ax2.set_xlabel("Effetto anno 2026 assunto")
    ax2.set_ylabel("Stima (frazione)")
    ax2.set_title("Quanto pesa: sotto 0,9 la quota in navetta supera 1", loc="left",
                  color=C_INK, fontsize=10.5)
    ax2.legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("Effetto anno 2026: fissato a 1 ± 10% (A22) perché i 18 giorni senza regole non bastano a misurarlo",
                 x=0.01, ha="left", fontsize=12, color=C_INK)
    note = ("Misure sui giorni 2026 a 5 € senza regole (giugno, 1 luglio, settembre; 18 giorni), intervalli bootstrap al "
            "90%. La regressione usa tutta la stagione 2025-2026\n(log auto ~ log traffico + mese + meteo + weekend + "
            "anno + regole). Banda verde: l'incertezza propagata nelle stime del modello.")
    fig.text(0.01, -0.02, note, fontsize=8.5, color=C_INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_monitoring(monitoring: pd.DataFrame, path: Path):
    """Gli indicatori di monitoraggio (D20) per anno, come rapporto al 2025, nei due periodi."""
    band = (1 - YEAR_EFFECT_UNCERTAINTY, 1 + YEAR_EFFECT_UNCERTAINTY)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    series = [("road_pellizzano_morning_up", "Traffico in salita a Pellizzano, mattina", C_BLUE),
              (CARS_PER_MORNING_CAR, "Auto a Fazzon per auto del mattino", C_ORANGE),
              (CARS_VS_RELATION, "Auto a Fazzon / relazione 2025 (A37)", C_VIOLET)]
    controls = [i for i in monitoring["indicator"].unique() if i.startswith("road_") and i not in
                ("road_pellizzano_morning_up", "road_pellizzano")]
    for ax, period, title in [(axes[0], "regime_window", "Finestra delle regole 2026 (4/7-31/8)"),
                              (axes[1], "rest_of_season", "Resto della stagione (giugno, settembre)")]:
        _style_ax(ax)
        ax.axhspan(*band, color=C_GOOD_STATUS, alpha=0.08, zorder=0)
        ax.axhline(1, color=C_INK2, linewidth=1, linestyle="--")
        m = monitoring[monitoring["period"] == period]
        for i, ind in enumerate(controls):
            d = m[m["indicator"] == ind].sort_values("year")
            ax.plot(d["year"], d["ratio_to_reference"], color=C_MUTED, linewidth=1, alpha=0.8,
                    label="Stazioni di controllo e totali (Pracorno, Celledizzo, Tonale, Revò)" if i == 0 else None)
        for k, (ind, label, color) in enumerate(series):
            d = m[m["indicator"] == ind].sort_values("year")
            ax.plot(d["year"], d["ratio_to_reference"], color=color, linewidth=2.2, marker="o", markersize=6,
                    label=label)
            if len(d):
                last = d.iloc[-1]
                ax.annotate(f"{last['ratio_to_reference']:.2f}", (last["year"], last["ratio_to_reference"]),
                            xytext=(7, 8 - 8 * k), textcoords="offset points", va="center", fontsize=9, color=C_INK2)
        ax.set_xticks(sorted(monitoring["year"].unique()))
        ax.set_title(title, loc="left", color=C_INK, fontsize=10.5)
    axes[0].set_ylabel("Rapporto al 2025 (stessi giorni del calendario)")
    axes[0].set_ylim(0, 1.3)
    axes[0].legend(frameon=False, loc="lower left", fontsize=8.5)
    fig.suptitle("Monitoraggio: la domanda in valle non cambia; con le regole 2026 un terzo delle auto, "
                 "fuori dalle regole come nel 2025\n(fascia verde: ±10%, l'incertezza ammessa sull'effetto anno, A22)",
                 x=0.01, ha="left", fontsize=12, color=C_INK)
    note = ("Auto a Fazzon solo dal 2025. Traffico: media giornaliera dei giorni validi (D10); Pracorno 2026 assente "
            "(sensore guasto). Rapporti grezzi, senza correzione per meteo e giorni della settimana.")
    fig.text(0.01, -0.01, note, fontsize=8.5, color=C_INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_cars_vs_traffic_by_year(daily: pd.DataFrame, days: pd.DataFrame, modal_shift: pd.DataFrame, path: Path):
    """Auto a Fazzon contro auto in salita a Pellizzano al mattino, per anno e condizione, con la
    relazione 2025 (bel tempo) e la stessa relazione per la risposta delle auto 2026."""
    d = days.merge(daily[["date", "cars", DEMAND_PREDICTOR, "weather"]], on="date")
    d = d[d["cars"].notna() & d[DEMAND_PREDICTOR].notna()]
    coef = fit_demand(calibration_days(daily))
    response = modal_shift.set_index("parameter").loc["car_response", "estimate"]
    fig, ax = plt.subplots(figsize=(12, 6.8))
    _style_ax(ax)
    for (year, cond), g in d[d["condition"].isin(CONDITIONS)].groupby(["year", "condition"]):
        ax.scatter(g[DEMAND_PREDICTOR], g["cars"], s=36, marker="o" if year == 2025 else "^",
                   color=CONDITIONS[cond][1], edgecolor=SURFACE, linewidth=0.8, zorder=3, alpha=0.9,
                   label=f"{year} - {CONDITIONS[cond][0]} ({len(g)})")
    xs = np.linspace(d[DEMAND_PREDICTOR].min(), d[DEMAND_PREDICTOR].max(), 50)
    good = pd.DataFrame({DEMAND_PREDICTOR: xs, "weather": "good"})
    ax.plot(xs, expected_cars(good, coef), color=C_INK2, linewidth=1.6, zorder=2)
    ax.plot(xs, expected_cars(good, coef) * response, color=C_ORANGE, linewidth=1.6, linestyle="--", zorder=2)
    ax.annotate("relazione 2025 (A37), bel tempo", (xs[-1], expected_cars(good, coef)[-1]), xytext=(-4, 8),
                textcoords="offset points", ha="right", fontsize=9, color=C_INK2)
    ax.annotate(f"x {response:.2f} (regole 2026)", (xs[-1], expected_cars(good, coef)[-1] * response),
                xytext=(-4, 8), textcoords="offset points", ha="right", fontsize=9, color=C_INK2)
    ax.set_xlabel(IT_LABELS[DEMAND_PREDICTOR])
    ax.set_ylabel(IT_LABELS["parking_tickets"])
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.set_title("Stesso traffico in valle, un terzo delle auto con le regole 2026 (cerchi: 2025; triangoli: 2026)",
                 loc="left", color=C_INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# %% Main
def main():
    print("Lettura dei dataset della pipeline Fazzon...")
    daily, road_daily, stations, reference = load_data()
    panel = build_panel(daily)
    panel.to_csv(OUTPUT_DIR / "panel_daily.csv", index=False)
    print(f"  {len(panel)} giorni, {panel['date'].min().date()} -> {panel['date'].max().date()}")

    print("\nTabella di correlazione (prime 10 per R2 entro-mese):")
    corr = correlation_table(panel)
    corr.to_csv(OUTPUT_DIR / "correlations.csv", index=False)
    print(corr.head(10).to_string(index=False))

    print("\nMatrice di correlazione (Pearson, coppie complete):")
    cmat = correlation_matrix(panel)
    cmat.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    print(f"  is_weekend vs bici_valley_total: r = {cmat.loc['is_weekend', 'bici_valley_total']:.3f}")
    print(f"  is_weekend vs bici_near_fazzon:  r = {cmat.loc['is_weekend', 'bici_near_fazzon']:.3f}")
    print(f"  auto SS42 verso monte vs presenze in valle (entro-mese): r = {correlation_road_vs_presences(panel):.3f}")

    print("\nPendenze di saturazione, soglia = mediana (auto vs bici/pedoni, vs presenze Vodafone in valle):")
    sat = saturation_slopes(panel, mode="median")
    print(sat.to_string(index=False))

    print("\nPendenze di saturazione, soglia ottimizzata per canale (minimizza l'errore totale):")
    sat_optimal = saturation_slopes(panel, mode="optimal")
    print(sat_optimal.to_string(index=False))

    print("\nPendenze di saturazione, soglia ottimizzata, domanda = auto SS42 Pellizzano verso monte (7-12):")
    road_panel = panel.dropna(subset=["strada_pellizzano_salita_mattina"])
    sat_road = saturation_slopes(road_panel, demand_col="strada_pellizzano_salita_mattina", mode="optimal")
    sat_road.insert(0, "demand", "strada_pellizzano_salita_mattina")
    print(sat_road.to_string(index=False))
    sat.insert(0, "demand", "vf_pres_apt_valdisole")
    sat_optimal.insert(0, "demand", "vf_pres_apt_valdisole")
    pd.concat([sat, sat_optimal, sat_road]).to_csv(OUTPUT_DIR / "saturation_slopes.csv", index=False)

    print("\nEsperimento naturale della navetta:")
    result = shuttle_natural_experiment(panel)
    for k, v in result.items():
        if k not in ("regression_coef_names", "regression_coef"):
            print(f"  {k}: {v}")

    print("\nEsperimento naturale della navetta, domanda = auto SS42 Pellizzano verso monte (7-12):")
    result_road = shuttle_natural_experiment(road_panel, demand_col="strada_pellizzano_salita_mattina")
    for k, v in result_road.items():
        if k not in ("regression_coef_names", "regression_coef"):
            print(f"  {k}: {v}")

    print("\nConfronto anno-su-anno sulla SS42 (Pellizzano vs valli di controllo):")
    did = shuttle_road_did(daily)
    did.to_csv(OUTPUT_DIR / "shuttle_road_did.csv", index=False)
    print(did.to_string(index=False))

    print("\nGenerazione dei grafici...")
    plot_target_overview(panel, OUTPUT_DIR / "01_target_overview.png")
    plot_correlation_bars(corr, OUTPUT_DIR / "02_predictor_correlations.png")
    plot_correlation_matrix(cmat, OUTPUT_DIR / "02b_correlation_matrix.png")
    plot_saturation(panel, "vf_pres_apt_valdisole", OUTPUT_DIR / "03_car_saturation.png", mode="median")
    plot_saturation(panel, "vf_pres_apt_valdisole", OUTPUT_DIR / "03b_car_saturation_optimal_breakpoint.png",
                     mode="optimal")
    plot_car_vs_bici_divergence(panel, "vf_pres_apt_valdisole", OUTPUT_DIR / "04_car_vs_bici_divergence.png",
                                 mode="median")
    plot_car_vs_bici_divergence(panel, "vf_pres_apt_valdisole",
                                 OUTPUT_DIR / "04b_car_vs_bici_divergence_optimal_breakpoint.png", mode="optimal")
    plot_shuttle_experiment(panel, "vf_pres_apt_valdisole", result, OUTPUT_DIR / "05_shuttle_natural_experiment.png")
    threshold = plot_bus_coverage_gap(panel, OUTPUT_DIR / "06_bus_data_coverage_gap.png")
    print(f"  soglia auto/giorno per giorno 'saturo' (quartile superiore) = {threshold:.0f}")
    plot_saturation(road_panel, "strada_pellizzano_salita_mattina", OUTPUT_DIR / "07_car_vs_road_traffic.png",
                    mode="optimal", title="Auto a Fazzon vs auto in salita sulla SS42 a Pellizzano")
    plot_shuttle_experiment(road_panel, "strada_pellizzano_salita_mattina", result_road,
                            OUTPUT_DIR / "05b_shuttle_natural_experiment_road.png")
    plot_shuttle_road_did(did, OUTPUT_DIR / "08_shuttle_road_year_over_year.png")
    plot_road_data_coverage(road_daily, panel, OUTPUT_DIR / "09_road_data_coverage.png")
    plot_sensor_map(stations, OUTPUT_DIR / "00_sensor_map.png")

    print("\nConfronto 2025 vs 2026 (medie giornaliere per periodo, stessi giorni della settimana):")
    park = load_parking_2025_2026(daily)
    park.to_csv(OUTPUT_DIR / "parking_2025_2026_daily.csv", index=False)
    comp = compare_2025_2026(park)
    comp.to_csv(OUTPUT_DIR / "compare_2025_2026.csv", index=False)
    print(comp.round(1).T.to_string())
    plot_2025_vs_2026_daily(park, OUTPUT_DIR / "10_2025_vs_2026_daily.png")
    plot_2025_vs_2026_periods(comp, OUTPUT_DIR / "11_2025_vs_2026_periods.png")

    print("\nDataset del modello (livelli per stagione e meteo, spostamento modale, risposta al prezzo)...")
    plot_demand_drivers(panel, daily, reference["fazzon_presence_stats"], OUTPUT_DIR / "12_demand_drivers.png")
    plot_modal_shift_split(reference["fazzon_modal_shift"], reference["fazzon_modal_shift_daily"],
                           OUTPUT_DIR / "13_modal_shift_split.png")
    plot_price_response(reference["fazzon_modal_shift"], OUTPUT_DIR / "14_price_response.png")

    print("\nCosa si misura in ogni anno (giorni per condizione di accesso):")
    mdays = measurement_days(daily)
    space = measurement_space(mdays)
    space.to_csv(OUTPUT_DIR / "measurement_space.csv", index=False)
    print(space.to_string(index=False))
    plot_measurement_space(mdays, result_road, OUTPUT_DIR / "15_measurement_space.png")

    print("\nRiferimento in condizioni 2025, errore fuori campione sul 2025 (A21 vs A37):")
    cv = reference_cross_validation(daily, reference["fazzon_presence_stats"])
    cv.to_csv(OUTPUT_DIR / "reference_cross_validation.csv", index=False)
    print(cv.round(1).to_string(index=False))
    plot_reference_choice(cv, reference["fazzon_modal_shift_daily"], reference["fazzon_modal_shift"],
                          OUTPUT_DIR / "16_reference_choice.png")

    print("\nEffetto anno 2026 (A22), misure:")
    checks = year_effect_checks(daily, reference["fazzon_modal_shift"], reference["fazzon_monitoring"])
    checks.to_csv(OUTPUT_DIR / "year_effect_checks.csv", index=False)
    print(checks.round(3).to_string(index=False))
    sensitivity = year_effect_sensitivity(reference["fazzon_modal_shift_daily"])
    sensitivity.to_csv(OUTPUT_DIR / "year_effect_sensitivity.csv", index=False)
    plot_year_effect(checks, sensitivity, OUTPUT_DIR / "17_year_effect.png")
    plot_monitoring(reference["fazzon_monitoring"], OUTPUT_DIR / "18_monitoring.png")
    plot_cars_vs_traffic_by_year(daily, mdays, reference["fazzon_modal_shift"], OUTPUT_DIR / "19_cars_vs_traffic_by_year.png")

    print(f"\nTutti gli output sono stati scritti in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
