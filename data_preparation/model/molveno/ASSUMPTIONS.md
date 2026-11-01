# Molveno: register of decisions and assumptions

Every modelling decision (`D*`) and assumption (`A*`) of the Molveno data-preparation pipeline
and of the Molveno model, with its evidence and the places where it is used. The code refers to
these IDs in comments; a value without an ID is a bug.

Status: **in use** · **to review** (in use, to be reviewed) · **to validate** · **planned** ·
**retired**.

Paths: `pipeline/` = `data_preparation/model/molveno/`, `model/` = `executor-backend/src/model/molveno/`.
Validation figures: `python -m model.molveno.estimator_comparison` (leave-one-year-out on the
training summers: train on two, test on the third; R², CRPS of the predictive distribution,
coverage of the 90 % interval).

## Decisions

| ID | Decision | Evidence / reason | Used in | Status |
|----|----------|-------------------|---------|--------|
| D1 | (Retired, replaced by D7.) Reference period 01/07–30/09/2023. | — | — | retired |
| D2 | (Retired, replaced by D8.) Tourists of a day taken from the following date. | — | — | retired |
| D3 | (Retired, replaced by D9.) Seasons by k-means on the presences. | Groups days by their own presences: not known in advance, cannot be set by a scenario. | — | retired |
| D4 | (Retired, replaced by D9, D11.) Weekday shift from the centred 7-day average. | — | — | retired |
| D5 | (Retired, replaced by D13.) Vodafone presences with all user countries summed. | — | — | retired |
| D6 | (Retired, replaced by D10.) Weather classes bad / unsettled / good. | — | — | retired |
| D7 | Training data: the summers (01/06–30/09) of 2023, 2024 and 2025 pooled, no year effect. The model covers the summer. | Monthly means almost identical across the three years: the year explains 0 % of the summer variation of both presences. Complete Vodafone and weather data. | `pipeline/presence_estimators.py` (`TRAINING_YEARS`, `SUMMER`) | in use |
| D8 | Tourists and excursionists of the same date. | The 2023 Vodafone presences are identical, day by day, in the older platform file (`vodafone_attendences`) and in `vodafone_attendences_new.csv`. | `pipeline/reference_datasets.py` (`build_molveno_daily`) | in use |
| D9 | Presence statistics estimated by gradient boosting (scikit-learn `HistGradientBoostingRegressor`) per summer day, on day of the summer, weekday, public holiday and measured weather; exported per calendar day x weekday x weather class. A scenario selects a date range (default: the whole summer). Comparison statistics: least squares on month x weekday + weather class, exported per month x weekday x weather class. | Validation (tourists / excursionists): R² 0.90 / 0.78, CRPS 227 / 432 (main statistics, as the model reads them) vs 0.80 / 0.63, 315 / 553 (comparison statistics). A finer calendar alone does not close the gap (weekly seasons 0.85 / 0.63; 8 seasons learnt by a regression tree 0.86 / 0.68; seasons learnt with 4 groups move by weeks from year to year). Aggregating the gradient boosting to months or half-months loses most of the gain (0.80 / 0.67, 0.83 / 0.68): the model needs the calendar day. | `pipeline/presence_estimators.py` (`gbm_presence_stats`, `additive_presence_stats`), `estimator_comparison.py`; `model/molveno_model.py` (`i_day_of_summer`, `cv_presence_estimator`), `molveno_presence_stats.py`, `molveno_backend.py` (period, default estimator) | in use |
| D10 | Weather: three classes, good / cool / rainy (A1), measured at station T0414 (A3). The gradient boosting is trained on the measured weather (tmax anomaly, rain); a class stands for the weather days of that class observed in the month (A4). | Within year, month and weekday, weather explains 32 % of the variation of excursionists with tmax anomaly and rain, 20 % with the classes (rainy −757, cool −477, good +396); tourists 14 % and 8 %. Measured weather vs class as a feature: excursionists R² 0.78 vs 0.78, CRPS 432 vs 437, coverage 95 % vs 93 %. | `pipeline/presence_estimators.py` (`add_weather`, `gbm_presence_stats`); `model/molveno_model.py` (`cv_weather`; food usage factor with `rainy`) | in use |
| D11 | Weekday: learnt by the gradient boosting together with the season (e.g. no weekend effect for excursionists in August); month x weekday cells in the comparison statistics. | Strong weekday effect (ANOVA p < 1e-9): weekend − weekday +541…+747 tourists; +1268…+1325 excursionists in June, July, September, +119 in August. | `pipeline/presence_estimators.py` | in use |
| D12 | Predictive distributions: normal; SD of the main statistics = spread of the predictions over the weather days of the class combined with the SD of the out-of-fold residuals (one training year left out at a time); comparison statistics: residual SD. Truncated at 0 in the model. | Coverage of the 90 % interval: 95 % (main), 90 % (comparison). Summing the season, weekday and weather SDs as independent doubles the SD (coverage 99 %). | `pipeline/presence_estimators.py` (`fit_gbm`, `fit_additive`); `model/molveno_presence_stats.py` (`_presence`) | in use |
| D13 | Vodafone presences of the comune of Molveno (`locId` 27, `TN_MKT_AL_3`): TOURIST = tourists, VISITOR = excursionists; Italians (`ITA`) plus the aggregated foreign total (`_STR`). | Single foreign countries are masked to 0 below 15 (privacy): summing them undercounts tourists by 8 % and excursionists by 7 % (2023–2025). As Fazzon D12. | `pipeline/standardize.py` (`standardize_vodafone`) | in use |
| D14 | Incertezza of the sustainability (the ± of the overall and per-constraint levels): the confidence interval of the average over a period with the scenario's number of days (the days of the period × the share of the weekdays and weather classes the scenario allows), computed on the simulated days (sd / √days, 80 %): how much the result varies from one year to another with the days that occur. It replaces the interval over the samples, which measured the precision of the simulation and shrank with more samples. The days are taken as independent (weather spells make the true variability larger); the uncertainty of the parameters is in the gradient of the field, not in the ±. Base scenario: ±4.5 points over the summer, ±9 for August | Decision of the user; the platform texts keep the term "incertezza" | `executor-backend/src/model/common/sustainability_field.py` (`expected_period_days`, `period_days`); `molveno_backend.py`` | in use |

## Assumptions

| ID | Assumption | Evidence / reason | Used in | Status |
|----|------------|-------------------|---------|--------|
| A1 | Weather classes: rainy = rain ≥ 5 mm; cool = not rainy and tmax below the month's climatological mean − 2 °C; good = the rest. Climatology: station T0414, 2023–2025. | Same definitions as Fazzon (A19 there), not re-tuned for Molveno. Shares in the summers 2023–2025: good 0.61, rainy 0.20, cool 0.18. | `pipeline/presence_estimators.py` (`RAINY_MM`, `COOL_DEGREES`, `weather_climatology`) | in use |
| A2 | Public holidays of the summer: 2 June, 15 August. School holidays are captured by the day of the summer (their dates move little from year to year). | — | `pipeline/presence_estimators.py` (`HOLIDAYS`) | in use |
| A3 | The weather of Molveno is measured at Meteotrentino station T0414 San Lorenzo in Banale (Pergoletti), 685 m, 9 km from Molveno: the reference station of the area "Giudicarie esteriori e Molveno" in the open-data service. | Station T0383 Molveno (835 m) is not available in the open-data service used (`datiStoriciCaratteristiciSerie*UnArea`). | `data_preparation/data_download/download_meteo_meteotrentino.py` (T0414), `molveno.yaml` | to review |
| A4 | A weather class of a scenario day stands for the mix of weather days of that class observed in the same month of the training summers. | — | `pipeline/presence_estimators.py` (`gbm_presence_stats`) | in use |
| A5 | Frequency of the weather classes per month: the shares of the training summers. | 3 summers, ~30 days per month and year. | `pipeline/presence_estimators.py` (`weather_frequencies`); `model/molveno_presence_stats.py` (`weather_weights`: weighted by the days of each month in the period) | in use |
| A6 | Gradient boosting hyperparameters fixed (300 iterations, learning rate 0.05, 15 leaves, ≥ 10 days per leaf, L2 1.0), not tuned. | Tuning on 366 days with 3 folds would overfit the validation. | `pipeline/presence_estimators.py` (`GBM_PARAMS`) | in use |
