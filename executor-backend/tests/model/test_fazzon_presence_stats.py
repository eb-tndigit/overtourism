import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from src.model.fazzon import fazzon_data
from src.model.fazzon.fazzon_presence_stats import (
    behaviour_params,
    load_presence_stats,
    make_presence_factories,
    presence_inputs,
)
from tests.conftest import PRESENCE_STATS, write_fazzon_stats

POLICY_2025 = {
    "parking price (euro per vehicle)": "5",
    "shuttle capacity (riders per day)": "0",
}


def test_reads_reference_dataset_from_data_dir():
    table = load_presence_stats()
    assert table.index.names == ["season", "weather"]
    assert len(table) == 12
    car, other = make_presence_factories(table, behaviour_params())
    assert car(season="august", weather="rainy", **POLICY_2025).mean() == pytest.approx(
        897.3 * 0.6
    )
    assert other(season="july", weather="good", **POLICY_2025).std() == pytest.approx(
        611.7 * 0.449 * 0.5
    )


def test_presence_inputs_reads_files_at_every_call(tmp_path, monkeypatch):
    stats = PRESENCE_STATS.copy()
    stats.loc[stats["season"] == "july", "mean_car_visitors"] = 1000.0
    write_fazzon_stats(tmp_path, stats=stats)
    monkeypatch.setenv(fazzon_data.DATA_DIR_ENV, str(tmp_path))
    seasons, weather, behaviour, car, _ = presence_inputs()
    assert seasons["july"] == pytest.approx(31 / 82)
    assert weather == pytest.approx({"good": 0.62, "cool": 0.2, "rainy": 0.18})
    assert behaviour["beta_per_euro"] == 0.09
    assert car(season="july", weather="cool", **POLICY_2025).mean() == pytest.approx(
        1000.0
    )


def test_arrivals_follow_the_modal_shift():
    """Car arrivals = demand x R(price); non-car arrivals gain the deterred visitors who walk
    up and the shuttle riders, capped by the shuttle capacity; the rest goes elsewhere.
    """
    table = load_presence_stats()
    car, other = make_presence_factories(table, behaviour_params())
    cell = table.loc[("august", "good")]
    r = np.exp(-0.09 * 15)  # tests/conftest.py MODAL_SHIFT
    price20 = {"parking price (euro per vehicle)": "20"}
    assert car(season="august", weather="good", **price20).mean() == pytest.approx(
        cell["mean_car_visitors"] * r
    )
    deterred = cell["mean_car_visitors"] * (1 - r)
    walk = deterred * 0.155
    for capacity, riders in (("0", 0.0), ("120", 120.0), ("3600", deterred * 0.77)):
        mean = other(
            season="august",
            weather="good",
            **price20,
            **{"shuttle capacity (riders per day)": capacity},
        ).mean()
        assert mean == pytest.approx(cell["mean_other_visitors"] + walk + riders)
    # below the reference price nothing changes (A27)
    assert car(season="august", weather="good", **POLICY_2025).mean() == pytest.approx(
        cell["mean_car_visitors"]
    )


def test_shuttle_usable_capacity():
    from src.model.fazzon.fazzon_presence_stats import (
        shuttle_capacity_level,
        shuttle_usable_capacity,
    )

    lakebus = dict(
        seats_per_bus=30, interval_min=30, start_hour=9, end_hour=19, peak_end_hour=12
    )
    # 2 buses: 20 runs x 60 seats = 1200/day; morning 6 runs x 60 = 360, 60 % in the morning -> 600
    assert shuttle_usable_capacity(buses=2, peak_share=0.6, **lakebus) == pytest.approx(
        600
    )
    assert shuttle_usable_capacity(buses=2, peak_share=0.2, **lakebus) == pytest.approx(
        1200
    )  # daily seats bind
    assert shuttle_usable_capacity(buses=0, peak_share=0.6, **lakebus) == 0
    assert shuttle_capacity_level(600) == "600" and shuttle_capacity_level(629) == "600"
    assert shuttle_capacity_level(10_000) == "3600"


def test_more_buses_help_only_up_to_the_shuttle_demand():
    """The reported issue: one bus must not carry everybody (capacity is bounded)."""
    from src.model.fazzon.fazzon_backend import FazzonBackend

    backend = FazzonBackend()
    price = {"parking price (euro per vehicle)": 20.0}
    other = [  # x = all visitors arriving
        np.mean(backend.evaluate({**price, "shuttle buses": float(b)}).samples_x)
        for b in (0, 1, 2, 4, 6)
    ]
    assert (
        other[0] < other[1] < other[2]
    )  # more seats, more riders while the capacity binds
    assert other[-1] == pytest.approx(other[-2], rel=0.02)  # then the demand binds


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_presence_stats(tmp_path)


def test_missing_cell_raises(tmp_path):
    write_fazzon_stats(
        tmp_path, stats=PRESENCE_STATS[PRESENCE_STATS["season"] != "june"]
    )
    with pytest.raises(ValueError, match="cells"):
        load_presence_stats(tmp_path)


def test_missing_column_raises(tmp_path):
    write_fazzon_stats(tmp_path, stats=PRESENCE_STATS.drop(columns="weather_freq_rel"))
    with pytest.raises(ValueError, match="missing columns"):
        load_presence_stats(tmp_path)


def test_weather_weights_average_the_seasons(tmp_path, monkeypatch):
    stats = PRESENCE_STATS.copy()
    july = stats["season"] == "july"
    stats.loc[july, "weather_freq_rel"] = stats.loc[july, "weather"].map(
        {"good": 0.5, "cool": 0.3, "rainy": 0.2}
    )
    write_fazzon_stats(tmp_path, stats=stats)
    seasons, weather, _, _, _ = presence_inputs(tmp_path)
    expected_good = seasons["july"] * 0.5 + (1 - seasons["july"]) * 0.62
    assert weather["good"] == pytest.approx(expected_good)
    assert sum(weather.values()) == pytest.approx(1.0)


def test_season_and_weather_can_be_fixed():
    from src.model.fazzon.fazzon_backend import FazzonBackend

    backend = FazzonBackend()
    for overrides in (
        {"weather": "rainy"},
        {"season": "august"},
        {"season": "august", "weather": "good"},
    ):
        output = backend.evaluate(overrides)
        assert len(output.samples_x) > 0


# ---------------------------------------------------------------- download / startup


def _fake_digitalhub(monkeypatch, source_dir=None, error=None):
    """Replace the digitalhub client: dataitems are copied from *source_dir*, or *error* is raised."""

    def get_dataitem(name):
        def download(dest, overwrite=False):
            if error is not None:
                raise error
            (Path(dest) / name).write_bytes((source_dir / name).read_bytes())

        return SimpleNamespace(download=download)

    monkeypatch.setitem(
        sys.modules,
        "digitalhub",
        SimpleNamespace(
            get_project=lambda name: SimpleNamespace(get_dataitem=get_dataitem)
        ),
    )


def test_download_replaces_files(tmp_path, monkeypatch):
    remote = PRESENCE_STATS.copy()
    remote["mean_car_visitors"] += 1.0
    source = write_fazzon_stats(tmp_path / "remote", stats=remote)
    target = write_fazzon_stats(tmp_path / "local")
    monkeypatch.setenv(fazzon_data.DATA_DIR_ENV, str(target))
    _fake_digitalhub(monkeypatch, source_dir=source)

    assert fazzon_data.download_fazzon_data(overwrite=True) == target
    table = load_presence_stats(target)
    assert table.loc[("august", "good"), "mean_car_visitors"] == pytest.approx(898.3)


def test_download_skipped_when_files_present(tmp_path, monkeypatch):
    target = write_fazzon_stats(tmp_path / "local")
    monkeypatch.setenv(fazzon_data.DATA_DIR_ENV, str(target))
    _fake_digitalhub(monkeypatch, error=AssertionError("must not download"))
    assert fazzon_data.download_fazzon_data() == target


def test_failed_download_raises_and_keeps_files(tmp_path, monkeypatch):
    target = write_fazzon_stats(tmp_path / "local")
    before = (target / fazzon_data.PRESENCE_STATS_FILE).read_bytes()
    monkeypatch.setenv(fazzon_data.DATA_DIR_ENV, str(target))
    _fake_digitalhub(monkeypatch, error=OSError("platform unreachable"))
    with pytest.raises(RuntimeError, match="platform unreachable"):
        fazzon_data.download_fazzon_data(overwrite=True)
    assert (target / fazzon_data.PRESENCE_STATS_FILE).read_bytes() == before


def test_unusable_download_raises_and_keeps_files(tmp_path, monkeypatch):
    source = write_fazzon_stats(
        tmp_path / "remote", stats=PRESENCE_STATS.drop(columns="season_freq_rel")
    )
    target = write_fazzon_stats(tmp_path / "local")
    before = (target / fazzon_data.PRESENCE_STATS_FILE).read_bytes()
    monkeypatch.setenv(fazzon_data.DATA_DIR_ENV, str(target))
    _fake_digitalhub(monkeypatch, source_dir=source)
    with pytest.raises(ValueError, match="missing columns"):
        fazzon_data.download_fazzon_data(overwrite=True)
    assert (target / fazzon_data.PRESENCE_STATS_FILE).read_bytes() == before


def test_startup_standalone_requires_files(tmp_path, monkeypatch):
    monkeypatch.setenv(fazzon_data.DATA_DIR_ENV, str(tmp_path / "empty"))
    monkeypatch.setenv("DT_OVERTURISM_STANDALONE_MODE", "true")
    with pytest.raises(FileNotFoundError, match="missing"):
        fazzon_data.ensure_fazzon_data()


def test_startup_downloads_when_not_standalone(tmp_path, monkeypatch):
    source = write_fazzon_stats(tmp_path / "remote")
    monkeypatch.setenv(fazzon_data.DATA_DIR_ENV, str(tmp_path / "local"))
    monkeypatch.setenv("DT_OVERTURISM_STANDALONE_MODE", "false")
    _fake_digitalhub(monkeypatch, source_dir=source)
    assert fazzon_data.ensure_fazzon_data() == tmp_path / "local"


def _evaluate_at_points(backend, overrides=None, **points):
    """Evaluate with the uncertain usage parameters pinned at single values (A39), so that
    the usage fields can be checked exactly; *points* override the pinned values."""
    from civic_digital_twins.dt_model import Scenario
    from scipy import stats

    def point(
        value,
    ):  # a distribution-backed index takes a distribution: a single point
        return stats.rv_discrete(values=([value], [1.0]))

    inp = backend.model.inputs
    pins = {
        inp.i_occupancy_low: point(2.5),
        inp.i_occupancy_high: point(3.5),
        inp.i_occupancy_ratio: point(0.555),
        inp.i_u_food_fraction_car: point(points.get("food_car", 0.50)),
        inp.i_u_food_fraction_other: point(points.get("food_other", 0.66)),
        inp.i_xo_food_sittings: point(2.0),
    }
    scenario = Scenario(
        backend.model,
        overrides={**pins, **(overrides or {})},
        parameter_axes=backend._parameter_axes,
    )
    return backend._evaluate_scenario(scenario)


def test_policy_moves_the_samples_and_scales_the_vehicles():
    """The access policy moves the arrivals (samples); on the field it acts only through the
    persons per vehicle (A2): same car arrivals, fewer vehicles at a higher price."""
    import numpy as np
    from src.model.fazzon.fazzon_backend import FazzonBackend

    backend = FazzonBackend()
    base = backend.evaluate({})
    policy = backend.evaluate(
        {"parking price (euro per vehicle)": 20.0, "shuttle buses": 2.0}
    )
    # x = all visitors arriving, y = car arrivals: total never below the car arrivals
    assert np.mean(policy.samples_y) < 0.4 * np.mean(base.samples_y)
    assert all(t >= c for t, c in zip(policy.samples_x, policy.samples_y))
    assert np.mean(np.subtract(policy.samples_x, policy.samples_y)) > np.mean(
        np.subtract(base.samples_x, base.samples_y)
    )
    model = backend.model
    base = _evaluate_at_points(backend)
    policy = _evaluate_at_points(
        backend, {model.cv_parking_price: "20", model.i_shuttle_buses: 2.0}
    )
    i = 10
    assert policy.usage_fields["parking"][i, i] / base.usage_fields["parking"][
        i, i
    ] == pytest.approx(2.5 / 3.5)
    # food depends on how the people at the lake arrived, not on what they paid
    assert policy.usage_fields["food"][i, i] == pytest.approx(
        base.usage_fields["food"][i, i]
    )


def test_field_axes_and_food_usage_by_arrival_mode():
    """x = all arrivals (0-3000), y = car arrivals (0-2000); at the same total, fewer car
    arrivals mean more diners (A10): the restaurant boundary is not parallel to the x axis.
    """
    from src.model.fazzon.fazzon_backend import FazzonBackend

    backend = FazzonBackend()
    out = _evaluate_at_points(backend)
    assert out.x_values.max() == 3000 and out.y_values.max() == 2000
    food = out.usage_fields["food"]
    t, c = 50, 25  # 1500 arrivals, 500 by car
    assert food[t, c] == pytest.approx((500 * 0.50 + 1000 * 0.66) / 2.0)
    assert food[t, 0] > food[t, c] > food[t, 50]
    same = _evaluate_at_points(backend, food_car=0.55, food_other=0.55)
    assert same.usage_fields["food"][t, 0] == pytest.approx(
        same.usage_fields["food"][t, c]
    )
    # with the ranges, the same ordering holds on average
    ranged = backend.evaluate({}).usage_fields["food"]
    assert ranged[t, 0] > ranged[t, c] > ranged[t, 50]


def test_behaviour_derived_from_vehicle_response_and_occupancy():
    """A39: at the point values the behaviour is the pipeline's; more persons per vehicle under
    the rules (same cars) means more visitors kept by car: smaller beta, fewer visitors lost,
    so larger shuttle and walk-up shares."""
    from src.model.fazzon.fazzon_presence_stats import behaviour_for

    b = behaviour_params()
    point = behaviour_for(b, b["car_response"], b["occupancy_low"], b["occupancy_high"])
    for key in ("beta_per_euro", "share_shuttle", "share_walk_up"):
        assert point[key] == pytest.approx(b[key])
    fuller = behaviour_for(b, b["car_response"], b["occupancy_low"], 4.0)
    assert fuller["beta_per_euro"] < b["beta_per_euro"]
    assert fuller["share_shuttle"] > b["share_shuttle"]
    assert fuller["share_shuttle"] + fuller["share_walk_up"] <= 1.0 + 1e-12
    # a weaker vehicle response (fewer cars kept) means a larger beta
    assert (
        behaviour_for(b, 0.8 * b["car_response"], 2.5, 3.5)["beta_per_euro"]
        > b["beta_per_euro"]
    )


def test_policy_kpis():
    """Arrivals out of the potential visitors: all in 2025 conditions; the shuttle brings
    back part of the visitors deterred by the price. Vehicles out of the 2025 rules: R(price)
    x persons per vehicle at 5 euro / at the price (A36)."""
    from src.model.fazzon.fazzon_backend import FazzonBackend
    from src.model.fazzon.fazzon_presence_stats import (
        behaviour_params,
        occupancy,
        price_response,
    )

    backend = FazzonBackend()
    price = {"parking price (euro per vehicle)": 20.0}
    kpis = {
        label: backend.evaluate(overrides).kpis
        for label, overrides in (
            ("2025", {}),
            ("price", price),
            ("lakebus", {**price, "shuttle buses": 2.0}),
        )
    }
    for days in ("mean", "peak"):
        assert kpis["2025"][f"arrivals {days}"]["share"] == 100
        assert kpis["2025"][f"vehicles {days}"]["share"] == 100
        assert (
            kpis["price"][f"arrivals {days}"]["share"]
            < kpis["lakebus"][f"arrivals {days}"]["share"]
            <= 100
        )
    for k in kpis.values():
        assert k["arrivals peak"]["value"] >= k["arrivals mean"]["value"]
        assert k["vehicles peak"]["value"] >= k["vehicles mean"]["value"]
    behaviour = behaviour_params()
    expected = (
        price_response(20.0, behaviour)
        * occupancy(5.0, behaviour)
        / occupancy(20.0, behaviour)
    )
    assert kpis["price"]["vehicles mean"]["share"] == pytest.approx(
        expected * 100, abs=0.01
    )
    # the shuttle does not change the vehicles
    assert kpis["lakebus"]["vehicles mean"] == kpis["price"]["vehicles mean"]
    august = backend.evaluate({**price, "season": "august"}).kpis
    assert august["vehicles mean"]["value"] > kpis["price"]["vehicles mean"]["value"]


def test_kpis_order_model_kpis_first():
    """Display order: the model KPIs, the sustainability level, the per-constraint levels,
    the critical constraint."""
    from src.model.fazzon.fazzon_backend import FazzonBackend

    keys = list(FazzonBackend().evaluate({}).kpis)
    assert keys == [
        "arrivals mean",
        "arrivals peak",
        "vehicles mean",
        "vehicles peak",
        "sustainability_level",
        "constraint level parking",
        "constraint level road",
        "constraint level food",
        "critical constraint",
    ]


def test_price_slider_is_converted_to_its_level():
    from src.model.fazzon.fazzon_backend import FazzonBackend, price_level

    assert [price_level(p) for p in (0, 5, 7.4, 7.6, 15, 29.6, 100)] == [
        "5",
        "5",
        "7",
        "8",
        "15",
        "30",
        "30",
    ]
    backend = FazzonBackend()
    a = backend.evaluate({"parking price (euro per vehicle)": 14.6})
    b = backend.evaluate({"parking price (euro per vehicle)": 15.0})
    c = backend.evaluate({"parking price (euro per vehicle)": 14.0})
    assert np.array_equal(a.samples_x, b.samples_x)  # same 1-euro class
    assert np.mean(c.samples_y) > np.mean(b.samples_y)  # 1 euro less, more car arrivals


def test_behaviour_params_missing_parameter_raises(tmp_path):
    from src.model.fazzon.fazzon_presence_stats import behaviour_params
    from tests.conftest import MODAL_SHIFT

    write_fazzon_stats(
        tmp_path, modal_shift=MODAL_SHIFT[MODAL_SHIFT["parameter"] != "beta_per_euro"]
    )
    with pytest.raises(ValueError, match="beta_per_euro"):
        behaviour_params(tmp_path)
