"""Fazzon (Lago dei Caprioli, Val di Sole) overtourism model definition.

The model uses the ``@define`` / ``compute()`` contract throughout — no
``legacy=True`` escape hatches.  Each concern sub-model declares its I/O
contract via ``@inputs`` / ``@outputs`` dataclasses and implements a single
``compute()`` method.  The root :class:`FazzonModel` wires all sub-models
inside its own ``compute()`` and exposes scenario-accessible indexes as
plain attributes set there.

Sub-models:

:class:`AccessModel`
    Vehicles of the car arrivals: persons per vehicle as a function of the parking price (A2).

:class:`ParkingModel`
    Simultaneous car occupancy vs. policy cap (≈ 150 simultaneous spaces).

:class:`RoadModel`
    Daily vehicle passages (cars + shuttle buses) vs. effective road capacity
    under semaphore control.

:class:`FoodModel`
    Peak simultaneous diners vs. total seat capacity (4 venues, 520 seats); visitors by car
    use the restaurants less than the others (picnic, A10).

:class:`FazzonModel`
    Root model — owns the CVs, two PVs, all ``i_*`` defaults, and wires the
    access model and the three concern sub-models.  Call :meth:`FazzonModel.default_inputs` to obtain a
    ready-to-use :class:`~FazzonModel.Inputs` instance.

Every default value refers to an entry of the Fazzon assumptions register,
``data_preparation/model/fazzon/ASSUMPTIONS.md`` (IDs ``A*`` / ``D*``).  The
lakeside constraint was dropped (D6).

Context variables
-----------------
cv_season   : june / july / august / september   (calendar days of each month; A14)
cv_weather  : good / cool / rainy  (climatology averaged over the seasons; D7, A19, A20, D13)
cv_parking_price : 5, 6, ..., 30 euro per vehicle (always fixed by the backend; D15)
cv_shuttle  : usable shuttle capacity class, riders per day (derived by the backend from the
              shuttle timetable parameters; D15, D16)

Presence variables
------------------
pv_visitors_car   : daily arrivals by private vehicle (valley traffic via the 2025 relation, modal shift; A37, A38, A2, A23, A26)
pv_visitors_other : daily arrivals by other means (A3, A4, A6, A28)

No day-type CV (D5).
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from civic_digital_twins.dt_model import (
    CategoricalIndex,
    ConditionalDistributionIndex,
    DistributionIndex,
    GenericIndex,
    Index,
    Model,
    define,
    graph,
    inputs,
    outputs,
)
from scipy import stats

from .fazzon_presence_stats import (
    DEMAND_LEVEL_NAME,
    OCCUPANCY_HIGH_NAME,
    OCCUPANCY_LOW_NAME,
    PRICE_INDEX_NAME,
    PRICE_LEVELS,
    SHUTTLE_CAPACITY_LEVELS,
    SHUTTLE_INDEX_NAME,
    VEHICLE_RESPONSE_NAME,
    WEATHERS,
    presence_inputs,
)

# Ranges of the uncertain parameters (A39).  PLACEHOLDER: a range chosen without a source,
# to be validated; the user can change every range in the dashboard.
OCCUPANCY_LOW_RANGE = (
    2.3,
    2.9,
)  # A2: low end PLACEHOLDER; high end: 2021 survey, 2.69-2.87 per car
OCCUPANCY_HIGH_RANGE = (3.0, 4.0)  # A2: PLACEHOLDER around the operator's ~3.5
DEMAND_LEVEL_RANGE = (0.9, 1.1)  # A22: year effect 1 +- 10 %
OCCUPANCY_RATIO_RANGE = (0.49, 0.61)  # A8: 2021 survey, 0.495-0.604 by day
FOOD_FRACTION_CAR_RANGE = (0.40, 0.60)  # A10: PLACEHOLDER
FOOD_FRACTION_OTHER_RANGE = (0.56, 0.76)  # A10: PLACEHOLDER
FOOD_SITTINGS_RANGE = (1.5, 2.5)  # A11: PLACEHOLDER


def _uniform(name: str, bounds: tuple[float, float]) -> DistributionIndex:
    lo, hi = bounds
    return DistributionIndex(name, stats.uniform, {"loc": lo, "scale": hi - lo})


def _triang(name: str, bounds: tuple[float, float], mode: float) -> DistributionIndex:
    """Triangular on *bounds* with its mode at the point value (clipped into the range)."""
    lo, hi = bounds
    c = min(max((mode - lo) / (hi - lo), 0.0), 1.0)
    return DistributionIndex(name, stats.triang, {"loc": lo, "scale": hi - lo, "c": c})


# ---------------------------------------------------------------------------
# Constraint
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Constraint:
    """Named pairing of a usage formula index and a capacity index."""

    name: str
    usage: Index
    capacity: Index


# ---------------------------------------------------------------------------
# AccessModel
# ---------------------------------------------------------------------------


@define("Access")
class AccessModel(Model):
    """Concern sub-model — vehicles needed by the car arrivals under the parking price (D15).

    The presence variables are already the arrivals under the access policy (price
    response and modal split are in their distributions, D8).  Here the car arrivals are
    turned into vehicles with the persons per vehicle, which grow with the price (A2).

    Formula::

        price       = numeric value of cv_parking_price (euro per vehicle)
        i_occupancy = clip(linear in price between the two occupancy points)
        i_cars      = pv_visitors_car / i_occupancy
    """

    @inputs
    class Inputs:
        """Contractual inputs of :class:`AccessModel`."""

        pv_visitors_car: ConditionalDistributionIndex
        cv_parking_price: CategoricalIndex
        i_occupancy_low_price: Index
        i_occupancy_low: Index
        i_occupancy_high_price: Index
        i_occupancy_high: Index

    @outputs
    class Outputs:
        """Contractual outputs of :class:`AccessModel`."""

        i_occupancy: Index
        i_cars: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Persons per vehicle at the price level, and the vehicles of the car arrivals."""
        price = graph.piecewise(
            *[
                (float(level), inputs.cv_parking_price == str(level))
                for level in PRICE_LEVELS
            ],
            (float(PRICE_LEVELS[0]), True),
        )
        slope = (inputs.i_occupancy_high - inputs.i_occupancy_low) / (
            inputs.i_occupancy_high_price - inputs.i_occupancy_low_price
        )
        linear = inputs.i_occupancy_low + (price - inputs.i_occupancy_low_price) * slope
        # clip to [low, high]; min(a, b) = -max(-a, -b) (dt_model.graph exports no minimum)
        occupancy = -graph.maximum(
            -graph.maximum(linear, inputs.i_occupancy_low), -inputs.i_occupancy_high
        )  # A2
        i_occupancy = Index("persons per vehicle", occupancy)
        i_cars = Index("vehicles", inputs.pv_visitors_car / i_occupancy)
        return AccessModel.Outputs(i_occupancy=i_occupancy, i_cars=i_cars)


# ---------------------------------------------------------------------------
# ParkingModel
# ---------------------------------------------------------------------------


@define("Parking")
class ParkingModel(Model):
    """Concern sub-model — simultaneous parking occupancy.

    Converts the daily vehicles to a simultaneous occupancy estimate using the observed
    occupancy ratio (dwell_time / operating_window = 0.555, A8), then compares against
    the policy cap (A12).

    Formula::

        i_u_parking = i_cars * i_occupancy_ratio
    """

    @inputs
    class Inputs:
        """Contractual inputs of :class:`ParkingModel`."""

        i_cars: Index
        i_occupancy_ratio: Index
        i_c_parking: DistributionIndex

    @outputs
    class Outputs:
        """Contractual outputs of :class:`ParkingModel`."""

        i_u_parking: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute simultaneous parking occupancy from daily vehicles."""
        i_u_parking = Index("parking usage", inputs.i_cars * inputs.i_occupancy_ratio)
        self.constraint = Constraint(
            name="parking", usage=i_u_parking, capacity=inputs.i_c_parking
        )
        return ParkingModel.Outputs(i_u_parking=i_u_parking)


# ---------------------------------------------------------------------------
# RoadModel
# ---------------------------------------------------------------------------


@define("Road")
class RoadModel(Model):
    """Concern sub-model — vehicle passages on the Fazzon access road.

    The access road (4.3 km, single-lane with semaphore under shuttle operation) carries
    private vehicles and shuttle buses.  The daily vehicle passages are compared to an
    effective road capacity (A13).  There is no road cascade on the other constraints:
    the presence variables are arrivals, already at the lake (D19).

    Formula::

        i_u_road = i_cars * i_xo_road_trips + i_shuttle_daily_trips
    """

    @inputs
    class Inputs:
        """Contractual inputs of :class:`RoadModel`."""

        i_cars: Index
        i_xo_road_trips: Index
        i_shuttle_daily_trips: Index
        i_c_road: DistributionIndex

    @outputs
    class Outputs:
        """Contractual outputs of :class:`RoadModel`."""

        i_u_road: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute road vehicle passages and the road constraint."""
        _road_vehicle_demand = (
            inputs.i_cars * inputs.i_xo_road_trips + inputs.i_shuttle_daily_trips
        )
        i_u_road = Index("road usage", _road_vehicle_demand)

        self.constraint = Constraint(
            name="road", usage=i_u_road, capacity=inputs.i_c_road
        )
        return RoadModel.Outputs(i_u_road=i_u_road)


# ---------------------------------------------------------------------------
# FoodModel
# ---------------------------------------------------------------------------


@define("Food")
class FoodModel(Model):
    """Concern sub-model — daily restaurant demand at the 4 lake venues.

    Models peak simultaneous diners as a fraction of the visitors reaching the lake
    (all arrivals, D19), divided by the daily seat turnover.  The fraction differs between
    the visitors arrived by car (who can bring a picnic) and the others (A10).  The
    constraint is peak simultaneous demand ≤ total seats (520).

    Formula::

        other    = max(pv_visitors_total - pv_visitors_car, 0)
        i_u_food = (pv_visitors_car * i_u_food_fraction_car
                    + other * i_u_food_fraction_other) / i_xo_food_sittings
    """

    @inputs
    class Inputs:
        """Contractual inputs of :class:`FoodModel`."""

        pv_visitors_car: ConditionalDistributionIndex
        pv_visitors_total: ConditionalDistributionIndex
        i_u_food_fraction_car: Index
        i_u_food_fraction_other: Index
        i_xo_food_sittings: Index
        i_c_seats: Index

    @outputs
    class Outputs:
        """Contractual outputs of :class:`FoodModel`."""

        i_u_food: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute peak simultaneous diners."""
        # above the diagonal (more car arrivals than arrivals) the field is not reachable
        other = graph.maximum(inputs.pv_visitors_total - inputs.pv_visitors_car, 0.0)
        diners = (
            inputs.pv_visitors_car * inputs.i_u_food_fraction_car
            + other * inputs.i_u_food_fraction_other
        )
        i_u_food = Index("food usage", diners / inputs.i_xo_food_sittings)
        self.constraint = Constraint(
            name="food", usage=i_u_food, capacity=inputs.i_c_seats
        )
        return FoodModel.Outputs(i_u_food=i_u_food)


# ---------------------------------------------------------------------------
# FazzonModel
# ---------------------------------------------------------------------------


@define("Fazzon")
class FazzonModel(Model):
    """Root overtourism model for Fazzon — wires the access model and the three concern sub-models.

    Instantiate via :meth:`default_inputs`::

        model = FazzonModel(inputs=FazzonModel.default_inputs())

    Callers can override any parameter via :class:`~dt_model.Scenario`
    overrides targeting the scalar indexes exposed as instance attributes
    (e.g. ``model.i_c_parking``, ``model.cv_parking_price``, ``model.i_shuttle_buses``).  These
    attributes are aliased from ``model.inputs.*`` inside ``compute()`` and
    refer to the same index object, so scenario overrides work identically
    whether you target ``model.i_c_parking`` or ``model.inputs.i_c_parking``.

    Context variables
    -----------------
    cv_season   : june / july / august / september   (prior: calendar days of each month, A14)
    cv_weather  : good / cool / rainy                (prior: climatology, A20, D13)
    cv_parking_price, cv_shuttle : access policy     (always fixed by the backend, D15)

    Presence variables
    ------------------
    pv_visitors_total : daily arrivals, all means           (axis x of the field)
    pv_visitors_car   : daily arrivals by private vehicle   (axis y of the field)
    pv_visitors_other : daily arrivals by other means       (scatter sampling only)

    Both depend on the season, the weather, the parking price level and whether the
    shuttle runs (D15): the access policy moves the arrivals on the field (modal shift),
    and changes the field only through the persons per vehicle (:class:`AccessModel`).

    The assumptions (``A*`` IDs) are listed, with their status, in
    ``data_preparation/model/fazzon/ASSUMPTIONS.md``.
    """

    @inputs
    class Inputs:
        """All domain parameters of :class:`FazzonModel`."""

        # Context variables
        cv_season: CategoricalIndex
        cv_weather: CategoricalIndex
        # Presence distributions
        pv_visitors_car: ConditionalDistributionIndex
        pv_visitors_total: ConditionalDistributionIndex
        pv_visitors_other: ConditionalDistributionIndex  # sampling only (scatter)
        # Access policy (D15)
        cv_parking_price: CategoricalIndex
        cv_shuttle: CategoricalIndex
        # Shuttle timetable (A29, A35): all buses leave together every interval, from the
        # first departure until the end of service (runs = (end - start) / interval)
        i_shuttle_buses: Index
        i_shuttle_seats_per_bus: Index
        i_shuttle_interval_min: Index
        i_shuttle_start_hour: Index
        i_shuttle_end_hour: Index
        i_shuttle_peak_end_hour: Index
        i_shuttle_peak_share: Index
        # Persons per vehicle (A2)
        i_occupancy_low_price: Index
        i_occupancy_low: DistributionIndex
        i_occupancy_high_price: Index
        i_occupancy_high: DistributionIndex
        # Uncertain quantities of the arrivals only (parents of the presence variables, A39)
        i_vehicle_response: DistributionIndex
        i_demand_level: DistributionIndex
        # Conversion parameters
        i_occupancy_ratio: DistributionIndex
        i_xo_road_trips: Index
        # Food parameters
        i_u_food_fraction_car: DistributionIndex
        i_u_food_fraction_other: DistributionIndex
        i_xo_food_sittings: DistributionIndex
        # Capacity parameters
        i_c_parking: DistributionIndex
        i_c_road: DistributionIndex
        i_c_seats: Index

    @outputs
    class Outputs:
        """Contractual outputs of :class:`FazzonModel`."""

        usage_indexes: list[GenericIndex]

    @classmethod
    def default_inputs(cls) -> FazzonModel.Inputs:
        """Return default domain inputs.

        All defaults are calibrated from 2025 Fazzon parking ticket data and
        the 2022 EETRA modal-split survey.  Values marked ASSUMPTION have not
        been confirmed by domain experts.

        Override individual fields with :func:`dataclasses.replace` or supply
        a :class:`~dt_model.Scenario` at evaluation time.

        The presence statistics are read from the Fazzon data folder at every call
        (see :mod:`src.model.fazzon.fazzon_data`).
        """
        season, weather, behaviour, car_visitors_stats, other_visitors_stats = (
            presence_inputs()
        )
        cv_season = CategoricalIndex("season", {s: season[s] for s in season})  # A14
        cv_weather = CategoricalIndex(
            "weather", {w: weather[w] for w in WEATHERS}
        )  # D7, D13
        # D15: the policy categories are always fixed by the backend (FazzonBackend); their
        # weights only matter if a caller leaves them free, and then mix the levels.
        cv_parking_price = CategoricalIndex(
            PRICE_INDEX_NAME, {str(p): 1.0 / len(PRICE_LEVELS) for p in PRICE_LEVELS}
        )
        cv_shuttle = CategoricalIndex(
            SHUTTLE_INDEX_NAME,
            {
                str(c): 1.0 / len(SHUTTLE_CAPACITY_LEVELS)
                for c in SHUTTLE_CAPACITY_LEVELS
            },
        )
        # A39: uncertain quantities of the arrivals, sampled by the ensemble
        i_occupancy_low = _triang(
            OCCUPANCY_LOW_NAME, OCCUPANCY_LOW_RANGE, behaviour["occupancy_low"]
        )
        i_occupancy_high = _triang(
            OCCUPANCY_HIGH_NAME, OCCUPANCY_HIGH_RANGE, behaviour["occupancy_high"]
        )
        # measured: normal with the 90 % interval of fazzon_modal_shift
        k_sd = (behaviour["car_response_high"] - behaviour["car_response_low"]) / (
            2 * stats.norm.ppf(0.95)
        )
        i_vehicle_response = DistributionIndex(
            VEHICLE_RESPONSE_NAME,
            stats.norm,
            {"loc": behaviour["car_response"], "scale": max(k_sd, 1e-9)},
        )
        i_demand_level = _uniform(DEMAND_LEVEL_NAME, DEMAND_LEVEL_RANGE)
        uncertain = [
            i_vehicle_response,
            i_occupancy_low,
            i_occupancy_high,
            i_demand_level,
        ]
        policy = [cv_season, cv_weather, cv_parking_price, cv_shuttle, *uncertain]

        pv_visitors_car = ConditionalDistributionIndex(
            "visitors_car",
            [cv_season, cv_weather, cv_parking_price, *uncertain],
            car_visitors_stats,
        )
        # Trial: y axis = all visitors arriving (car + other)
        pv_visitors_total = ConditionalDistributionIndex(
            "visitors_total",
            policy,
            car_visitors_stats.total,
        )
        # Non-car arrivals: not in the graph, used by the backend to sample the scatter
        # (total = car + non-car draw of the same day)
        pv_visitors_other = ConditionalDistributionIndex(
            "visitors_other",
            policy,
            other_visitors_stats,
        )

        return cls.Inputs(
            cv_season=cv_season,
            cv_weather=cv_weather,
            cv_parking_price=cv_parking_price,
            cv_shuttle=cv_shuttle,
            pv_visitors_car=pv_visitors_car,
            pv_visitors_total=pv_visitors_total,
            pv_visitors_other=pv_visitors_other,
            # Shuttle: no bus in 2025 conditions (the Greenbus pilot only ran 28/06-20/07, D1).
            # 2026 Lakebus (A29): 2 buses x 30 seats, together every 30 min, 9-19.
            i_shuttle_buses=Index("shuttle buses", 0.0),
            i_shuttle_seats_per_bus=Index("shuttle seats per bus", 30.0),
            i_shuttle_interval_min=Index("shuttle interval (minutes)", 30.0),
            i_shuttle_start_hour=Index("shuttle first departure (hour)", 9.0),
            # end of service = last return to Pellizzano (last departure 18:30)
            i_shuttle_end_hour=Index("shuttle end of service (hour)", 19.0),
            i_shuttle_peak_end_hour=Index(
                "shuttle morning peak end (hour)", 12.0
            ),  # A35
            i_shuttle_peak_share=Index(
                "shuttle riders in the morning peak (share)", 0.6
            ),  # A35
            # Persons per vehicle, point values from the pipeline (fazzon_modal_shift, A2)
            i_occupancy_low_price=Index(
                "occupancy low price", behaviour["occupancy_low_price"]
            ),
            i_occupancy_low=i_occupancy_low,
            i_occupancy_high_price=Index(
                "occupancy high price", behaviour["occupancy_high_price"]
            ),
            i_occupancy_high=i_occupancy_high,
            i_vehicle_response=i_vehicle_response,
            i_demand_level=i_demand_level,
            # Conversion parameters
            # A8: simultaneous / daily cars, 2021 survey (mean 0.555, 0.495-0.604 by day)
            i_occupancy_ratio=_uniform(
                "parking occupancy ratio", OCCUPANCY_RATIO_RANGE
            ),
            i_xo_road_trips=Index(
                "road round-trip multiplier", 2.0
            ),  # A9: in + out = 2 passages
            # Food. A10: 0.55 of all visitors (EETRA 2022, >50 %) in 2025 conditions, where
            # 31 % arrive by other means (A3): 0.69 x 0.50 + 0.31 x 0.66 = 0.55 at the
            # centres of the ranges
            i_u_food_fraction_car=_uniform(
                "restaurant usage fraction, visitors by car", FOOD_FRACTION_CAR_RANGE
            ),
            i_u_food_fraction_other=_uniform(
                "restaurant usage fraction, visitors by other means",
                FOOD_FRACTION_OTHER_RANGE,
            ),
            i_xo_food_sittings=_uniform(
                "food seat turnover", FOOD_SITTINGS_RANGE
            ),  # A11: ~2 sittings/seat/day
            # Capacities
            # A12 — Parking: policy cap is 150 simultaneous cars.  Triangular captures
            # uncertainty about whitelist slots (30 reserved), enforcement, and
            # peak-hour variability.  Mode=200 = physical upper bound.
            i_c_parking=DistributionIndex(
                "parking capacity (simultaneous cars)",
                stats.triang,
                {"loc": 150.0, "scale": 100.0, "c": 0.5},  # mode=200, range [150, 250]
            ),
            # A13 — Road: effective daily vehicle capacity under semaphore control.
            # Peak observed 2025: 422 cars × 2 trips = 844 passages; road not saturated.
            i_c_road=DistributionIndex(
                "road capacity (vehicle passages/day)",
                stats.uniform,
                {
                    "loc": 800.0,
                    "scale": 600.0,
                },  # ASSUMPTION uniform[800, 1400] veh passages/day
            ),
            # A15 — Restaurant seats: 4 venues, 520 total (from data — no uncertainty).
            i_c_seats=Index("restaurant seats", 520.0),
        )

    def compute(self, inputs: Inputs) -> Outputs:
        """Wire the three concern sub-models from flat inputs."""
        # A29: every run goes up and comes back: 2 passages per bus and run
        i_shuttle_daily_trips = Index(
            "shuttle daily vehicle passages",
            inputs.i_shuttle_buses
            * (inputs.i_shuttle_end_hour - inputs.i_shuttle_start_hour)
            * 60.0
            / inputs.i_shuttle_interval_min
            * 2.0,
        )
        access = AccessModel(
            inputs=AccessModel.Inputs(
                pv_visitors_car=inputs.pv_visitors_car,
                cv_parking_price=inputs.cv_parking_price,
                i_occupancy_low_price=inputs.i_occupancy_low_price,
                i_occupancy_low=inputs.i_occupancy_low,
                i_occupancy_high_price=inputs.i_occupancy_high_price,
                i_occupancy_high=inputs.i_occupancy_high,
            )
        )
        parking = ParkingModel(
            inputs=ParkingModel.Inputs(
                i_cars=access.outputs.i_cars,
                i_occupancy_ratio=inputs.i_occupancy_ratio,
                i_c_parking=inputs.i_c_parking,
            )
        )
        road = RoadModel(
            inputs=RoadModel.Inputs(
                i_cars=access.outputs.i_cars,
                i_xo_road_trips=inputs.i_xo_road_trips,
                i_shuttle_daily_trips=i_shuttle_daily_trips,
                i_c_road=inputs.i_c_road,
            )
        )
        food = FoodModel(
            inputs=FoodModel.Inputs(
                pv_visitors_car=inputs.pv_visitors_car,
                pv_visitors_total=inputs.pv_visitors_total,
                i_u_food_fraction_car=inputs.i_u_food_fraction_car,
                i_u_food_fraction_other=inputs.i_u_food_fraction_other,
                i_xo_food_sittings=inputs.i_xo_food_sittings,
                i_c_seats=inputs.i_c_seats,
            )
        )

        self.constraints = [
            parking.constraint,
            road.constraint,
            food.constraint,
        ]

        # Expose scenario-accessible indexes as direct attributes.
        # These are identity-aliases of the corresponding inputs fields:
        # scenario overrides work identically on either name.
        self.i_c_parking = inputs.i_c_parking
        self.i_c_road = inputs.i_c_road
        self.i_c_seats = inputs.i_c_seats
        self.cv_parking_price = inputs.cv_parking_price
        self.cv_shuttle = inputs.cv_shuttle
        self.i_shuttle_buses = inputs.i_shuttle_buses
        self.i_shuttle_daily_trips = i_shuttle_daily_trips
        # Computed indexes, exposed for inspection.
        self.i_cars = access.outputs.i_cars
        self.i_occupancy = access.outputs.i_occupancy

        return FazzonModel.Outputs(usage_indexes=[c.usage for c in self.constraints])
