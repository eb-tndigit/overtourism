"""Molveno overtourism model definition — modular decomposition.

The model is split into four concern sub-models plus a root model.  Context
variables (``cv_*``) and presence variables (``pv_*``) are constructed
directly on the root :class:`MolvenoModel` and wired down to each concern
sub-model through its ``Inputs`` dataclass.

:class:`ParkingModel` — *Parking usage*
    **Inputs**: ``pv_tourists``, ``pv_excursionists``, ``cv_weather``,
    ``i_u_tourists_parking``, ``i_u_excursionists_parking``,
    ``i_xa_tourists_per_vehicle``, ``i_xa_excursionists_per_vehicle``,
    ``i_xo_tourists_parking``, ``i_xo_excursionists_parking``,
    ``i_c_parking``
    **Outputs**: ``i_u_parking``

:class:`BeachModel` — *Beach usage*
    **Inputs**: ``pv_tourists``, ``pv_excursionists``, ``cv_weather``,
    ``i_u_tourists_beach``, ``i_u_excursionists_beach``,
    ``i_xo_tourists_beach`` *(uncertain)*, ``i_xo_excursionists_beach``,
    ``i_c_beach``
    **Outputs**: ``i_u_beach``

:class:`AccommodationModel` — *Accommodation usage*
    **Inputs**: ``pv_tourists``,
    ``i_u_tourists_accommodation``, ``i_xa_tourists_accommodation``,
    ``i_c_accommodation``
    **Outputs**: ``i_u_accommodation``

:class:`FoodModel` — *Food-service usage*
    **Inputs**: ``pv_tourists``, ``pv_excursionists``, ``cv_weather``,
    ``i_u_tourists_food``, ``i_u_excursionists_food_rainy_weather``,
    ``i_u_excursionists_food_good_cool_weather``,
    ``i_xa_visitors_food``, ``i_xo_visitors_food``, ``i_c_food``
    **Outputs**: ``i_u_food``

:class:`MolvenoModel` — *Root, owns CVs, PVs, and all* ``i_*`` *defaults*
    Creates the context variables (weekday, weather and presence estimator,
    :class:`~civic_digital_twins.dt_model.CategoricalIndex`; the day of the
    summer, a :class:`~civic_digital_twins.dt_model.DistributionIndex` over the
    period of the scenario), the two presence variables (read from the
    data-preparation pipeline, see ``molveno_presence_stats``), and all ``i_*``
    indexes with their default values, then passes them to the four concern
    sub-models.  Retains the domain
    attributes (``cvs``, ``pvs``, ``constraints``) required by
    :class:`~dt_model.CrossProductEnsemble`.

Design rules:

* **All** ``i_*`` parameters are ``Inputs`` to the sub-model that uses
  them, including uncertain ``DistributionIndex`` values.  The default
  values are created by :class:`MolvenoModel` and passed down via
  constructors.  A caller who wants to override a parameter simply
  supplies a different index object at construction time.
* Context variables (``cv_*``) and presence variables (``pv_*``) are
  attributes of :class:`MolvenoModel` and are wired as ``Inputs`` to the
  concern sub-models that consume them.
* Each concern sub-model's ``Outputs`` contains only the usage-formula
  index (``i_u_*``).  Capacity indexes (``i_c_*``) remain as ``Inputs``
  because they are parameters, not computed results.
* Each concern sub-model stores its
  :class:`~overtourism.model.molveno.molveno_model.Constraint` as a
  plain instance attribute (``self.constraint``) because
  :class:`~overtourism.model.molveno.molveno_model.Constraint` is not a
  :class:`~dt_model.model.index.GenericIndex` and must not appear inside
  an :class:`~dt_model.model.model.IOProxy`.
* :class:`MolvenoModel` subclasses :class:`~dt_model.model.model.Model`
  directly and exposes ``.cvs``, ``.pvs``, and ``.constraints`` attributes
  so that :class:`~dt_model.CrossProductEnsemble`
  and the evaluation code can consume them without modification.
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
    expose,
    graph,
    inputs,
    outputs,
)
from scipy import stats

from .molveno_presence_stats import (
    DAY_INDEX_NAME,
    ESTIMATOR_INDEX_NAME,
    ESTIMATORS,
    DAY_TYPE_WEIGHTS,
    presence_inputs,
)

# ---------------------------------------------------------------------------
# Constraint
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Constraint:
    """Named pairing of a usage formula index and a capacity index.

    Both *usage* and *capacity* are formula-mode or distribution-backed
    :class:`~dt_model.model.index.Index` objects, so the entire constraint is
    expressed in terms of :class:`~dt_model.model.index.GenericIndex` — no
    engine-layer types appear in the public API.

    Identity-based hashing (``eq=False``) keeps ``Constraint`` objects usable
    as dict keys, matching the convention used by ``graph.Node`` and
    ``GenericIndex``.
    """

    name: str
    usage: Index  # formula-mode Index wrapping the usage expression
    capacity: Index  # constant, distribution-backed, or formula-mode Index


# ---------------------------------------------------------------------------
# ParkingModel
# ---------------------------------------------------------------------------


@define("Parking")
class ParkingModel(Model):
    """Concern sub-model — parking usage.

    All parameters (usage factors, conversion factors, capacity) are
    received as ``Inputs`` so that callers can override any default.
    :class:`MolvenoModel` creates the indexes with their default values and
    passes them in.

    The usage formula ``i_u_parking`` is the single contractual ``Output``.
    The :class:`~overtourism.model.molveno.molveno_model.Constraint` is
    stored as a plain instance attribute ``self.constraint``.

    Attributes
    ----------
    constraint : Constraint
        The parking constraint (usage / capacity pair).
    """

    @inputs
    class Inputs:
        """Contractual inputs of :class:`ParkingModel`."""

        pv_tourists: ConditionalDistributionIndex
        pv_excursionists: ConditionalDistributionIndex
        cv_weather: CategoricalIndex
        i_u_tourists_parking: Index
        i_u_excursionists_parking: DistributionIndex
        i_xa_tourists_per_vehicle: Index
        i_xa_excursionists_per_vehicle: Index
        i_xo_tourists_parking: Index
        i_xo_excursionists_parking: Index
        i_c_parking: DistributionIndex

    @outputs
    class Outputs:
        """Contractual outputs of :class:`ParkingModel`."""

        i_u_parking: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute parking usage from inputs."""
        i_u_parking = Index(
            "parking usage",
            inputs.pv_tourists
            * inputs.i_u_tourists_parking
            / (inputs.i_xa_tourists_per_vehicle * inputs.i_xo_tourists_parking)
            + inputs.pv_excursionists
            * inputs.i_u_excursionists_parking
            / (
                inputs.i_xa_excursionists_per_vehicle
                * inputs.i_xo_excursionists_parking
            ),
        )
        # Constraint stored as a plain attribute — not a GenericIndex.
        self.constraint = Constraint(
            name="parking", usage=i_u_parking, capacity=inputs.i_c_parking
        )
        return ParkingModel.Outputs(i_u_parking=i_u_parking)


# ---------------------------------------------------------------------------
# BeachModel
# ---------------------------------------------------------------------------


@define("Beach")
class BeachModel(Model):
    """Concern sub-model — beach usage.

    All parameters (usage factors, rotation factors, capacity) are received
    as ``Inputs``.  The uncertain rotation factor ``i_xo_tourists_beach`` is
    passed in from :class:`MolvenoModel` so it appears in the root
    ``model.indexes`` and is sampled by
    :class:`~dt_model.CrossProductEnsemble`.

    Attributes
    ----------
    constraint : Constraint
        The beach constraint (usage / capacity pair).
    """

    @inputs
    class Inputs:
        """Contractual inputs of :class:`BeachModel`."""

        pv_tourists: ConditionalDistributionIndex
        pv_excursionists: ConditionalDistributionIndex
        cv_weather: CategoricalIndex
        i_u_tourists_beach: Index
        i_u_excursionists_beach: Index
        i_xo_tourists_beach: DistributionIndex
        i_xo_excursionists_beach: Index
        i_c_beach: DistributionIndex

    @outputs
    class Outputs:
        """Contractual outputs of :class:`BeachModel`."""

        i_u_beach: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute beach usage from inputs."""
        i_u_beach = Index(
            "beach usage",
            inputs.pv_tourists * inputs.i_u_tourists_beach / inputs.i_xo_tourists_beach
            + inputs.pv_excursionists
            * inputs.i_u_excursionists_beach
            / inputs.i_xo_excursionists_beach,
        )
        # Constraint stored as a plain attribute — not a GenericIndex.
        self.constraint = Constraint(
            name="beach", usage=i_u_beach, capacity=inputs.i_c_beach
        )
        return BeachModel.Outputs(i_u_beach=i_u_beach)


# ---------------------------------------------------------------------------
# AccommodationModel
# ---------------------------------------------------------------------------


@define("Accommodation")
class AccommodationModel(Model):
    """Concern sub-model — accommodation usage.

    Attributes
    ----------
    constraint : Constraint
        The accommodation constraint (usage / capacity pair).
    """

    @inputs
    class Inputs:
        """Contractual inputs of :class:`AccommodationModel`."""

        pv_tourists: ConditionalDistributionIndex
        i_u_tourists_accommodation: DistributionIndex
        i_xa_tourists_accommodation: Index
        i_c_accommodation: DistributionIndex

    @outputs
    class Outputs:
        """Contractual outputs of :class:`AccommodationModel`."""

        i_u_accommodation: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute accommodation usage from inputs."""
        i_u_accommodation = Index(
            "accommodation usage",
            inputs.pv_tourists
            * inputs.i_u_tourists_accommodation
            / inputs.i_xa_tourists_accommodation,
        )
        # Constraint stored as a plain attribute — not a GenericIndex.
        self.constraint = Constraint(
            name="accommodation",
            usage=i_u_accommodation,
            capacity=inputs.i_c_accommodation,
        )
        return AccommodationModel.Outputs(i_u_accommodation=i_u_accommodation)


# ---------------------------------------------------------------------------
# FoodModel
# ---------------------------------------------------------------------------


@define("Food")
class FoodModel(Model):
    """Concern sub-model — food-service usage.

    The excursionist usage factor depends on the weather: it is
    ``i_u_excursionists_food_rainy_weather`` when ``cv_weather`` is ``"rainy"``
    and ``i_u_excursionists_food_good_cool_weather`` when it is
    ``"good"`` or ``"cool"``.

    Attributes
    ----------
    constraint : Constraint
        The food-service constraint (usage / capacity pair).
    """

    @inputs
    class Inputs:
        """Contractual inputs of :class:`FoodModel`."""

        pv_tourists: ConditionalDistributionIndex
        pv_excursionists: ConditionalDistributionIndex
        cv_weather: CategoricalIndex
        i_u_tourists_food: Index
        i_u_excursionists_food_rainy_weather: Index
        i_u_excursionists_food_good_cool_weather: Index
        i_xa_visitors_food: Index
        i_xo_visitors_food: Index
        i_c_food: DistributionIndex

    @outputs
    class Outputs:
        """Contractual outputs of :class:`FoodModel`."""

        i_u_food: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute food-service usage from inputs."""
        i_u_excursionists_food = Index(
            "excursionist food service usage factor",
            graph.piecewise(
                (
                    inputs.i_u_excursionists_food_rainy_weather,
                    inputs.cv_weather == "rainy",
                ),
                (inputs.i_u_excursionists_food_good_cool_weather, True),
            ),
        )
        i_u_food = Index(
            "food usage",
            (
                inputs.pv_tourists * inputs.i_u_tourists_food
                + inputs.pv_excursionists * i_u_excursionists_food
            )
            / (inputs.i_xa_visitors_food * inputs.i_xo_visitors_food),
        )
        # Constraint stored as a plain attribute — not a GenericIndex.
        self.constraint = Constraint(
            name="food", usage=i_u_food, capacity=inputs.i_c_food
        )
        return FoodModel.Outputs(i_u_food=i_u_food)


# ---------------------------------------------------------------------------
# MolvenoModel  (root)
# ---------------------------------------------------------------------------


@define("base model")
class MolvenoModel(Model):
    """Root overtourism model that wires the four concern sub-models.

    All domain parameters are declared as ``Inputs``; supply defaults via
    :meth:`default_inputs` or override individual fields with
    :func:`dataclasses.replace`::

        m = MolvenoModel(inputs=MolvenoModel.default_inputs())
    """

    @inputs
    class Inputs:
        """All domain parameters of :class:`MolvenoModel`."""

        # Context variables
        cv_weekday: CategoricalIndex
        cv_weather: CategoricalIndex
        i_day_of_summer: DistributionIndex
        cv_presence_estimator: CategoricalIndex
        # Presence distributions
        pv_tourists: ConditionalDistributionIndex
        pv_excursionists: ConditionalDistributionIndex
        # Distribution-backed uncertainty parameters
        i_c_parking: DistributionIndex
        i_c_beach: DistributionIndex
        i_c_accommodation: DistributionIndex
        i_c_food: DistributionIndex
        i_xo_tourists_beach: DistributionIndex
        i_u_excursionists_parking: DistributionIndex
        # Parking parameters
        i_u_tourists_parking: Index
        i_booked_parking_spaces: Index
        i_xa_tourists_per_vehicle: Index
        i_xa_excursionists_per_vehicle: Index
        i_xo_tourists_parking: Index
        i_xo_excursionists_parking: Index
        # Beach parameters
        i_u_tourists_beach: Index
        i_u_excursionists_beach: Index
        i_xo_excursionists_beach: Index
        # Accommodation parameters
        i_u_tourists_accommodation: DistributionIndex
        i_xa_tourists_accommodation: Index
        # Food parameters
        i_u_tourists_food: Index
        i_u_excursionists_food_rainy_weather: Index
        i_u_excursionists_food_good_cool_weather: Index
        i_xa_visitors_food: Index
        i_xo_visitors_food: Index
        # Presence-transformation parameters
        i_p_tourists_reduction_factor: Index
        i_p_excursionists_reduction_factor: Index
        i_p_tourists_saturation_level: Index
        i_p_excursionists_saturation_level: Index

    @outputs
    class Outputs:
        """Contractual outputs of :class:`MolvenoModel`."""

        usage_indexes: list[GenericIndex]

    @expose
    class Expose:
        """Sub-model output proxies for inspection."""

        parking: ParkingModel.Outputs
        beach: BeachModel.Outputs
        accommodation: AccommodationModel.Outputs
        food: FoodModel.Outputs

    @classmethod
    def default_inputs(cls) -> Inputs:
        """Return the default domain inputs for all parameters.

        Pass to :class:`MolvenoModel` or override individual fields with
        :func:`dataclasses.replace`::

            m = MolvenoModel(inputs=MolvenoModel.default_inputs())
        """
        # Statistics read from the data-preparation pipeline (molveno_presence_stats).
        tables, weather, tourist_presences_stats, excursionist_presences_stats = (
            presence_inputs()
        )
        # A14: Monday-Friday, Saturday, Sunday, weighted by their days in the week
        cv_weekday = CategoricalIndex("weekday", dict(DAY_TYPE_WEIGHTS))
        cv_weather = CategoricalIndex("weather", weather)
        # D9: the day of the summer, uniform over the period of the scenario (default: the
        # whole summer); the presence estimator is fixed by the backend.
        i_day_of_summer = DistributionIndex(
            DAY_INDEX_NAME, stats.uniform, {"loc": 0.0, "scale": float(tables.n_days)}
        )
        cv_presence_estimator = CategoricalIndex(
            ESTIMATOR_INDEX_NAME, {e: 1.0 / len(ESTIMATORS) for e in ESTIMATORS}
        )
        parents = [cv_weekday, cv_weather, i_day_of_summer, cv_presence_estimator]
        pv_tourists = ConditionalDistributionIndex(
            "tourists", parents, tourist_presences_stats
        )
        pv_excursionists = ConditionalDistributionIndex(
            "excursionists", parents, excursionist_presences_stats
        )
        return cls.Inputs(
            cv_weekday=cv_weekday,
            cv_weather=cv_weather,
            i_day_of_summer=i_day_of_summer,
            cv_presence_estimator=cv_presence_estimator,
            pv_tourists=pv_tourists,
            pv_excursionists=pv_excursionists,
            # Distribution-backed uncertainty parameters
            # A7: the public car parks of the comune, 713 car spaces, 529 paid (Comune di
            # Molveno, 2024)
            i_c_parking=DistributionIndex(
                "parking capacity", stats.uniform, {"loc": 683.0, "scale": 60.0}
            ),
            # A10: no source; from the 2023 peak (4,676 at once, no saturation) upwards
            i_c_beach=DistributionIndex(
                "beach capacity", stats.uniform, {"loc": 4700.0, "scale": 800.0}
            ),
            # A11: hotel and extra-hotel beds of the comune, 3,429 (2023) - 3,503 (2025), ISPAT
            i_c_accommodation=DistributionIndex(
                "accommodation capacity",
                stats.uniform,
                {"loc": 3429.0, "scale": 74.0},
            ),
            # A12: seats of the restaurants and bars of the lake area, village and Pradel included
            i_c_food=DistributionIndex(
                "food service capacity",
                stats.uniform,
                {"loc": 2768.0, "scale": 115.0},
            ),
            # A10: beach entries / people at once, 1.6-2.4 (BS4 counters 2023)
            i_xo_tourists_beach=DistributionIndex(
                "tourists on beach rotation factor",
                stats.uniform,
                {"loc": 1.6, "scale": 0.8},
            ),
            # A8: day visitors arriving by car, 0.78 (survey 2023; 0.31 cars per day visitor
            # entering the beach area x 2.5 per car)
            i_u_excursionists_parking=DistributionIndex(
                "excursionist parking usage factor",
                stats.uniform,
                {"loc": 0.70, "scale": 0.16},
            ),
            # Parking parameters
            i_u_tourists_parking=Index("tourist parking usage factor", 0.02),
            i_xa_tourists_per_vehicle=Index(
                "tourists per vehicle allocation factor", 2.5
            ),
            i_xa_excursionists_per_vehicle=Index(
                "excursionists per vehicle allocation factor", 2.5
            ),
            i_xo_tourists_parking=Index("tourists in parking rotation factor", 1.05),
            # A9: paid stays of 5.1 h on the busy days of 2023, over the paid day 8-18
            i_xo_excursionists_parking=Index(
                "excursionists in parking rotation factor", 2.0
            ),
            # D15: spaces under compulsory booking (one car per space and day); default: the
            # Ischia car park, booking-only since 2025
            i_booked_parking_spaces=Index("booked parking spaces", 195.0),
            # Beach parameters (A10: BS4 counters 2023, entries per day ~ 0.35 x tourists +
            # 0.78 x excursionists, 2.1 entries per place at once)
            i_u_tourists_beach=Index("tourist beach usage factor", 0.35),
            i_u_excursionists_beach=Index("excursionist beach usage factor", 0.78),
            i_xo_excursionists_beach=Index(
                "excursionists on beach rotation factor", 2.1
            ),
            # Accommodation parameters (A11: ISPAT overnight stays / Vodafone tourists, 10-90 %
            # of the summer days 2023: 0.64-0.93)
            i_u_tourists_accommodation=DistributionIndex(
                "tourist accommodation usage factor",
                stats.uniform,
                {"loc": 0.65, "scale": 0.25},
            ),
            i_xa_tourists_accommodation=Index(
                "tourists per accommodation allocation factor", 1.0
            ),
            # Food parameters
            i_u_tourists_food=Index("tourist food service usage factor", 0.20),
            i_u_excursionists_food_rainy_weather=Index(
                "excursionist food service usage factor (rainy weather)", 0.80
            ),
            i_u_excursionists_food_good_cool_weather=Index(
                "excursionist food service usage factor (good / cool weather)",
                0.40,
            ),
            i_xa_visitors_food=Index("visitors in food service allocation factor", 0.9),
            i_xo_visitors_food=Index("visitors in food service rotation factor", 2.0),
            # Presence-transformation parameters
            i_p_tourists_reduction_factor=Index("tourists reduction factor", 1.0),
            i_p_excursionists_reduction_factor=Index(
                "excursionists reduction factor", 1.0
            ),
            i_p_tourists_saturation_level=Index("tourists saturation level", 10000),
            i_p_excursionists_saturation_level=Index(
                "excursionists saturation level", 10000
            ),
        )

    def compute(self, inputs: Inputs) -> tuple[Outputs, Expose]:
        """Wire concern sub-models from inputs."""
        parking = ParkingModel(
            inputs=ParkingModel.Inputs(
                pv_tourists=inputs.pv_tourists,
                pv_excursionists=inputs.pv_excursionists,
                cv_weather=inputs.cv_weather,
                i_u_tourists_parking=inputs.i_u_tourists_parking,
                i_u_excursionists_parking=inputs.i_u_excursionists_parking,
                i_xa_tourists_per_vehicle=inputs.i_xa_tourists_per_vehicle,
                i_xa_excursionists_per_vehicle=inputs.i_xa_excursionists_per_vehicle,
                i_xo_tourists_parking=inputs.i_xo_tourists_parking,
                i_xo_excursionists_parking=inputs.i_xo_excursionists_parking,
                i_c_parking=inputs.i_c_parking,
            )
        )
        beach = BeachModel(
            inputs=BeachModel.Inputs(
                pv_tourists=inputs.pv_tourists,
                pv_excursionists=inputs.pv_excursionists,
                cv_weather=inputs.cv_weather,
                i_u_tourists_beach=inputs.i_u_tourists_beach,
                i_u_excursionists_beach=inputs.i_u_excursionists_beach,
                i_xo_tourists_beach=inputs.i_xo_tourists_beach,
                i_xo_excursionists_beach=inputs.i_xo_excursionists_beach,
                i_c_beach=inputs.i_c_beach,
            )
        )
        accommodation = AccommodationModel(
            inputs=AccommodationModel.Inputs(
                pv_tourists=inputs.pv_tourists,
                i_u_tourists_accommodation=inputs.i_u_tourists_accommodation,
                i_xa_tourists_accommodation=inputs.i_xa_tourists_accommodation,
                i_c_accommodation=inputs.i_c_accommodation,
            )
        )
        food = FoodModel(
            inputs=FoodModel.Inputs(
                pv_tourists=inputs.pv_tourists,
                pv_excursionists=inputs.pv_excursionists,
                cv_weather=inputs.cv_weather,
                i_u_tourists_food=inputs.i_u_tourists_food,
                i_u_excursionists_food_rainy_weather=(
                    inputs.i_u_excursionists_food_rainy_weather
                ),
                i_u_excursionists_food_good_cool_weather=(
                    inputs.i_u_excursionists_food_good_cool_weather
                ),
                i_xa_visitors_food=inputs.i_xa_visitors_food,
                i_xo_visitors_food=inputs.i_xo_visitors_food,
                i_c_food=inputs.i_c_food,
            )
        )

        self.constraints = [
            parking.constraint,
            beach.constraint,
            accommodation.constraint,
            food.constraint,
        ]

        return (
            MolvenoModel.Outputs(usage_indexes=[c.usage for c in self.constraints]),
            MolvenoModel.Expose(
                parking=parking.outputs,
                beach=beach.outputs,
                accommodation=accommodation.outputs,
                food=food.outputs,
            ),
        )
