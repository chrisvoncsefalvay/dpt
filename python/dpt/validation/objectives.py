"""Independent Decimal objective references for explicitly supplied scalar inputs."""

from __future__ import annotations

from decimal import Decimal, localcontext


def scalar_objective(
    prediction: float, observation: float, *, poisson: bool
) -> tuple[Decimal, Decimal]:
    """J(p,y)=(p-y)²/2 or J(p,y)=p-y+y log(y/p) for Poisson counts.

    Return the scalar objective and its derivative with respect to prediction:
    p-y for least squares or 1-y/p for Poisson counts. ``prediction`` and
    ``observation`` are scalar Python floats interpreted as their exact binary64
    values, and the returned pair contains 100-digit ``Decimal`` values. The
    observation is fixed and only prediction is differentiated.

    Least squares accepts values in matching signal units and returns squared
    signal units and signal units. Poisson inputs use expected and observed
    photon counts, with a non-negative integer observation. At y=0 its objective
    and derivative are p and 1, including p=0, while y>0 requires p>0.

        >>> from dpt.validation.objectives import scalar_objective
        >>> tuple(float(v) for v in scalar_objective(3., 1., poisson=False))
        (2.0, 2.0)
        >>> tuple(float(v) for v in scalar_objective(2., 0., poisson=True))
        (2.0, 1.0)

    """
    with localcontext() as context:
        context.prec = 100
        predicted = Decimal.from_float(prediction)
        observed = Decimal.from_float(observation)
        if poisson:
            if predicted < 0 or observed < 0 or observed != observed.to_integral_value():
                raise ValueError("invalid counting-domain input")
            if observed == 0:
                return predicted, Decimal(1)
            if predicted == 0:
                raise ValueError("positive observed count has zero predicted probability")
            return (
                predicted - observed + observed * (observed / predicted).ln(),
                Decimal(1) - observed / predicted,
            )
        difference = predicted - observed
        return difference * difference / 2, difference
