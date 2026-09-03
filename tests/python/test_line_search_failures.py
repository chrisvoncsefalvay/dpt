"""Accepted line-search states and numerical failure reporting."""

import pytest

from dpt.contracts import NumericalError, TrialDomainError
from dpt.registration import Evaluation, RecoveryPolicy, _line_search, recover_parameters


def test_line_search_returns_best_armijo_when_kink_prevents_wolfe():
    seen = []

    def objective(x):
        loss = abs(x[0] - 0.3)
        seen.append(loss)
        return Evaluation(loss, (1.0 if x[0] > 0.3 else -1.0,))

    base = Evaluation(0.7, (1.0,))
    chosen = _line_search(
        objective, (1.0,), base, (-1.0,), RecoveryPolicy(curvature=0.1, line_search_evaluations=5)
    )
    assert chosen is not None
    assert chosen[1].loss < base.loss
    assert chosen[1].loss == min(seen)


def test_trial_domain_is_recorded_but_numerical_failure_aborts():
    def objective(x):
        if x[0] <= 0:
            raise TrialDomainError("positive chart")
        return Evaluation(x[0], (1.0,))

    result = recover_parameters(objective, (0.1,), policy=RecoveryPolicy(max_iterations=1))
    assert result.rejected_trials and result.rejected_trials[0].message == "positive chart"

    def corrupt(x):
        if x[0] != 0.1:
            raise NumericalError("corrupt fixed data")
        return Evaluation(0.1, (1.0,))

    with pytest.raises(NumericalError, match="corrupt"):
        recover_parameters(corrupt, (0.1,))
