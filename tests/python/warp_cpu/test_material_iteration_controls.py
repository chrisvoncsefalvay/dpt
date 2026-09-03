"""CPU resident iteration controls using the shared scientific cases."""

import pytest

from helpers.material_iteration_cases import (
    test_armijo_rejection_and_geometric_growth_have_original_rounding as test_armijo_rejection_and_geometric_growth_have_original_rounding,  # noqa: E501
)
from helpers.material_iteration_cases import (
    test_gll_reference_admits_ascent_below_actual_window_maximum as test_gll_reference_admits_ascent_below_actual_window_maximum,  # noqa: E501
)
from helpers.material_iteration_cases import (
    test_inertial_restart_retries_same_alpha_without_backtrack as test_inertial_restart_retries_same_alpha_without_backtrack,  # noqa: E501
)
from helpers.material_iteration_cases import (
    test_pending_trial_errors_prevent_evaluator_entry as test_pending_trial_errors_prevent_evaluator_entry,  # noqa: E501
)
from helpers.material_iteration_cases import (
    test_relative_mapping_uses_initial_gradient_scale_and_combined_tolerance as test_relative_mapping_uses_initial_gradient_scale_and_combined_tolerance,  # noqa: E501
)
from helpers.material_iteration_cases import (
    test_resident_coupled_metric_preserves_certificate as test_resident_coupled_metric_preserves_certificate,  # noqa: E501
)
from helpers.material_iteration_cases import (
    test_resident_projector_preserves_eager_arithmetic as test_resident_projector_preserves_eager_arithmetic,  # noqa: E501
)
from helpers.material_iteration_cases import (
    test_secant_is_bounded_proposal_or_preserves_geometric_step as test_secant_is_bounded_proposal_or_preserves_geometric_step,  # noqa: E501
)
from helpers.material_iteration_cases import (
    test_trial_status_distinguishes_fixed_input_from_rejectable_numerics as test_trial_status_distinguishes_fixed_input_from_rejectable_numerics,  # noqa: E501
)


@pytest.fixture
def device() -> str:
    return "cpu"
