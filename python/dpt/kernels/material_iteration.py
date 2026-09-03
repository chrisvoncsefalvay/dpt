"""Resident scalar controls for one bounded projected Armijo/GLL iteration."""

import warp as wp

OPTIONS = {"fast_math": False, "fuse_fp": False, "enable_backward": False}
wp.set_module_options(OPTIONS)

# The FP64 control vector is also the compact completed host summary.
STEP = wp.constant(0)
MAPPING_STEP = wp.constant(1)
MOMENTUM = wp.constant(2)
LOSS = wp.constant(3)
DATA = wp.constant(4)
PRIOR = wp.constant(5)
MAPPING = wp.constant(6)
TOLERANCE = wp.constant(7)
NORMALISER = wp.constant(8)
INITIAL_MAPPING = wp.constant(9)
INERTIAL_T = wp.constant(10)
NEXT_T = wp.constant(11)
REFERENCE = wp.constant(12)
TRIAL_LOSS = wp.constant(13)
SLOPE = wp.constant(14)
SECANT = wp.constant(15)
PROPOSED_STEP = wp.constant(16)
TRIAL_DATA = wp.constant(17)
TRIAL_PRIOR = wp.constant(18)
CONTROL_SIZE = 19

ITERATION = wp.constant(0)
LOOP = wp.constant(1)
EVALUATE = wp.constant(2)
FATAL = wp.constant(3)
STATIONARY = wp.constant(4)
ACCEPTED = wp.constant(5)
BACKTRACKS = wp.constant(6)
ATTEMPTS = wp.constant(7)
RESTART = wp.constant(8)
EVALUATIONS = wp.constant(9)
NUMERICAL = wp.constant(10)
REFERENCE_COUNT = wp.constant(11)
EVALUATION_STARTED = wp.constant(12)
FLAG_SIZE = 13

# Each attempted proposal has step, momentum, loss/data/prior, slope,
# backtrack ordinal, failure kind (1 Armijo, 2 non-descent, 3 numerical),
# successful value-evaluation flag, acceptance flag and evaluator entry flag.
LEDGER_WIDTH = 11


@wp.struct
class IterationPolicy:
    maximum_backtracks: int
    window: int
    relative_mapping: int
    use_secant: int
    use_inertia: int
    maximum_step: wp.float64
    minimum_secant: wp.float64
    backtrack_factor: wp.float64
    armijo: wp.float64
    mapping_step: wp.float64
    relative_tolerance: wp.float64


@wp.kernel
def reset_iteration(flags: wp.array(dtype=wp.int32), controls: wp.array(dtype=wp.float64)):
    for index in range(1, 11):
        flags[index] = 0
    controls[SECANT] = wp.float64(0.0)
    controls[TRIAL_LOSS] = wp.float64(wp.inf)
    controls[SLOPE] = wp.float64(0.0)


@wp.kernel
def collect_status(
    status: wp.array(dtype=wp.int32),
    snapshot: wp.array(dtype=wp.int32),
    index: int,
    flags: wp.array(dtype=wp.int32),
    fatal_all: int,
    fixed_input: int,
):
    code = status[0]
    snapshot[index] = snapshot[index] | code
    if code != 0:
        if fatal_all != 0 or (fixed_input != 0 and (code & 1) != 0):
            flags[FATAL] = 1
        else:
            flags[NUMERICAL] = 1
    status[0] = 0


@wp.kernel
def collect_trial_status(
    status: wp.array(dtype=wp.int32),
    snapshots: wp.array(dtype=wp.int32),
    width: int,
    index: int,
    flags: wp.array(dtype=wp.int32),
    fixed_input: int,
):
    row = flags[ATTEMPTS]
    code = status[0]
    snapshots[row * width + index] = snapshots[row * width + index] | code
    # The checkpoint delivers all diagnostics but raises its first typed
    # error. Later fixed-input words must not turn an earlier rejectable
    # numerical trial into a fatal search failure.
    if code != 0 and flags[FATAL] == 0 and flags[NUMERICAL] == 0:
        if fixed_input != 0 and (code & 1) != 0:
            flags[FATAL] = 1
        else:
            flags[NUMERICAL] = 1
    status[0] = 0


@wp.kernel
def base_value(
    summary: wp.array(dtype=wp.float64),
    controls: wp.array(dtype=wp.float64),
    flags: wp.array(dtype=wp.int32),
    references: wp.array(dtype=wp.float64),
    policy: IterationPolicy,
):
    loss = summary[2]
    if not wp.isfinite(loss):
        flags[FATAL] = 2
    else:
        controls[LOSS] = loss
        controls[DATA] = summary[0]
        controls[PRIOR] = summary[1]
        flags[EVALUATIONS] = 1
        count = flags[REFERENCE_COUNT]
        if count == policy.window:
            for index in range(policy.window - 1):
                references[index] = references[index + 1]
            count -= 1
        references[count] = loss
        count += 1
        flags[REFERENCE_COUNT] = count
        reference = references[0]
        for index in range(1, count):
            reference = wp.max(reference, references[index])
        controls[REFERENCE] = reference


@wp.kernel
def relative_scale(
    maximum: wp.array(dtype=wp.float64),
    controls: wp.array(dtype=wp.float64),
    flags: wp.array(dtype=wp.int32),
    policy: IterationPolicy,
):
    scale = maximum[0]
    if not wp.isfinite(scale):
        flags[FATAL] = 3
    else:
        if scale == wp.float64(0.0):
            scale = wp.float64(1.0)
        controls[NORMALISER] = scale
        step = policy.mapping_step / scale
        controls[MAPPING_STEP] = step
        if not wp.isfinite(step) or step <= wp.float64(0.0):
            flags[FATAL] = 4


@wp.kernel
def mapping_result(
    trial_controls: wp.array(dtype=wp.float64),
    controls: wp.array(dtype=wp.float64),
    flags: wp.array(dtype=wp.int32),
    policy: IterationPolicy,
):
    mapping = trial_controls[0]
    controls[MAPPING] = mapping
    if not wp.isfinite(mapping) or not wp.isfinite(trial_controls[1]):
        flags[FATAL] = 5
    elif flags[FATAL] == 0:
        if flags[ITERATION] == 0:
            controls[INITIAL_MAPPING] = mapping
            controls[TOLERANCE] *= controls[NORMALISER]
            controls[TOLERANCE] += policy.relative_tolerance * mapping
        if mapping <= controls[TOLERANCE]:
            flags[STATIONARY] = 1


@wp.kernel
def secant_result(
    totals: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    controls: wp.array(dtype=wp.float64),
    flags: wp.array(dtype=wp.int32),
    policy: IterationPolicy,
):
    numerator = totals[0]
    denominator = totals[1]
    if (
        status[0] == 0
        and wp.isfinite(numerator)
        and wp.isfinite(denominator)
        and numerator > wp.float64(0.0)
        and denominator > wp.float64(0.0)
    ):
        step = numerator / denominator
        if wp.isfinite(step) and step > wp.float64(0.0):
            step = wp.min(policy.maximum_step, wp.max(policy.minimum_secant, step))
            controls[STEP] = step
            controls[SECANT] = step
    status[0] = 0


@wp.kernel
def begin_search(
    controls: wp.array(dtype=wp.float64),
    flags: wp.array(dtype=wp.int32),
    policy: IterationPolicy,
):
    controls[PROPOSED_STEP] = controls[STEP]
    t = controls[INERTIAL_T]
    following = wp.float64(0.5) * (
        wp.float64(1.0) + wp.sqrt(wp.float64(1.0) + wp.float64(4.0) * (t * t))
    )
    controls[NEXT_T] = following
    controls[MOMENTUM] = wp.float64(0.0)
    if policy.use_inertia != 0:
        controls[MOMENTUM] = (t - wp.float64(1.0)) / following
    if flags[FATAL] == 0 and flags[STATIONARY] == 0 and controls[STEP] != wp.float64(0.0):
        flags[LOOP] = 1


@wp.kernel
def prepare_attempt(controls: wp.array(dtype=wp.float64), flags: wp.array(dtype=wp.int32)):
    flags[EVALUATE] = 0
    flags[NUMERICAL] = 0
    flags[EVALUATION_STARTED] = 0
    controls[TRIAL_LOSS] = wp.float64(wp.inf)
    controls[TRIAL_DATA] = wp.float64(0.0)
    controls[TRIAL_PRIOR] = wp.float64(0.0)


@wp.kernel
def evaluation_started(flags: wp.array(dtype=wp.int32)):
    flags[EVALUATION_STARTED] = 1


@wp.kernel
def proposal_result(
    trial_controls: wp.array(dtype=wp.float64),
    controls: wp.array(dtype=wp.float64),
    flags: wp.array(dtype=wp.int32),
):
    slope = trial_controls[1]
    controls[SLOPE] = slope
    if not wp.isfinite(slope) or not wp.isfinite(trial_controls[0]):
        flags[NUMERICAL] = 1
    if flags[FATAL] == 0 and flags[NUMERICAL] == 0 and slope < wp.float64(0.0):
        flags[EVALUATE] = 1


@wp.kernel
def trial_value(
    summary: wp.array(dtype=wp.float64),
    controls: wp.array(dtype=wp.float64),
    flags: wp.array(dtype=wp.int32),
):
    if flags[NUMERICAL] == 0 and flags[FATAL] == 0:
        if wp.isfinite(summary[2]):
            controls[TRIAL_LOSS] = summary[2]
            controls[TRIAL_DATA] = summary[0]
            controls[TRIAL_PRIOR] = summary[1]
        else:
            flags[NUMERICAL] = 1


@wp.kernel
def decide_attempt(
    controls: wp.array(dtype=wp.float64),
    flags: wp.array(dtype=wp.int32),
    ledger: wp.array(dtype=wp.float64),
    policy: IterationPolicy,
):
    row = flags[ATTEMPTS] * LEDGER_WIDTH
    failure = int(1)  # noqa: UP018, RUF046 - Mutable Warp runtime integer.
    successful_evaluation = int(0)  # noqa: UP018, RUF046 - Mutable Warp runtime integer.
    if flags[NUMERICAL] != 0:
        failure = 3
        controls[SLOPE] = wp.float64(0.0)
    elif controls[SLOPE] >= wp.float64(0.0):
        failure = 2
    elif flags[EVALUATE] != 0 and flags[FATAL] == 0:
        successful_evaluation = 1
        flags[EVALUATIONS] += 1
    accepted = int(0)  # noqa: UP018, RUF046 - Mutable Warp runtime integer.
    loss = controls[TRIAL_LOSS]
    slope = controls[SLOPE]
    reference = controls[REFERENCE]
    if (
        flags[FATAL] == 0
        and wp.isfinite(controls[LOSS])
        and wp.isfinite(loss)
        and wp.isfinite(slope)
        and wp.isfinite(reference)
        and slope < wp.float64(0.0)
        and loss < reference
        and loss <= reference + policy.armijo * slope
    ):
        accepted = 1
        flags[ACCEPTED] = 1
    ledger[row] = controls[STEP]
    ledger[row + 1] = controls[MOMENTUM]
    ledger[row + 2] = loss
    ledger[row + 3] = controls[TRIAL_DATA]
    ledger[row + 4] = controls[TRIAL_PRIOR]
    ledger[row + 5] = slope
    ledger[row + 6] = wp.float64(flags[BACKTRACKS])
    ledger[row + 7] = wp.float64(failure)
    ledger[row + 8] = wp.float64(successful_evaluation)
    ledger[row + 9] = wp.float64(accepted)
    ledger[row + 10] = wp.float64(flags[EVALUATION_STARTED])
    flags[ATTEMPTS] += 1
    if accepted != 0 or flags[FATAL] != 0:
        flags[LOOP] = 0
    elif controls[MOMENTUM] != wp.float64(0.0):
        flags[RESTART] = failure
        controls[MOMENTUM] = wp.float64(0.0)
    else:
        controls[STEP] *= policy.backtrack_factor
        flags[BACKTRACKS] += 1
        if flags[BACKTRACKS] >= policy.maximum_backtracks or controls[STEP] == wp.float64(0.0):
            flags[LOOP] = 0


@wp.kernel
def gate(flags: wp.array(dtype=wp.int32), output: wp.array(dtype=wp.int32), kind: int):
    value = int(0)  # noqa: UP018, RUF046 - Mutable Warp runtime integer.
    if kind == 0 and flags[FATAL] == 0:
        value = 1
    elif kind == 1 and flags[FATAL] == 0 and flags[STATIONARY] == 0:
        value = 1
    elif kind == 2 and flags[EVALUATE] != 0 and flags[FATAL] == 0 and flags[NUMERICAL] == 0:
        value = 1
    elif kind == 3 and flags[ITERATION] == 0 and flags[FATAL] == 0:
        value = 1
    elif kind == 4 and flags[ITERATION] > 0 and flags[FATAL] == 0:
        value = 1
    output[0] = value


@wp.kernel
def pack(
    controls: wp.array(dtype=wp.float64),
    flags: wp.array(dtype=wp.int32),
    ledger: wp.array(dtype=wp.float64),
    base_status: wp.array(dtype=wp.int32),
    trial_status: wp.array(dtype=wp.int32),
    payload: wp.array(dtype=wp.float64),
):
    index = wp.tid()
    a = controls.shape[0]
    b = a + flags.shape[0]
    c = b + ledger.shape[0]
    d = c + base_status.shape[0]
    if index < a:
        payload[index] = controls[index]
    elif index < b:
        payload[index] = wp.float64(flags[index - a])
    elif index < c:
        payload[index] = ledger[index - b]
    elif index < d:
        payload[index] = wp.float64(base_status[index - c])
    else:
        payload[index] = wp.float64(trial_status[index - d])
