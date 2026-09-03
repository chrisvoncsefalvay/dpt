# Fitting parameters

Use `recover_parameters` to minimise a deterministic objective, or `recover_expected_signal` to fit the mean of a stochastic signal. Keep the parameter chart and observation fixed throughout the solve.

## Deterministic recovery

`dpt.registration.recover_parameters` accepts an evaluator returning `Evaluation(loss, gradient)`. Its parameters and gradient are tuples in a dimensionless chart. The evaluator must finish its device status checks before returning a value.

The following host-only calculation exercises the controller without preparing CUDA:

```python
from dpt.recovery import Evaluation, RecoveryPolicy, recover_parameters, solver_termination

target = (0.25, -0.5)
events = []

def evaluate(parameters):
    residual = tuple(x - y for x, y in zip(parameters, target, strict=True))
    return Evaluation(0.5 * sum(r * r for r in residual), residual)

result = recover_parameters(
    evaluate,
    (0.0, 0.0),
    policy=RecoveryPolicy(gradient_tolerance=1e-10),
    observe_lifecycle=events.append,
)
assert result.stationary
assert events[0].status == "initial"
assert events[-1].status == "terminal"
assert solver_termination(result).category == "stationary"
```

The limited-memory solver seeks strong Wolfe conditions. If its search ends before meeting the curvature condition, it can retain the best checked point satisfying strict decrease and Armijo. Curvature pairs still pass a separate numerical guard. A search that exhausts its budget without a suitable point leaves the accepted parameters unchanged.

`RecoveryResult.reason` separates gradient stationarity, small steps, loss stagnation and resource limits. Its `stationary` property is true only for the gradient criterion. Assess parameter error against an independent reference when evaluating a fit.

## Observe solver progress

Pass `observe_lifecycle` to a recovery driver or `MaterialReconstruction.solve`
to receive immutable `SolverEvent` records. The first event is `initial`, or
`resumed` after a checkpoint's accepted state has been verified. Subsequent
events report accepted or rejected controller steps and one terminal state.
Iteration zero is the verified starting state. Later iteration numbers count
controller steps, including rejected steps.

Each record identifies the solver, iteration, accepted-step count and last
accepted objective. Rejection retains that objective and accepted-step count,
with its cause in `detail`. Stochastic recovery reports `objective=None`
because its acceptance tests estimate objective changes. Its terminal event
follows the held-out validation check.

`solver_termination(result)` reads a controller result or material solver
report and returns the original reason with a common category. For a pose or
spectral-pose wrapper, pass `result.optimisation`.

| Category | Meaning |
|---|---|
| `stationary` | A deterministic gradient criterion has been met |
| `stagnation` | A step, objective change or search radius reached its limit |
| `budget` | The iteration, evaluation or history budget was exhausted |
| `cancelled` | The caller cancelled the solve |
| `failed` | Search, damping, evaluation or exception failure |
| `uncertainty` | A stochastic gradient band or unresolved sampling decision |

Lifecycle reporting exports scalar state and adds no device work or field
readback. Observer return values are ignored. A raised observer
exception aborts the call without retrying the observer. Exceptions carry the last committed scalar state in
`error.solver_lifecycle`. The original exception still propagates.

Export a checkpoint explicitly through the solver's existing checkpoint
interface. A resumed solve verifies the accepted state before notifying its
observer. Checkpoint identity, chart, policy and oracle checks remain part of
that interface. The [lifecycle API](../reference/api-recovery.md#solver-lifecycle)
lists the event fields and termination helpers.

Supply `objective_id` to `MaterialReconstruction.solve` to retain checkpoint
state after each accepted step. For a completed solve without an ID, supply it
explicitly to `checkpoint(objective_id)`.

## Pose and calibration coordinates

`PoseChart` combines a fixed object-to-world anchor with six physical scales. Coordinates are ordered translation x/y/z followed by rotation x/y/z. Scales carry millimetres and radians respectively. Changing the anchor or scales starts a new solve because existing curvature pairs belong to the old chart. See [geometry conventions](../concepts/geometry.md).

`PrimaryPoseEvaluator` composes projection, primary image formation and the objective on one prepared CPU device or owning CUDA stream. Its attenuation field, observation, weights and chart remain immutable during recovery. It defaults to FP64 intermediates, while the attenuation field remains FP32. An evaluator owns reusable scratch; use a separate evaluator for each concurrent recovery. After recovery, evaluate the returned parameters again before exporting an image: the last scratch contents may belong to a rejected trial.

The controller records and rejects `TrialDomainError`. Other `NumericalError` exceptions abort the solve, as do corrupt fixed inputs and incomplete transport histories. Pose rotation-radius and some spectral calibration-domain checks raise `NumericalError` and terminate recovery. Choose chart scales and initial parameters within the declared domain. Preserve these exceptions when composing an evaluator. Only `TrialDomainError` permits trial rejection.

## Use residual sensitivities

For a squared residual objective, Gauss–Newton uses `J.T @ J` to propose a step.
Levenberg–Marquardt adds `damping * I` and adjusts damping from the measured
reduction relative to the quadratic model. Choose either with
`PoseLeastSquaresPolicy.method`.

`recover_least_squares` accepts `NormalEquations(loss, gradient, gauss_newton)`
in one fixed parameter chart. The following two-coordinate example has an
identity Jacobian and a constrained optimum:

```python
from dpt.pose_optimization import (
    NormalEquations, ParameterBounds, PoseLeastSquaresPolicy, recover_least_squares,
)

def normal_equations(parameters):
    residual = (parameters[0] - 1.0, parameters[1] + 0.25)
    return NormalEquations(
        0.5 * sum(value * value for value in residual),
        residual,
        ((1.0, 0.0), (0.0, 1.0)),
    )

bounded = recover_least_squares(
    normal_equations, (0.0, 0.0),
    bounds=ParameterBounds((-0.5, -0.5), (0.5, 0.5)),
    policy=PoseLeastSquaresPolicy(gradient_tolerance=1e-10),
)
assert bounded.stationary
assert abs(bounded.parameters[0] - 0.5) < 1e-10
assert abs(bounded.parameters[1] + 0.25) < 1e-10
```

For primary images, prepare `PrimaryPoseLeastSquaresEvaluator` from one or
several `PrimaryPoseProblem` views and a shared `PoseChart`. It computes pose
sensitivities, residual products and normal matrices on the owning stream.
Each view transfers 43 FP64 control values: one loss, six gradient coordinates
and a six-by-six normal matrix, together with status words. The image arrays
remain resident.

Each view keeps its declared sum/mean reduction and weights. The multi-view
objective adds those view objectives. This adapter accepts unmasked squared
error in counts or log transmission.

`ParameterBounds` supplies inclusive chart-coordinate bounds. Its feasible
active-set solve retains coupling between free coordinates, and its stopping
test uses the projected gradient. Translation bounds from `ParameterBounds.pose_translation`
refer to the scaled Lie-chart translation coordinates in mm. Physical gain and
exposure bounds become logarithmic coordinate bounds in a spectral fit.

For an evaluator supplying only loss and gradient, use
`recover_bounded_parameters` with `BoundedGradientPolicy`. It uses an identity
proposal metric. For a scalar line search, `cubic_strong_wolfe` supplies
safeguarded cubic bracket/zoom interpolation and checked decrease/curvature
conditions.

## Capture material solver iterations

Use `MaterialReconstructionSettings(execution="graph")` on CUDA to capture a
complete bounded iteration: the accepted-state gradient, projected
proposal, Armijo/GLL search and candidate decision. The graph uses resident
buffers and returns one packed control record per iteration. It requires CUDA
toolkit and driver 12.4 or newer.

Prepare the solver's arrays and penalty before the solve, and keep their
bindings fixed. The graph applies the same constraints, step policies and
stopping criteria as eager execution. Initial and rejection observers run
before candidate fields are committed. Accepted observers, callbacks and
checkpoint writers run after that commit. Explicit image and field exports
still synchronise with the owning stream.

See the iteration driver in `python/dpt/material_iteration_graph.py`
and resident controls in `python/dpt/kernels/material_iteration.py`
for the implementation, and
[§12.6 of the book](https://photontransport.com/chapters/reconstruction/#126-extend-reconstruction-to-spectral-material-decomposition)
for the material reconstruction model.

## Add a custom material penalty

`MaterialReconstruction` accepts a prepared `ResidentPenalty` callback. It
borrows the FP32 material-major fields, writes one FP64 loss and adds its
Euclidean derivative to the supplied gradient when requested. Prepare the
callback's scratch buffers before the solve and enqueue work on the supplied
context stream. Write the loss on every call, preserve the input fields and
report arithmetic failures through the supplied status word.

For graph execution, implement `GraphResidentPenalty` and set `graph_safe`
to exactly `True`. Keep the callback, its buffers and launch bindings alive
and fixed. Calls enqueue resident work with no allocation, host export,
synchronisation or data-dependent Python branching.

To use Fisher scaling, implement `FisherResidentPenalty.add_fisher_curvature`.
It receives an initially zero FP64 scalar array of one value per voxel. Add
nonnegative values `k[v]` satisfying

```text
penalty Hessian <= diag(k[v] * minimum_metric_eigenvalue)
```

Repeat each voxel's bound over materials and include all spatial and material
couplings. This makes the block metric `k[v] * H` a majorant for the custom
smooth penalty. The callback's values join data and built-in penalty curvature
before flooring and normalisation. For a quadratic penalty, absolute Hessian
row sums divided by the supplied minimum metric eigenvalue give one suitable
bound.

Use the [TV/L1 proximal primitives](regularization.md) in a composite proximal
loop. Their nonsmooth subgradients do not define a smooth Hessian majorant.
See the [material reconstruction API](../reference/api-recovery.md#dptmaterial_reconstruction)
for the callback signatures and
[§12.6 of the book](https://photontransport.com/chapters/reconstruction/#126-extend-reconstruction-to-spectral-material-decomposition)
for the material metric and Fisher row bound.

## Recovery of an expected signal

`recover_expected_signal(oracle, initial, *, seed, policy=None)` minimises

```text
loss(theta) = 0.5 * sum_p weights[p]
                        * (E[score[p](theta)] - observation[p])**2
```

The oracle supplies a mean from one history batch and a mean derivative from an independent batch, whose product estimates the gradient of this objective. Candidate and incumbent may reuse common random numbers within each factor. The two factors remain independent. Replicates also require independent identities. [Transport contracts](../reference/transport.md) explain what those declarations cover.

`prepare_transport_inverse` supplies this oracle for a fixed `ParallelBeam` and selected log material-density/log source-amplitude parameters. Coordinates are absolute logarithms of positive scales. For example, coordinate `log(2)` sets the density scale to two. Inactive materials retain their base values. Moving geometry and changing source or scattering distributions are outside this oracle's derivative contract.

Set `local_model=True` during preparation and `StochasticPolicy(proposal="quadratic")` to use a dense positive-semidefinite proposal metric. Preparation supports at most 16 parameters and defaults to a 256 MiB additional memory budget. The estimated Gram matrix is sampling-biased and is used only as a proposal metric. Fresh acceptance histories decide whether its step is useful.

## Acceptance and final validation

Acceptance requires at least four independent replicates. The controller uses a one-sided Student-t band with `acceptance_alpha` divided across the predeclared batch-size looks. A mean change at or above the required decrease is rejected. Acceptance requires the upper band to lie below that threshold. Ambiguous decisions may grow the batch or exhaust the sampling budget. The Student-t band is calibrated for independent Gaussian replicate errors and the declared batch-size looks within one acceptance decision.

Gradient norm and final-validation bands are empirical heuristics. The transport oracle declares deterministic sampling only from continuous absorption, a fixed pencil source and globally zero immutable scattering coefficients. That classification is fixed during preparation, independently of observed sample variance.

Quadratic recovery defaults its held-out final batch to `maximum_batch` for stochastic sampling, or `initial_batch` for the declared deterministic case. Its budget is reserved before further proposals. Normal termination paths use that checkpoint. A failed check returns unresolved. Linear recovery without an explicit final batch retains its heuristic stopping rule.

Inspect `reason`, `termination_detail`, acceptance attempts, domain rejections and final-validation attempts together. `unique_histories` records the controller's identity budget, while oracle replay counts measure tracing work. Use the [recovery API](../reference/api-recovery.md) to inspect the complete result fields.
