# Recovery and reconstruction API

## FDK initialisation

FDK provides a starting volume from a full circular cone scan. Supply optical
depths in `(views, rows, columns)` order. The result uses the grid's
`(nz, ny, nx)` layout and attenuation in inverse millimetres.
[Initialiser source](#dpt.fdk.fdk_initialise) is included below.

::: dpt.fdk.FDKSpec

::: dpt.fdk.FDKExecution

::: dpt.fdk.FDKResult

::: dpt.fdk.fdk_initialise
    options:
      show_source: true

Use evaluators to compose the forward model and its gradient, and controllers to propose and accept parameter updates. The [recovery guide](../guides/recovery.md) explains chart coordinates, trial rejection and stochastic acceptance.

## Solver lifecycle

Use `observe_lifecycle` for scalar accepted/rejected/terminal progress across
the recovery drivers. `solver_termination` reads a result or material report
through a common reason/category interface. The [progress guide](../guides/recovery.md#observe-solver-progress)
explains iteration numbering, observer errors and checkpoint verification.

[§10.9 Inverse problems with stochastic gradients](https://photontransport.com/chapters/differentiating-transport/#109-inverse-problems-with-stochastic-gradients).

::: dpt.solver_lifecycle.SolverEvent

::: dpt.solver_lifecycle.SolverTermination

::: dpt.solver_lifecycle.termination_reason

::: dpt.solver_lifecycle.solver_termination

## `dpt.registration`


::: dpt.registration.Evaluation

::: dpt.registration.Evaluator

::: dpt.registration.RecoveryPolicy

::: dpt.registration.IterationRecord

::: dpt.registration.RejectedTrial

::: dpt.registration.RecoveryResult

::: dpt.registration.recover_parameters

## `dpt.recovery`


::: dpt.recovery.PoseChart

::: dpt.recovery.PrimaryPoseProblem

::: dpt.recovery.PoseRecoveryResult

::: dpt.recovery.PrimaryPoseEvaluator

::: dpt.recovery.recover_pose

## Sensitivity recovery and bounds

Use [sensitivity recovery](../guides/recovery.md#use-residual-sensitivities) to
form Gauss–Newton/Levenberg–Marquardt steps in a fixed chart and keep trials
inside declared parameter bounds.

::: dpt.pose_optimization.NormalEquations

::: dpt.pose_optimization.ParameterBounds

::: dpt.pose_optimization.PoseLeastSquaresPolicy

::: dpt.pose_optimization.LeastSquaresResult

::: dpt.pose_optimization.recover_least_squares

::: dpt.pose_optimization.PrimaryPoseLeastSquaresEvaluator

::: dpt.pose_optimization.recover_pose_sensitivities

::: dpt.pose_optimization.BoundedGradientPolicy

::: dpt.pose_optimization.BoundedParameterResult

::: dpt.pose_optimization.recover_bounded_parameters

## Cubic Wolfe search


::: dpt.line_search.ScalarEvaluation

::: dpt.line_search.CubicWolfePolicy

::: dpt.line_search.CubicWolfeResult

::: dpt.line_search.cubic_strong_wolfe

## `dpt.spectral_recovery`


::: dpt.spectral_recovery.CalibrationBlock

::: dpt.spectral_recovery.SpectralView

::: dpt.spectral_recovery.SpectralPoseProblem

::: dpt.spectral_recovery.CalibrationEstimate

::: dpt.spectral_recovery.SpectralRecoveryResult

::: dpt.spectral_recovery.SpectralPoseEvaluator

::: dpt.spectral_recovery.recover_spectral_pose

## `dpt.material_reconstruction`


::: dpt.material_reconstruction.MaterialReconstructionView

::: dpt.material_reconstruction.MaterialReconstructionSignalView

::: dpt.material_reconstruction.MaterialReconstructionSettings

::: dpt.material_reconstruction.StepPolicy

::: dpt.material_reconstruction.MaterialReconstruction

::: dpt.material_reconstruction.ResidentPenalty

::: dpt.material_reconstruction.GraphResidentPenalty

::: dpt.material_reconstruction.FisherResidentPenalty

## Regularisation primitives

Use [TV, L1 and support constraints](../guides/regularization.md) to evaluate a
penalty, obtain its active-coordinate subgradient or take a certified proximal
step. Fields and outputs stay on the workspace's device. Scalar values and
certificates are returned to Python.

::: dpt.regularization.RegularizationSpec

::: dpt.regularization.RegularizationProxPolicy

::: dpt.regularization.RegularizationProxResult

::: dpt.regularization.RegularizationWorkspace

::: dpt.regularization.prepare_regularization

::: dpt.regularization.regularization_value_and_gradient

::: dpt.regularization.regularization_prox

## `dpt.material_preconditioning`


::: dpt.material_preconditioning.FisherScaling

::: dpt.material_preconditioning.prepare_fisher_scaling

## `dpt.stochastic_recovery`


::: dpt.stochastic_recovery.IndependentSquaredOracle

::: dpt.stochastic_recovery.QuadraticSquaredOracle

::: dpt.stochastic_recovery.StochasticPolicy

::: dpt.stochastic_recovery.StochasticStep

::: dpt.stochastic_recovery.GradientAttempt

::: dpt.stochastic_recovery.WorkReservation

::: dpt.stochastic_recovery.LocalModel

::: dpt.stochastic_recovery.AcceptanceAttempt

::: dpt.stochastic_recovery.StochasticRecoveryResult

::: dpt.stochastic_recovery.DomainRejection

::: dpt.stochastic_recovery.AcceptanceDecision

::: dpt.stochastic_recovery.acceptance_decision

::: dpt.stochastic_recovery.recover_expected_signal

## `dpt.stochastic_models`


::: dpt.stochastic_models.QuadraticProposal

::: dpt.stochastic_models.validate_curvature

::: dpt.stochastic_models.quadratic_reduction

::: dpt.stochastic_models.quadratic_proposal
