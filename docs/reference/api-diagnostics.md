# Numerical diagnostics API

Use the independent CPU references to check small calculations. The [first calculation](../getting-started/first-calculation.md) and [spectral guide](../guides/spectral-detector.md) show the references in use.

## `dpt.operator_checks`

See [operator diagnostics](../guides/operator-diagnostics.md) for inner-product scaling, power-iteration convergence and the production projection adapter.

::: dpt.operator_checks.LinearOperator

::: dpt.operator_checks.AdjointCheck

::: dpt.operator_checks.OperatorNorm

::: dpt.operator_checks.check_adjoint
    options:
      show_source: true

::: dpt.operator_checks.estimate_operator_norm
    options:
      show_source: true

::: dpt.operator_checks.projection_linear_operator
    options:
      show_source: true

## `dpt.validation.projection`


::: dpt.validation.projection.sample_reference

::: dpt.validation.projection.integrate_sampled_field

::: dpt.validation.projection.quadratic_sample_values

::: dpt.validation.projection.integrate_quadratic_field

## `dpt.validation.spectral`


::: dpt.validation.spectral.SpectralReference

::: dpt.validation.spectral.spectral_reference

## `dpt.validation.identifiability`


::: dpt.validation.identifiability.SensitivitySpectrum

::: dpt.validation.identifiability.sensitivity_spectrum

## `dpt.statistics`


::: dpt.statistics.mean_standard_error

::: dpt.statistics.student_t_critical


## `dpt.gpu_check`


::: dpt.gpu_check.PreflightError

::: dpt.gpu_check.run_preflight

## Physical coefficient sources

The [material-data guide](../guides/physical-data.md) explains interaction kinds,
units, absorption edges and verified downloads.

::: dpt.physical_data.PhotonCoefficientTable

::: dpt.physical_data.ProviderIdentity

::: dpt.physical_data.read_nist_coefficients

::: dpt.physical_data.xraylib_coefficients

::: dpt.reference_data.ReferenceDataset

::: dpt.reference_data.CachedReference

::: dpt.reference_data.fetch_reference_data
