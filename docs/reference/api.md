# Python API


Browse the signatures, defaults and docstrings by package area. Each entry gives the callable signature and its docstring.

| Area | Reference |
|---|---|
| Coordinates, compact acquisitions, imaging volumes, material inputs and errors | [Geometry and physical inputs](api-geometry.md) |
| Prepared projection, spectral pipelines, detectors, objectives and tape ownership | [Deterministic operators](api-operators.md) |
| Photon histories, typed simulation results, derivative measures and estimators | [Transport](api-transport.md) |
| Parameter controllers, composed evaluators, field penalties and material reconstruction | [Recovery and reconstruction](api-recovery.md) |
| Adjoint/norm diagnostics, numerical references and uncertainty | [Diagnostics](api-diagnostics.md) |

This reference covers the principal Python boundaries. Private runtime helpers and Warp kernel entry points are implementation details. The supplied-data application modules are command-line drivers. Their case schemas and outputs are documented in [applications](../applications/cases.md).

Warp arrays must satisfy the documented dtype, shape and device contracts, including parameters annotated as `Any`. Use the [operator reference](operators.md), [transport reference](transport.md) and [execution rules](../concepts/execution.md) when composing interfaces.
