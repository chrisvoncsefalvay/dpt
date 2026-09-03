"""Complete bounded CUDA iterations with transactional host publication.

The captured operation evaluates the incumbent gradient, diagnostic mapping,
BB proposal and bounded Armijo/GLL search. Its candidate is resident until the
initial and rejection observers return. Publication then precedes accepted
observers, legacy callbacks and checkpoint writers, as in eager execution.
"""

from __future__ import annotations

import math
import time
from typing import Any

from ._runtime import HostReadback, StatusCheckpoint, load_kernels, scoped_graph_capture
from .contracts import ContractError, NumericalError, raise_device_status


def _reference_capacity(window: int, iterations: int, start: int, count: int) -> int:
    """Retain every admitted GLL reference and every possible future append."""
    return max(1, min(window, count + iterations - start))


def _decode_status_snapshot(
    solver: Any, entries: list[tuple[Any, Any]], codes: Any
) -> NumericalError | ContractError | None:
    """Reproduce canonical checkpoint diagnostics from one completed payload."""
    codes = tuple(int(code) for code in codes)
    snapshots = {id(owner): code for code, (owner, _) in zip(codes, entries, strict=True)}
    errors: list[ContractError | NumericalError] = []
    diagnostics: list[tuple[str, int, str]] = []
    for code, (owner, _) in zip(codes, entries, strict=True):
        if not code:
            continue
        try:
            if owner is solver:
                solver._check_code(code)
            elif hasattr(owner, "_raise_status"):
                owner._raise_status(code)
            elif owner is solver._spectral:
                raise_device_status(code, "spectral")
            else:
                # MaterialProjectionWorkspace.check_status first checks its
                # nested projection while all snapshot words remain sticky.
                owner.projection._raise_status(snapshots[id(owner.projection)])
                raise_device_status(
                    code,
                    "material projection",
                    messages={1: "invalid material path cotangents"},
                )
        except (ContractError, NumericalError) as error:
            errors.append(error)
            diagnostics.append((type(owner).__name__, code, str(error)))
    if errors:
        for owner, code, message in diagnostics:
            errors[0].add_note(f"{owner} status {code}: {message}")
        errors[0].__dict__["status_diagnostics"] = tuple(diagnostics)
        return errors[0]
    return None


class _IterationGraph:
    """Fixed-pointer controls, bounded attempted-step ledger and graph cache."""

    def __init__(self, solver: Any, state: Any):
        self.solver = solver
        self.wp = solver.wp
        if solver.penalty_operator is not solver._graph_penalty:
            raise ContractError("the prepared graph penalty binding must remain fixed")
        self.k = load_kernels("dpt.kernels.material_iteration")
        self.projectors = load_kernels("dpt.kernels.material_iteration_projection")
        if not self.wp.is_conditional_graph_supported():
            raise ContractError(
                "bounded iteration graphs require CUDA toolkit and driver 12.4 or newer"
            )
        self.policy = self.k.IterationPolicy()
        settings = state.policy
        # A solve can append at most its remaining iteration count. Capping the
        # storage by that count preserves a larger declared GLL window exactly.
        self.window = _reference_capacity(
            settings.nonmonotone_window,
            settings.iterations,
            state.start_iteration,
            len(state.reference_losses),
        )
        self.attempt_limit = settings.maximum_backtracks + int(state.use_inertia)
        if self.attempt_limit > (2**31 - 1) // self.k.LEDGER_WIDTH:
            raise ContractError("bounded iteration ledger exceeds the index range")
        for name, value in (
            ("maximum_backtracks", settings.maximum_backtracks),
            ("window", self.window),
            ("relative_mapping", int(state.relative_mapping)),
            ("use_secant", int(state.use_secant)),
            ("use_inertia", int(state.use_inertia)),
            ("maximum_step", settings.maximum_step),
            ("minimum_secant", settings.secant_minimum_step),
            ("backtrack_factor", settings.backtracking_factor),
            ("armijo", settings.armijo),
            ("mapping_step", settings.mapping_step),
            ("relative_tolerance", settings.relative_gradient_mapping_tolerance),
        ):
            setattr(self.policy, name, value)
        self.checkpoint = StatusCheckpoint(
            solver.context,
            (
                solver,
                solver._spectral,
                solver._objective,
                *(view.projection(False) for view in solver.views),
                *(view.projection(True) for view in solver.views),
            ),
        )
        checkpoint_bindings: Any = self.checkpoint
        self.entries: list[tuple[Any, Any]] = checkpoint_bindings._entries
        accepted_bindings: Any = solver._evaluation_checkpoints[False]
        base_pointers = {int(status.ptr) for _, status in accepted_bindings._entries}
        self.base_indices = {
            index
            for index, (_, status) in enumerate(self.entries)
            if int(status.ptr) in base_pointers
        }
        self.status_width = len(self.entries) + 1
        self.ledger_size = self.attempt_limit * self.k.LEDGER_WIDTH
        self.trial_status_size = self.attempt_limit * self.status_width
        self.payload_size = (
            self.k.CONTROL_SIZE
            + self.k.FLAG_SIZE
            + self.ledger_size
            + 2 * len(self.entries)
            + self.trial_status_size
        )
        if self.payload_size > 2**31 - 1:
            raise ContractError("bounded iteration completion exceeds the index range")
        wp, device = self.wp, solver.device
        with solver.context.scope():
            self.controls = wp.zeros(self.k.CONTROL_SIZE, dtype=wp.float64, device=device)
            self.flags = wp.zeros(self.k.FLAG_SIZE, dtype=wp.int32, device=device)
            self.control_host = wp.zeros(
                self.k.CONTROL_SIZE, dtype=wp.float64, device="cpu", pinned=True
            )
            self.flag_host = wp.zeros(self.k.FLAG_SIZE, dtype=wp.int32, device="cpu", pinned=True)
            self.reference = wp.zeros(self.window, dtype=wp.float64, device=device)
            self.reference_host = wp.zeros(self.window, dtype=wp.float64, device="cpu", pinned=True)
            self.ledger = wp.empty(self.ledger_size, dtype=wp.float64, device=device)
            self.base_status = wp.zeros(2 * len(self.entries), dtype=wp.int32, device=device)
            self.trial_status = wp.zeros(self.trial_status_size, dtype=wp.int32, device=device)
            self.payload = wp.empty(self.payload_size, dtype=wp.float64, device=device)
            self.reader = HostReadback(solver.context, self.payload)
            # Distinct conditional handles prevent nested captures aliasing the
            # predicate updated by an outer node or the while-body condition.
            self.gates = [wp.zeros(1, dtype=wp.int32, device=device) for _ in range(6)]
            self.loop_condition = self.flags[self.k.LOOP : self.k.LOOP + 1]
            seen: set[int] = set()
            self.predictions: list[tuple[Any, Any, Any]] = []
            for view in solver.views:
                for value in (view.paths, *(channel.mean for channel in view.channels)):
                    if int(value.ptr) not in seen:
                        seen.add(int(value.ptr))
                        self.predictions.append((value, wp.empty_like(value), wp.empty_like(value)))
            self.pending_trial = [
                (status, wp.empty_like(status))
                for index, (_, status) in enumerate(self.entries)
                if index not in self.base_indices
            ]
        self.graphs: dict[bool, Any] = {}
        self._warmed = False
        self.binding = self._binding()

    def reset(self, state: Any) -> None:
        """Start a solve or resume using the existing resident allocations."""
        if self._binding() != self.binding:
            raise ContractError("bounded iteration graph bindings must remain fixed")
        with self.solver.context.scope():
            values = self.reference_host.numpy()
            values[:] = 0
            count = min(self.window, len(state.reference_losses))
            if count:
                values[:count] = state.reference_losses[-count:]
            self.wp.copy(self.reference, self.reference_host, stream=self.solver.stream)

    def _binding(self) -> tuple[Any, ...]:
        s = self.solver
        arrays = [s.fractions, s.gradient, s.trial, s._diagonal_device, s._penalty_device]
        if s.voxel_metric_scale is not None:
            arrays.append(s.voxel_metric_scale)
        if s._inertial_previous is not None:
            arrays.append(s._inertial_previous)
        if s._secant_fields is not None:
            arrays.extend((s._secant_fields, s._secant_gradient))
        for view in s.views:
            arrays.extend((view.paths, view.trial_paths))
            for channel in view.channels:
                arrays.extend((channel.mean, channel.counts, channel.weights, channel.response))
                arrays.extend(
                    value
                    for value in (channel.objective_weights, channel.valid)
                    if value is not None
                )
        return (
            id(s.penalty_operator),
            s.settings,
            s._data_gradient_binding(),
            *(int(value.ptr) for value in arrays),
        )

    def _launch(self, kernel: Any, inputs: list[Any], *, dim: int = 1, tiled: bool = False) -> None:
        function = self.wp.launch_tiled if tiled else self.wp.launch
        kwargs: dict[str, Any] = {"stream": self.solver.stream, "record_tape": False}
        if tiled:
            kwargs["block_dim"] = 256
        function(kernel, dim=dim, inputs=inputs, **kwargs)

    def _if(self, kind: int, slot: int, body: Any) -> None:
        self._launch(self.k.gate, [self.flags, self.gates[slot], kind])
        self.wp.capture_if(self.gates[slot], body, stream=self.solver.stream)

    def _collect_base(self, *, diagnostic: bool = False) -> None:
        for index, (owner, status) in enumerate(self.entries):
            if index not in self.base_indices:
                continue
            self._launch(
                self.k.collect_status,
                [
                    status,
                    self.base_status,
                    index + int(diagnostic) * len(self.entries),
                    self.flags,
                    1,
                    int(owner is not self.solver),
                ],
            )

    def _collect_trial(self, *, projection: bool = False) -> None:
        if projection:
            self._launch(
                self.k.collect_trial_status,
                [
                    self.solver._status,
                    self.trial_status,
                    self.status_width,
                    self.status_width - 1,
                    self.flags,
                    0,
                ],
            )
        else:
            for index, (owner, status) in enumerate(self.entries):
                self._launch(
                    self.k.collect_trial_status,
                    [
                        status,
                        self.trial_status,
                        self.status_width,
                        index,
                        self.flags,
                        int(owner is not self.solver),
                    ],
                )

    def _validate(self, *, trial: bool) -> None:
        projection = self.solver.views[0].projection(trial)
        self._launch(
            projection.kernels.validate_fractions
            if projection.field_domain == "fractions"
            else projection.kernels.validate_nonnegative,
            [
                projection.fields,
                projection.materials,
                projection.projection.grid.voxels,
                projection.status,
            ],
            dim=projection.projection.grid.voxels,
        )

    def _max(self) -> Any:
        s = self.solver
        previous, count = s._mappings, s._voxel_tiles
        for following in s._max_buffers:
            self._launch(
                s.kernels.max_partials,
                [previous, count, following],
                dim=(count + 255) // 256,
                tiled=True,
            )
            previous, count = following, (count + 255) // 256
        return previous

    def _project(self, *, diagnostic: bool) -> None:
        s = self.solver
        use_metric = not diagnostic or bool(self.policy.relative_mapping)
        selected = self.projectors.projected_step_tiled(
            s.materials, s.field_domain == "nonnegative"
        )
        metric_arguments: tuple[Any, ...] = (
            use_metric and s.material_diagonal_metric is not None,
            s._diagonal_device,
        )
        if use_metric and s.material_metric is not None:
            selected = (
                self.projectors.projected_metric_step_tiled
                if s.field_domain == "fractions"
                else self.projectors.projected_metric_nonnegative_step_tiled
            )
            metric_arguments = s._metric_arguments
        self._launch(
            selected,
            [
                s.fractions,
                s.gradient,
                s.grid.voxels,
                self.controls,
                self.k.MAPPING_STEP if diagnostic else self.k.STEP,
                s.fractions if diagnostic or s._inertial_previous is None else s._inertial_previous,
                self.k.MOMENTUM,
                use_metric and s.voxel_metric_scale is not None,
                s.voxel_metric_scale if s.voxel_metric_scale is not None else s.fractions,
                *metric_arguments,
                s.trial,
                s._components,
                s._mappings,
                s._status,
            ],
            dim=s._voxel_tiles,
            tiled=True,
        )
        s._reduction.finish(s._voxel_tiles, s._slope, s._status, source=s._components)
        maximum = self._max()
        self._launch(s.kernels.trial_controls, [maximum, s._slope, s._status, s._trial_controls])

    def _relative(self) -> None:
        s = self.solver
        self._launch(
            s.kernels.gradient_magnitudes_tiled,
            [s.gradient, s.materials, s.grid.voxels, s._mappings, s._status],
            dim=s._voxel_tiles,
            tiled=True,
        )
        maximum = self._max()
        self._collect_base(diagnostic=True)
        self._launch(self.k.relative_scale, [maximum, self.controls, self.flags, self.policy])

    def _secant(self) -> None:
        s = self.solver
        self._launch(
            s.kernels.secant_products_tiled,
            [
                s.size,
                s.fractions,
                s.gradient,
                s._secant_fields,
                s._secant_gradient,
                s.grid.voxels,
                s.material_metric is not None,
                s._secant_metric,
                s.voxel_metric_scale is not None,
                s.voxel_metric_scale if s.voxel_metric_scale is not None else s.fractions,
                s.material_diagonal_metric is not None,
                s._diagonal_device,
                s._components,
                s._secant_products,
            ],
            dim=s._scalar_tiles,
            tiled=True,
        )
        s._reduction.finish(s._scalar_tiles, s._secant_numerator, s._status, source=s._components)
        s._reduction.finish(
            s._scalar_tiles, s._secant_denominator, s._status, source=s._secant_products
        )
        self._launch(
            self.k.secant_result,
            [s._secant_totals, s._status, self.controls, self.flags, self.policy],
        )

    def _attempt(self) -> None:
        s = self.solver
        self._launch(self.k.prepare_attempt, [self.controls, self.flags])
        self._project(diagnostic=False)
        self._collect_trial(projection=True)
        self._launch(self.k.proposal_result, [s._trial_controls, self.controls, self.flags])

        def value() -> None:
            self._launch(self.k.evaluation_started, [self.flags])
            self._collect_trial()

            def validated() -> None:
                self._validate(trial=True)
                self._collect_trial()

                def evaluated() -> None:
                    s._launch_evaluation(
                        gradient=False, trial=True, reuse_forward=False, fields=s.trial
                    )
                    self._collect_trial()
                    self._launch(self.k.trial_value, [s._summary, self.controls, self.flags])

                self._if(2, 5, evaluated)

            self._if(2, 4, validated)

        self._if(2, 3, value)
        self._launch(self.k.decide_attempt, [self.controls, self.flags, self.ledger, self.policy])

    def _search(self) -> None:
        s = self.solver
        if self.policy.use_secant:
            self._if(4, 2, self._secant)
            s._remember_secant()
        self._launch(self.k.begin_search, [self.controls, self.flags, self.policy])
        self.wp.capture_while(self.loop_condition, self._attempt, stream=s.stream)

    def _diagnostic(self) -> None:
        if self.policy.relative_mapping:
            self._if(3, 1, self._relative)

        def projected() -> None:
            self._project(diagnostic=True)
            self._collect_base(diagnostic=True)
            self._launch(
                self.k.mapping_result,
                [self.solver._trial_controls, self.controls, self.flags, self.policy],
            )

        self._if(0, 1, projected)
        self._if(1, 1, self._search)

    def _body(self, reuse: bool) -> None:
        s = self.solver
        self._launch(self.k.reset_iteration, [self.flags, self.controls])
        self.base_status.zero_()
        self.trial_status.zero_()
        self.ledger.zero_()
        self._collect_base()

        def base() -> None:
            self._validate(trial=False)
            self._collect_base()

            def evaluated() -> None:
                s._launch_evaluation(
                    gradient=True, trial=False, reuse_forward=reuse, fields=s.fractions
                )
                self._collect_base()
                self._launch(
                    self.k.base_value,
                    [s._summary, self.controls, self.flags, self.reference, self.policy],
                )
                for value, incumbent, _ in self.predictions:
                    self.wp.copy(incumbent, value, stream=s.stream)

            self._if(0, 2, evaluated)

        self._if(0, 0, base)
        self._if(0, 0, self._diagnostic)
        for value, _, candidate in self.predictions:
            self.wp.copy(candidate, value, stream=s.stream)
        self._launch(
            self.k.pack,
            [
                self.controls,
                self.flags,
                self.ledger,
                self.base_status,
                self.trial_status,
                self.payload,
            ],
            dim=self.payload_size,
        )

    def _warm(self) -> None:
        if self._warmed:
            return
        s = self.solver
        s._evaluation_checkpoints[False].consume()
        s.views[0].projection(False).validate_inputs(stream=s.stream)
        s._launch_evaluation(gradient=True, trial=False, reuse_forward=False, fields=s.fractions)
        s._evaluation_checkpoints[False].consume()
        # The trial evaluator is warmed at the incumbent, never at an
        # unaccepted optimisation proposal or by publishing caller fields.
        for status, saved in self.pending_trial:
            self.wp.copy(saved, status, stream=s.stream)
            status.zero_()
        try:
            self.wp.copy(s.trial, s.fractions, stream=s.stream)
            s._launch_evaluation(gradient=False, trial=True, reuse_forward=False, fields=s.trial)
            s._evaluation_checkpoints[True].consume()
        finally:
            for status, saved in self.pending_trial:
                self.wp.copy(status, saved, stream=s.stream)
        self.wp.load_module(self.k, device=s.device, max_workers=0)
        selected = self.projectors.projected_step_tiled(
            s.materials, s.field_domain == "nonnegative"
        )
        self.wp.load_module(selected.module, device=s.device, block_dim=256, max_workers=0)
        self.wp.load_module(self.projectors, device=s.device, block_dim=256, max_workers=0)
        self._warmed = True

    def run(self, state: Any, iteration: int) -> tuple[Any, Any, Any, Any, Any]:
        s, k = self.solver, self.k
        if self._binding() != self.binding:
            raise ContractError("bounded iteration graph bindings must remain fixed")
        if (
            s.penalty_operator is not None
            and getattr(s.penalty_operator, "graph_safe", None) is not True
        ):
            raise ContractError("a custom graph penalty must retain graph_safe=True")
        with s.context.scope():
            self._warm()
            values = self.control_host.numpy()
            values[:] = 0
            for index, value in (
                (k.STEP, state.step),
                (k.MAPPING_STEP, state.mapping_step),
                (k.TOLERANCE, state.tolerance),
                (k.NORMALISER, state.mapping_normaliser),
                (k.INITIAL_MAPPING, state.initial_mapping),
                (k.INERTIAL_T, state.inertial_t),
            ):
                values[index] = value
            flags = self.flag_host.numpy()
            flags[:] = 0
            flags[k.ITERATION] = iteration
            flags[k.REFERENCE_COUNT] = min(self.window, len(state.reference_losses))
            self.wp.copy(self.controls, self.control_host, stream=s.stream)
            self.wp.copy(self.flags, self.flag_host, stream=s.stream)
            reuse = bool(s._accepted_forward_available)
            graph = self.graphs.get(reuse)
            if graph is None:
                with scoped_graph_capture(s.context) as capture:
                    self._body(reuse)
                graph = capture.graph
                self.graphs[reuse] = graph
            self.wp.capture_launch(graph, stream=s.stream)
            payload = self.reader.read()
        a, b = k.CONTROL_SIZE, k.CONTROL_SIZE + k.FLAG_SIZE
        c, d = b + self.ledger_size, b + self.ledger_size + 2 * len(self.entries)
        return payload[:a], payload[a:b], payload[b:c], payload[c:d], payload[d:]

    def predictions_at(self, *, candidate: bool) -> None:
        with self.solver.context.scope():
            for value, incumbent, proposed in self.predictions:
                self.wp.copy(value, proposed if candidate else incumbent, stream=self.solver.stream)

    def decode(
        self, codes: Any, *, projection: bool = False
    ) -> NumericalError | ContractError | None:
        entries = [(self.solver, None)] if projection else self.entries
        return _decode_status_snapshot(self.solver, entries, codes)


def iterate_material_graph(
    solver: Any,
    state: Any,
    checkpoint: Any,
    objective_id: Any,
    callback: Any,
    save_checkpoint: Any,
    lifecycle: Any,
) -> None:
    """Replay one complete CUDA iteration, then publish at lifecycle boundaries."""
    graph = getattr(solver, "_iteration_graph_state", None)
    capacity = _reference_capacity(
        state.policy.nonmonotone_window,
        state.policy.iterations,
        state.start_iteration,
        len(state.reference_losses),
    )
    try:
        if graph is None or graph._binding() != graph.binding or capacity > graph.window:
            graph = _IterationGraph(solver, state)
            solver._iteration_graph_state = graph
        graph.reset(state)
    except BaseException:
        solver._iteration_graph_state = None
        solver._accepted_forward_available = False
        solver._prediction_state = "uninitialised"
        raise
    k = graph.k
    state.lifecycle_started = False
    state.verified_checkpoint = checkpoint is None
    failure_names = {1: "armijo", 2: "non_descent", 3: "nonfinite"}
    for iteration in range(state.start_iteration, state.policy.iterations):
        state.completed_iterations = iteration + 1
        try:
            controls, flags, ledger, base_status, trial_status = graph.run(state, iteration)
        except BaseException:
            solver._accepted_forward_available = False
            solver._prediction_state = "uninitialised"
            raise
        error = graph.decode(base_status[: len(graph.entries)])
        if error is not None:
            solver._accepted_forward_available = False
            solver._prediction_state = "uninitialised"
            raise error
        fatal = int(flags[k.FATAL])
        if fatal and int(flags[k.EVALUATIONS]) == 0:
            raise NumericalError("material reconstruction objective is nonfinite")
        state.loss = float(controls[k.LOSS])
        solver.evaluations += 1
        solver._last_loss = {
            "data_objective": float(controls[k.DATA]),
            "regularisation_objective": float(controls[k.PRIOR]),
        }
        solver._accepted_forward_available = False
        solver._prediction_state = "accepted"
        graph.predictions_at(candidate=False)
        if not state.verified_checkpoint:
            if not math.isclose(
                state.loss,
                state.resumed["accepted_loss"],
                rel_tol=state.checkpoint_relative_tolerance,
                abs_tol=0,
            ):
                raise ContractError(
                    "checkpoint accepted objective differs from the current fitting model"
                )
            state.verified_checkpoint = True
        if not state.lifecycle_started:
            lifecycle.start(
                objective=state.loss,
                iteration=state.start_iteration,
                accepted_steps=state.accepted,
                resumed=checkpoint is not None,
            )
        state.lifecycle_started = True
        state.gradient_current = True
        state.base_losses = dict(solver._last_loss)
        state.reference_losses.append(state.loss)
        del state.reference_losses[: -state.policy.nonmonotone_window]
        state.mapping_normaliser = float(controls[k.NORMALISER])
        state.mapping_step = float(controls[k.MAPPING_STEP])
        state.mapping = float(controls[k.MAPPING])
        state.initial_mapping = float(controls[k.INITIAL_MAPPING])
        state.tolerance = float(controls[k.TOLERANCE])
        error = graph.decode(base_status[len(graph.entries) :])
        if error is not None:
            raise error
        if fatal and not int(flags[k.ATTEMPTS]):
            messages = {
                3: "initial gradient scale is nonfinite",
                4: "relative diagnostic step is not representable",
                5: "projected update or slope is nonfinite",
            }
            raise NumericalError(
                messages.get(fatal, "material reconstruction arithmetic is not representable")
            )
        if int(flags[k.STATIONARY]):
            state.reason = "projected_gradient_tolerance"
            break
        accepted = False
        candidate_predictions = False
        proposed_step = float(controls[k.PROPOSED_STEP])
        secant = float(controls[k.SECANT]) or None
        restart = None
        for attempt in range(int(flags[k.ATTEMPTS])):
            row = ledger[attempt * k.LEDGER_WIDTH : (attempt + 1) * k.LEDGER_WIDTH]
            codes = trial_status[attempt * graph.status_width : (attempt + 1) * graph.status_width]
            error = graph.decode(codes[-1:], projection=True) or graph.decode(codes[:-1])
            step, momentum = float(row[0]), float(row[1])
            backtracks, failure = int(row[6]), failure_names[int(row[7])]
            state.step = step
            if int(row[10]):
                if not candidate_predictions:
                    graph.predictions_at(candidate=True)
                    candidate_predictions = True
                solver._prediction_state = "uninitialised"
            if isinstance(error, ContractError):
                raise error
            if int(row[8]):
                solver.evaluations += 1
                solver._last_loss = {
                    "data_objective": float(row[3]),
                    "regularisation_objective": float(row[4]),
                }
                solver._prediction_state = "trial"
            if failure == "nonfinite":
                if error is None:
                    message = (
                        "material reconstruction objective is nonfinite"
                        if int(row[10])
                        else "projected update or slope is nonfinite"
                    )
                    error = NumericalError(message)
                state.rejected.append(
                    {
                        "iteration": iteration + 1,
                        "step": step,
                        "backtracks": backtracks,
                        "error": str(error),
                        "momentum": momentum,
                    }
                )
            if int(row[9]):
                with solver.context.scope():
                    if state.use_inertia:
                        solver.wp.copy(
                            solver._inertial_previous, solver.fractions, stream=solver.stream
                        )
                    solver.wp.copy(solver.fractions, solver.trial, stream=solver.stream)
                solver._accepted_forward_available = state.policy.cache_accepted_forward
                if state.use_inertia:
                    state.inertial_t = 1.0 if restart is not None else float(controls[k.NEXT_T])
                solver._prediction_state = "accepted"
                state.gradient_current = False
                record = {
                    "iteration": iteration + 1,
                    "accepted": True,
                    "objective_before": state.loss,
                    "line_search_reference": float(controls[k.REFERENCE]),
                    "objective": float(row[2]),
                    **solver._last_loss,
                    "gradient_mapping_before": state.mapping,
                    "step": step,
                    "backtracks": backtracks,
                    "proposed_step": proposed_step,
                    "secant_step": secant,
                    "momentum": momentum,
                    "inertial_restart": restart,
                    "wall_seconds": time.perf_counter() - state.started,
                }
                state.history.append(record)
                state.accepted += 1
                state.loss = float(row[2])
                state.step = state.step_policy.grow(step)
                if objective_id is not None:
                    solver._remember_solve_state(state)
                lifecycle.accepted(iteration + 1, state.loss)
                if callback is not None:
                    callback(iteration + 1, dict(record))
                if save_checkpoint is not None:
                    save_checkpoint(solver.checkpoint())
                accepted = True
                break
            lifecycle.rejected(iteration + 1, failure)
            if momentum:
                restart = failure
        if not accepted:
            state.step = float(controls[k.STEP])
            state.reason = "line_search_failed"
            state.history.append(
                {
                    "iteration": iteration + 1,
                    "accepted": False,
                    "objective": state.loss,
                    **state.base_losses,
                    "gradient_mapping_before": state.mapping,
                    "step": state.step,
                    "wall_seconds": time.perf_counter() - state.started,
                    "failure": state.reason,
                }
            )
            break
