"""Native conditional physical collision measures with rigid-frame jets.

Incoming energy and material data are fixed. The direction coupling is
``outgoing_dot = cross(cross(incoming, incoming_dot), outgoing)``. Terminal
measures are conditional densities per steradian or steradian and outgoing keV.
Detector-area Jacobians, attenuation and moving support measures are composed
by the geometry estimator. Counter domains are the canonical transport domains.
"""

# Warp annotations are runtime DSL types.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false
# Warp struct decorators produce runtime DSL records rather than Python classes.
# pyright: reportGeneralTypeIssues=false
# Warp's concrete vector aliases omit Python indexing from their stubs.
# pyright: reportIndexIssue=false
# These helpers compose existing private canonical sampling primitives.
# pyright: reportPrivateUsage=false
# Mutable Warp loop counters require explicit integer constructors.
# ruff: noqa: UP018, RUF046
import importlib
from typing import Any

from .bound_compton_kernels import (
    bound_energy_event,
    bound_joint_pdf,
    bound_joint_scores,
    impulse_kn,
    impulse_occupancy_scale,
    impulse_scattering_weight_scaled,
    impulse_tail_angle_proposal,
    impulse_use_scaled_tail,
)
from .bound_compton_mixed_kernels import (
    impulse_mixed_angle_proposal,
    impulse_use_mixed_conditioning,
)
from .kernels import (
    Parameters,
    _compton_event_domain,
    _compton_packed_event_domain,
    coefficients_at,
    energy_coordinate,
    scatter_direction,
)
from .physical_kernels import (
    HC,
    PhysicalParameters,
    energy_table_value,
    rayleigh_phase_pdf,
    rayleigh_scatter_event,
)
from .sampling_kernels import transport_random4

wp: Any = importlib.import_module("warp")

wp.set_module_options({"fast_math": False, "fuse_fp": False, "enable_backward": False})

MASS = wp.constant(wp.float64(510.99895069))
TWO_PI = wp.constant(wp.float64(6.283185307179586476925286766559))
ISOTROPIC = wp.constant(0)
FREE_COMPTON = wp.constant(1)
BOUND_COMPTON = wp.constant(2)
RAYLEIGH = wp.constant(3)
FLUORESCENCE = wp.constant(4)
ABSORPTION = wp.constant(5)
CUTOFF = wp.constant(6)


@wp.struct
class NativePhysicalCollision:
    """Completed conditional collision and attached world-ray jet.

    ``status`` is canonical0 continuing,1 absorbed,2 exhausted event budget,
    4 invalid input/arithmetic,
    5 unsupported energy,6 exhausted angular budget. Channel0..6 is isotropic,
    free Compton, bound Compton, Rayleigh, fluorescence, absorption or cutoff.
    Energy/deposition are keV and total rate is mm^-1. ``line_index`` is local
    to the material and minus one when absent. Bound binding/momentum are the
    selected shell binding keV and p_z/(m_e*c). ``layout``0/1/2 denotes legacy,
    packed Compton or bound-v3. ``used_roles`` bits0..5 indicate collision,
    coherent selection, fluorescence, KN, Rayleigh and bound-energy addresses.
    Angular/mark domains include the actual descendant offset65536*branch.

    """

    origin: wp.vec3d
    origin_dot: wp.vec3d
    direction: wp.vec3d
    direction_dot: wp.vec3d
    energy: wp.float64
    deposited: wp.float64
    total_rate: wp.float64
    binding: wp.float64
    momentum: wp.float64
    status: int
    channel: int
    line_index: int
    layout: int
    used_roles: int
    angular_domain: wp.uint32
    mark_domain: wp.uint32


@wp.struct
class NativePhysicalTerminal:
    """One mixed outgoing kernel and endpoint jet at fixed incoming keV.

    ``present`` is zero for an absent/zero-density contribution. A nonzero
    status invalidates the observation. Density has units sr^-1 for measure0
    and sr^-1 keV^-1 for measure1, which holds the bound outgoing mark fixed.
    Direction/origin jets carry world mm geometry. ``energy_dot`` is keV per
    active geometry coordinate and ``density_dot`` has the corresponding units.

    """

    origin: wp.vec3d
    origin_dot: wp.vec3d
    direction: wp.vec3d
    direction_dot: wp.vec3d
    cosine: wp.float64
    cosine_dot: wp.float64
    energy: wp.float64
    energy_dot: wp.float64
    density: wp.float64
    density_dot: wp.float64
    status: int
    present: int
    channel: int
    line_index: int
    measure: int


@wp.struct
class _NativeRates:
    absorption: wp.float64
    scattering: wp.float64
    rayleigh: wp.float64
    total: wp.float64
    coordinate: wp.vec2d
    status: int


@wp.struct
class _BoundAngle:
    cosine: wp.float64
    azimuth: wp.float64
    energy: wp.float64
    binding: wp.float64
    momentum: wp.float64
    status: int
    used_mark: int


@wp.func
def _ray_jet_valid(
    origin: wp.vec3d, direction: wp.vec3d, origin_dot: wp.vec3d, direction_dot: wp.vec3d
) -> int:
    valid = int(1)
    for component in range(3):
        if (
            not wp.isfinite(origin[component])
            or not wp.isfinite(direction[component])
            or not wp.isfinite(origin_dot[component])
            or not wp.isfinite(direction_dot[component])
        ):
            valid = 0
    if wp.abs(wp.dot(direction, direction) - wp.float64(1.0)) > wp.float64(1e-12):
        valid = 0
    scale = wp.max(
        wp.abs(direction_dot[0]), wp.max(wp.abs(direction_dot[1]), wp.abs(direction_dot[2]))
    )
    if scale > wp.float64(1e150):
        scaled_dot = direction_dot / scale
        if wp.abs(wp.dot(direction, scaled_dot)) > wp.float64(1e-12) * wp.length(scaled_dot):
            valid = 0
    else:
        if wp.abs(wp.dot(direction, direction_dot)) > wp.float64(1e-12) * wp.max(
            wp.float64(1.0), wp.length(direction_dot)
        ):
            valid = 0
    return valid


@wp.func
def _reference_model_valid(p: Parameters) -> int:
    valid = int(1)
    if (
        p.continuous_absorption != 0
        or p.energy_bins < 1
        or p.max_angle_trials < 1
        or p.max_angle_trials > 65536
        or p.compton < 0
        or p.compton > 2
    ):
        valid = 0
    if p.compton == 2:
        if p.physics.enabled == 0 or p.physics.cutoff < wp.float64(0.1) or p.compton_packed != 0:
            valid = 0
        if p.sampling.kind != 0 and p.sampling.layout != 2:
            valid = 0
    elif p.sampling.kind != 0:
        if p.compton_packed != 0 and p.sampling.layout != 1:
            valid = 0
        if p.compton_packed == 0 and p.sampling.layout == 1:
            valid = 0
    return valid


@wp.func
def _physical_rates(
    energy: wp.float64,
    material: int,
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    p: Parameters,
) -> _NativeRates:
    result = _NativeRates()
    result.status = 4
    if _reference_model_valid(p) == 0 or not wp.isfinite(energy) or energy <= wp.float64(0.0):
        return result
    if (
        energies.shape[0] != p.energy_bins
        or absorption.shape[0] != scattering.shape[0]
        or absorption.shape[0] % p.energy_bins != 0
    ):
        return result
    materials = absorption.shape[0] // p.energy_bins
    if material < 0 or material >= materials:
        return result
    if energy < energies[0] or energy > energies[p.energy_bins - 1]:
        result.status = 5
        return result
    if p.compton == 2 and (energy < wp.float64(0.1) or energy > wp.float64(5000.0)):
        result.status = 5
        return result
    result.coordinate = energy_coordinate(energy, energies, p)
    interaction = coefficients_at(material, result.coordinate, absorption, scattering, p)
    result.absorption = interaction[0]
    result.scattering = interaction[1]
    result.total = interaction[0] + interaction[1]
    if (
        not wp.isfinite(result.total)
        or result.total <= wp.float64(0.0)
        or not wp.isfinite(interaction[0])
        or not wp.isfinite(interaction[1])
        or interaction[0] < wp.float64(0.0)
        or interaction[1] < wp.float64(0.0)
    ):
        return result
    if p.physics.enabled != 0:
        if (
            p.physics.rayleigh.shape[0] != absorption.shape[0]
            or p.physics.q_offsets.shape[0] != materials + 1
        ):
            return result
        start = p.physics.q_offsets[material]
        end = p.physics.q_offsets[material + 1]
        if (
            start < 0
            or end <= start + 1
            or end > p.physics.q.shape[0]
            or energy / HC > p.physics.q[end - 1]
        ):
            result.status = 5
            return result
        result.rayleigh = energy_table_value(
            p.physics.rayleigh, material, result.coordinate, p.energy_bins
        )
        if (
            not wp.isfinite(result.rayleigh)
            or result.rayleigh < wp.float64(0.0)
            or result.rayleigh > result.scattering
        ):
            return result
        if p.compton == 2 and p.physics.compton_shell_offsets.shape[0] != materials + 1:
            return result
    result.status = 0
    return result


@wp.func
def _bound_angle(
    energy: wp.float64,
    material: int,
    seed: wp.uint64,
    history: wp.uint64,
    event: wp.uint32,
    max_trials: int,
    physical: PhysicalParameters,
) -> _BoundAngle:
    """Canonical bounded impulse sampler retaining its binding/momentum mark."""
    result = _BoundAngle()
    result.energy = energy
    result.status = 6
    mixed_mode = impulse_use_mixed_conditioning(energy, material, physical)
    scale = impulse_occupancy_scale(energy, material, physical)
    envelope = impulse_scattering_weight_scaled(energy, -wp.float64(1.0), material, physical, scale)
    tail_mode = envelope <= wp.float64(0.0) or impulse_use_scaled_tail(
        energy, -wp.float64(1.0), material, physical
    )
    for trial in range(max_trials):
        domain = (
            wp.uint32(0x80000000)
            + wp.uint32(physical.random_branch) * wp.uint32(65536)
            + wp.uint32(trial)
        )
        draws = transport_random4(seed, history, event, domain, physical.sampling)
        cosine = wp.float64(2.0) * draws[0] - wp.float64(1.0)
        acceptance = wp.float64(0.0)
        if mixed_mode:
            proposal = impulse_mixed_angle_proposal(energy, material, draws[0], physical)
            cosine, acceptance = proposal[0], proposal[1]
        elif tail_mode:
            proposal = impulse_tail_angle_proposal(energy, material, draws[0], physical)
            cosine, acceptance = proposal[0], proposal[1]
        else:
            acceptance = (
                wp.float64(0.5)
                * impulse_kn(energy, cosine)
                * impulse_scattering_weight_scaled(energy, cosine, material, physical, scale)
                / envelope
            )
        if not wp.isfinite(cosine) or not wp.isfinite(acceptance):
            result.status = 4
            return result
        if draws[1] < acceptance:
            result.used_mark = 1
            broadening = bound_energy_event(
                energy, cosine, material, seed, history, event, 0, physical
            )
            result.cosine = cosine
            result.azimuth = TWO_PI * draws[2]
            result.energy = broadening[0]
            result.binding = broadening[1]
            result.momentum = broadening[2]
            result.status = int(broadening[3])
            break
    return result


def _makeconditional_physical_event_jet(model: int, packed: bool, label: str):
    """Bind a prepared model without duplicating the mathematical body."""

    @wp.func(
        name="conditional_physical_event_jet"
        if model < 0
        else f"conditional_physical_event_jet_{label}",
        module="unique",
    )
    def conditional_physical_event_jet(
        position: wp.vec3d,
        direction: wp.vec3d,
        position_dot: wp.vec3d,
        direction_dot: wp.vec3d,
        energy: wp.float64,
        material: int,
        density_factor: wp.float64,
        energies: wp.array(dtype=wp.float64),
        absorption: wp.array(dtype=wp.float64),
        scattering: wp.array(dtype=wp.float64),
        p: Parameters,
        seed: wp.uint64,
        history: wp.uint64,
        event: wp.uint32,
        branch: int,
    ) -> NativePhysicalCollision:
        """One real analogue collision using the existing addressed physical law.

        A caller-owned analogue reference Parameters binds prepared immutable tables
        and its proposal identity. This function neither samples a free flight nor
        accepts a Woodcock mark. All continuing energies stay fixed under the rigid
        geometry coupling. A nonzero fatal status invalidates the observation.

        Inputs are world-space vec3d rays/jets, positive keV and a dimensionless
        density factor, material-major energy-fastest mm^-1 tables and the owning
        device Parameters. Original seed/history, event and descendant counters
        retain their canonical meaning. Binding and momentum marks remain fixed.

        """
        result = NativePhysicalCollision()
        result.origin = position
        result.origin_dot = position_dot
        result.energy = energy
        result.channel = ABSORPTION
        result.line_index = -1
        result.status = 4
        if wp.static(model >= 0):
            if p.compton != wp.static(model) or p.compton_packed != wp.static(int(packed)):
                return result
        if (
            _ray_jet_valid(position, direction, position_dot, direction_dot) == 0
            or _reference_model_valid(p) == 0
            or branch < 0
            or branch >= 4096
            or not wp.isfinite(energy)
            or energy <= wp.float64(0.0)
            or not wp.isfinite(density_factor)
            or density_factor <= wp.float64(0.0)
        ):
            return result
        if event >= wp.uint32(p.max_events):
            result.status = 2
            return result
        if (wp.static(model == 2)) if wp.static(model >= 0) else (p.compton == 2):
            result.layout = 2
        elif (wp.static(packed)) if wp.static(model >= 0) else (p.compton_packed != 0):
            result.layout = 1
        if p.physics.enabled != 0 and energy < p.physics.cutoff:
            result.channel = CUTOFF
            result.energy = wp.float64(0.0)
            result.deposited = energy
            result.status = 1
            return result
        rates = _physical_rates(energy, material, energies, absorption, scattering, p)
        if rates.status != 0:
            result.status = rates.status
            return result
        result.total_rate = density_factor * rates.total
        if not wp.isfinite(result.total_rate) or result.total_rate <= wp.float64(0.0):
            return result
        physical = p.physics
        physical.random_branch = branch
        physical.sampling = p.sampling
        offset = wp.uint32(branch) * wp.uint32(65536)
        draws = transport_random4(seed, history, event, offset, p.sampling)
        result.used_roles = 1
        if not wp.isfinite(draws[1]):
            return result
        cosine = wp.float64(0.0)
        azimuth = wp.float64(0.0)
        if draws[1] < rates.absorption / rates.total:
            result.status = 1
            result.energy = wp.float64(0.0)
            result.deposited = energy
            if physical.enabled != 0:
                result.mark_domain = wp.uint32(2) + offset
                result.used_roles |= 4
                marks = transport_random4(seed, history, event, result.mark_domain, p.sampling)
                if (
                    not wp.isfinite(marks[0])
                    or not wp.isfinite(marks[1])
                    or not wp.isfinite(marks[2])
                ):
                    result.status = 4
                    return result
                cumulative = wp.float64(0.0)
                selected = int(-1)
                start = physical.fluorescence_offsets[material]
                end = physical.fluorescence_offsets[material + 1]
                for line in range(start, end):
                    if energy >= physical.fluorescence_threshold[line]:
                        cumulative += energy_table_value(
                            physical.fluorescence_probability, line, rates.coordinate, p.energy_bins
                        )
                        if selected < 0 and marks[0] < cumulative:
                            selected = line
                if not wp.isfinite(cumulative) or cumulative > wp.float64(1.0):
                    result.status = 4
                    return result
                if selected >= 0:
                    result.line_index = selected - start
                    result.channel = FLUORESCENCE
                    result.energy = physical.fluorescence_energy[selected]
                    result.deposited = energy - result.energy
                    cosine = wp.float64(2.0) * marks[1] - wp.float64(1.0)
                    azimuth = TWO_PI * marks[2]
                    result.status = 0
            if result.status != 0:
                return result
        else:
            result.channel = p.compton
            result.status = 0
            if physical.enabled != 0:
                result.used_roles |= 2
                mark = transport_random4(seed, history, event, wp.uint32(3) + offset, p.sampling)
                if not wp.isfinite(mark[0]):
                    result.status = 4
                    return result
                if mark[0] < rates.rayleigh / rates.scattering:
                    result.channel = RAYLEIGH
            if result.channel == RAYLEIGH:
                result.angular_domain = wp.uint32(0xC0000000) + offset
                result.used_roles |= 16
                sample = rayleigh_scatter_event(
                    energy, material, seed, history, event, p.max_angle_trials, physical
                )
                cosine = sample[0]
                azimuth = sample[1]
                result.status = int(sample[3])
            elif (
                (wp.static(model == 2))
                if wp.static(model >= 0)
                else (result.channel == BOUND_COMPTON)
            ):
                result.angular_domain = wp.uint32(0x80000000) + offset
                result.used_roles |= 8
                sample_bound = _bound_angle(
                    energy, material, seed, history, event, p.max_angle_trials, physical
                )
                cosine = sample_bound.cosine
                azimuth = sample_bound.azimuth
                result.energy = sample_bound.energy
                result.binding = sample_bound.binding
                result.momentum = sample_bound.momentum
                result.status = sample_bound.status
                if sample_bound.used_mark != 0:
                    result.mark_domain = wp.uint32(0x10000000) + offset
                    result.used_roles |= 32
            elif (
                (wp.static(model == 1))
                if wp.static(model >= 0)
                else (result.channel == FREE_COMPTON)
            ):
                result.angular_domain = wp.uint32(0x80000000) + offset
                result.used_roles |= 8
                if (wp.static(packed)) if wp.static(model >= 0) else (p.compton_packed != 0):
                    sampled = _compton_packed_event_domain(
                        energy,
                        seed,
                        history,
                        event,
                        p.max_angle_trials,
                        result.angular_domain,
                        draws[2],
                        p.sampling,
                    )
                    cosine = sampled.cosine
                    azimuth = sampled.azimuth
                    result.energy = sampled.energy
                    result.status = sampled.status
                else:
                    sampled = _compton_event_domain(
                        energy,
                        seed,
                        history,
                        event,
                        p.max_angle_trials,
                        result.angular_domain,
                        p.sampling,
                    )
                    cosine = sampled.cosine
                    azimuth = sampled.azimuth
                    result.energy = sampled.energy
                    result.status = sampled.status
            else:
                cosine = wp.float64(2.0) * draws[2] - wp.float64(1.0)
                azimuth = TWO_PI * draws[3]
            if result.status != 0:
                return result
            result.deposited = energy - result.energy
        result.direction = scatter_direction(direction, cosine, azimuth)
        result.direction_dot = wp.cross(wp.cross(direction, direction_dot), result.direction)
        if (
            not wp.isfinite(result.energy)
            or result.energy <= wp.float64(0.0)
            or result.deposited < wp.float64(0.0)
            or _ray_jet_valid(position, result.direction, position_dot, result.direction_dot) == 0
        ):
            result.status = 4
        return result

    return conditional_physical_event_jet


conditional_physical_event_jet = _makeconditional_physical_event_jet(-1, False, "generic")


@wp.func
def _kn_shape_slope(energy: wp.float64, cosine: wp.float64) -> wp.vec3d:
    alpha = energy / MASS
    ratio = wp.float64(1.0) / (wp.float64(1.0) + alpha * (wp.float64(1.0) - cosine))
    sine2 = wp.float64(1.0) - cosine * cosine
    shape = ratio * ratio * ratio + ratio - ratio * ratio * sine2
    slope = (alpha * ratio) * ratio * (
        wp.float64(3.0) * ratio * ratio + wp.float64(1.0) - wp.float64(2.0) * ratio * sine2
    ) + wp.float64(2.0) * cosine * ratio * ratio
    return wp.vec3d(shape, slope, ratio)


@wp.func
def _kn_normalisation(energy: wp.float64) -> wp.float64:
    alpha = energy / MASS
    normal = wp.float64(8.0 / 3.0)
    if alpha > wp.float64(0.05):
        inverse = wp.float64(1.0) / alpha
        logarithm = wp.log(wp.float64(1.0) + wp.float64(2.0) * alpha)
        normal = (
            wp.float64(2.0)
            * inverse
            * (
                (wp.float64(1.0) + inverse)
                * (
                    wp.float64(2.0) * (wp.float64(1.0) + inverse) / (wp.float64(2.0) + inverse)
                    - logarithm * inverse
                )
                + logarithm * wp.float64(0.5)
                - (inverse + wp.float64(3.0))
                / ((inverse + wp.float64(2.0)) * (inverse + wp.float64(2.0)))
            )
        )
    else:
        normal = wp.float64(-10880905.198268399)
        normal = wp.float64(5175602.943358396) + alpha * normal
        normal = wp.float64(-2455262.1660818714) + alpha * normal
        normal = wp.float64(1161297.018231854) + alpha * normal
        normal = wp.float64(-547445.1241830066) + alpha * normal
        normal = wp.float64(257100.29803921568) + alpha * normal
        normal = wp.float64(-120227.35238095238) + alpha * normal
        normal = wp.float64(55945.658608058606) + alpha * normal
        normal = wp.float64(-25885.069597069596) + alpha * normal
        normal = wp.float64(11896.540792540793) + alpha * normal
        normal = wp.float64(-5424.09696969697) + alpha * normal
        normal = wp.float64(2449.3252525252524) + alpha * normal
        normal = wp.float64(-1092.9777777777779) + alpha * normal
        normal = wp.float64(480.5079365079365) + alpha * normal
        normal = wp.float64(-207.23809523809524) + alpha * normal
        normal = wp.float64(87.16190476190476) + alpha * normal
        normal = wp.float64(-35.46666666666667) + alpha * normal
        normal = wp.float64(13.866666666666667) + alpha * normal
        normal = wp.float64(-5.333333333333333) + alpha * normal
        normal = wp.float64(2.6666666666666665) + alpha * normal
    return TWO_PI * normal


@wp.func
def _atomic_value_slope(q: wp.float64, material: int, physical: PhysicalParameters) -> wp.vec2d:
    lower = physical.q_offsets[material]
    upper = physical.q_offsets[material + 1] - 1
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if physical.q[middle] <= q:
            lower = middle
        else:
            upper = middle
    slope = (physical.form_factor[upper] - physical.form_factor[lower]) / (
        physical.q[upper] - physical.q[lower]
    )
    value = physical.form_factor[lower] + slope * (q - physical.q[lower])
    return wp.vec2d(value, slope)


@wp.func
def _bound_cosine_score(
    energy: wp.float64,
    cosine: wp.float64,
    outgoing: wp.float64,
    material: int,
    physical: PhysicalParameters,
) -> wp.float64:
    """Complete conditioned cosine score with both photon energies fixed."""
    return bound_joint_scores(energy, cosine, outgoing, material, physical)[0]


def _maketerminal_physical_kernel_jet(model: int, label: str):
    """Bind a prepared model without duplicating the mathematical body."""

    @wp.func(
        name="terminal_physical_kernel_jet"
        if model < 0
        else f"terminal_physical_kernel_jet_{label}",
        module="unique",
    )
    def terminal_physical_kernel_jet(
        position: wp.vec3d,
        direction: wp.vec3d,
        position_dot: wp.vec3d,
        direction_dot: wp.vec3d,
        endpoint: wp.vec3d,
        endpoint_dot: wp.vec3d,
        energy: wp.float64,
        material: int,
        kind: int,
        line_index: int,
        bound_outgoing_mark: wp.float64,
        energies: wp.array(dtype=wp.float64),
        absorption: wp.array(dtype=wp.float64),
        scattering: wp.array(dtype=wp.float64),
        p: Parameters,
    ) -> NativePhysicalTerminal:
        """Evaluate kind0 noncoherent,1 Rayleigh or2 material-local radiative line.

        Bound kind0 requires a detached fixed outgoing keV integration mark and
        returns the complete joint measure. The caller divides out that mark's
        proposal density and composes physical detector area/attenuation/response.
        Exact moving support/interpolation boundaries return fatal status4.

        Inputs are world mm rays/endpoints and their derivatives, fixed incoming
        keV and material-major energy-fastest mm^-1 tables. Density is conditional
        rate/total multiplied by the selected differential physical phase law.
        The fixed incoming energy makes its normalisation constant under geometry.

        """
        result = NativePhysicalTerminal()
        result.origin = position
        result.origin_dot = position_dot
        result.energy = energy
        result.line_index = -1
        result.status = 4
        if wp.static(model >= 0):
            if p.compton != wp.static(model):
                return result
        if (
            kind < 0
            or kind > 2
            or _reference_model_valid(p) == 0
            or _ray_jet_valid(position, direction, position_dot, direction_dot) == 0
            or not wp.isfinite(energy)
            or energy <= wp.float64(0.0)
        ):
            return result
        if p.physics.enabled != 0 and energy < p.physics.cutoff:
            result.status = 0
            return result
        rates = _physical_rates(energy, material, energies, absorption, scattering, p)
        if rates.status != 0:
            result.status = rates.status
            return result
        displacement = endpoint - position
        displacement_dot = endpoint_dot - position_dot
        distance = wp.length(displacement)
        if not wp.isfinite(distance) or distance <= wp.float64(0.0):
            return result
        result.direction = displacement / distance
        result.direction_dot = (
            displacement_dot - result.direction * wp.dot(result.direction, displacement_dot)
        ) / distance
        if _ray_jet_valid(position, result.direction, position_dot, result.direction_dot) == 0:
            return result
        result.cosine = wp.clamp(
            wp.dot(direction, result.direction), -wp.float64(1.0), wp.float64(1.0)
        )
        result.cosine_dot = wp.dot(direction_dot, result.direction) + wp.dot(
            direction, result.direction_dot
        )
        result.status = 0
        if kind == 1:
            result.channel = RAYLEIGH
            if rates.rayleigh <= wp.float64(0.0):
                return result
            sine_half = wp.sqrt((wp.float64(1.0) - result.cosine) * wp.float64(0.5))
            q = energy / HC * sine_half
            atom = _atomic_value_slope(q, material, p.physics)
            if (
                q == wp.float64(0.0)
                and atom[1] != wp.float64(0.0)
                and wp.length(result.direction_dot - direction_dot) != wp.float64(0.0)
            ):
                result.status = 4
                return result
            if result.cosine_dot != wp.float64(0.0):
                for node in range(
                    p.physics.q_offsets[material] + 1, p.physics.q_offsets[material + 1] - 1
                ):
                    if q == p.physics.q[node]:
                        result.status = 4
                        return result
            phase = rayleigh_phase_pdf(energy, material, result.cosine, p.physics)
            result.density = rates.rayleigh / rates.total * phase
            if atom[0] > wp.float64(0.0) and result.cosine_dot != wp.float64(0.0):
                q_dot = -energy / (wp.float64(4.0) * HC * sine_half) * result.cosine_dot
                result.density_dot = result.density * (
                    wp.float64(2.0)
                    * result.cosine
                    * result.cosine_dot
                    / (wp.float64(1.0) + result.cosine * result.cosine)
                    + wp.float64(2.0) * atom[1] / atom[0] * q_dot
                )
        elif kind == 2:
            result.channel = FLUORESCENCE
            if p.physics.enabled == 0:
                return result
            start = p.physics.fluorescence_offsets[material]
            end = p.physics.fluorescence_offsets[material + 1]
            line = start + line_index
            if line_index < 0 or line >= end:
                result.status = 4
                return result
            if energy < p.physics.fluorescence_threshold[line]:
                return result
            probability = energy_table_value(
                p.physics.fluorescence_probability, line, rates.coordinate, p.energy_bins
            )
            result.line_index = line_index
            result.energy = p.physics.fluorescence_energy[line]
            result.density = (
                rates.absorption / rates.total * probability / (wp.float64(2.0) * TWO_PI)
            )
        else:
            result.channel = p.compton
            probability = (rates.scattering - rates.rayleigh) / rates.total
            if probability <= wp.float64(0.0):
                return result
            if (wp.static(model == 0)) if wp.static(model >= 0) else (p.compton == 0):
                result.density = probability / (wp.float64(2.0) * TWO_PI)
            elif (wp.static(model == 1)) if wp.static(model >= 0) else (p.compton == 1):
                law = _kn_shape_slope(energy, result.cosine)
                normalisation = _kn_normalisation(energy)
                if not wp.isfinite(normalisation) or normalisation <= wp.float64(0.0):
                    result.status = 4
                    return result
                result.energy = energy * law[2]
                result.density = probability * law[0] / normalisation
                if result.cosine_dot != wp.float64(0.0):
                    result.energy_dot = (result.energy / MASS) * result.energy * result.cosine_dot
                    result.density_dot = probability * law[1] / normalisation * result.cosine_dot
            else:
                if not wp.isfinite(bound_outgoing_mark) or bound_outgoing_mark <= wp.float64(0.0):
                    result.status = 4
                    return result
                result.energy = bound_outgoing_mark
                result.measure = 1
                if result.cosine_dot != wp.float64(0.0):
                    for shell in range(
                        p.physics.compton_shell_offsets[material],
                        p.physics.compton_shell_offsets[material + 1],
                    ):
                        if bound_outgoing_mark == energy - p.physics.compton_binding[shell]:
                            result.status = 4
                            return result
                result.density = probability * bound_joint_pdf(
                    energy, result.cosine, bound_outgoing_mark, material, p.physics
                )
                if result.density > wp.float64(0.0) and result.cosine_dot != wp.float64(0.0):
                    if result.cosine <= -wp.float64(1.0) or result.cosine >= wp.float64(1.0):
                        result.status = 4
                        return result
                    result.density_dot = (
                        result.density
                        * _bound_cosine_score(
                            energy, result.cosine, bound_outgoing_mark, material, p.physics
                        )
                        * result.cosine_dot
                    )
        if p.physics.enabled != 0:
            if result.energy == p.physics.cutoff and result.energy_dot != wp.float64(0.0):
                result.status = 4
                return result
            if result.energy < p.physics.cutoff:
                result.density = wp.float64(0.0)
                result.density_dot = wp.float64(0.0)
        if (
            not wp.isfinite(result.energy)
            or not wp.isfinite(result.energy_dot)
            or not wp.isfinite(result.density)
            or not wp.isfinite(result.density_dot)
            or result.density < wp.float64(0.0)
        ):
            result.status = 4
        elif result.density > wp.float64(0.0):
            result.present = 1
        return result

    return terminal_physical_kernel_jet


terminal_physical_kernel_jet = _maketerminal_physical_kernel_jet(-1, "generic")
