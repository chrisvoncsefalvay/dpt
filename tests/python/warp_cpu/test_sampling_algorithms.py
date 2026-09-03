"""Independent distributions, absolute counter coordinates and sparse VJP checks."""

# Warp annotations are executable DSL types.
# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownVariableType=false
# pyright: reportMissingParameterType=false, reportUnknownArgumentType=false
# pyright: reportUntypedFunctionDecorator=false, reportPrivateUsage=false
import importlib
import math
from dataclasses import replace
from itertools import pairwise
from typing import Any

import numpy as np
import pytest
from scipy import stats

from dpt._runtime import cpu_test_context
from dpt.contracts import ContractError
from dpt.detector import (
    GaussianReadNoiseExecution,
    ObservationIdentity,
    add_gaussian_read_noise,
    prepare_detector,
)
from dpt.transport import (
    HistoryBatch,
    MaterialGrid,
    ParallelBeam,
    PlanarDetector,
    SamplingDesign,
    TransportError,
    TransportSpec,
    prepare_transport_sampling,
)
from dpt.transport.forward import prepare_transport, trace_histories
from dpt.transport.kernels import ComptonEvent, _compton_event_domain, _compton_packed_event_domain
from dpt.transport.sampling import sampling_uniforms
from dpt.transport.sampling_kernels import RandomDesign, inspect_random_design, transport_random4
from dpt.transport.source import sample_parallel_beam

wp: Any = importlib.import_module("warp")


@wp.kernel
def angular_probe(
    energy: wp.float64,
    seed: wp.uint64,
    first: wp.uint64,
    event: wp.uint32,
    budget: int,
    packed: int,
    design: RandomDesign,  # pyright: ignore[reportGeneralTypeIssues]
    output: wp.array(dtype=wp.vec4d),
):
    index = wp.tid()
    history = first + wp.uint64(index)
    result = ComptonEvent()
    if packed != 0:
        # This probe includes the production free-flight random block.
        from_dpt = wp.vec4d()
        from_dpt = transport_random4(seed, history, event, wp.uint32(0), design)
        result = _compton_packed_event_domain(
            energy,
            seed,
            history,
            event,
            budget,
            wp.uint32(0x80000000),
            from_dpt[2],  # pyright: ignore[reportIndexIssue]
            design,
        )
    else:
        result = _compton_event_domain(
            energy, seed, history, event, budget, wp.uint32(0x80000000), design
        )
    output[index] = wp.vec4d(
        result.cosine, result.azimuth, result.energy, wp.float64(result.status)
    )


def independent_uniforms(seed, identities, event, domain):
    """Vectorised published Random123 integer permutation, separate from DPT."""
    ids = np.asarray(identities, dtype=np.uint64)
    mask = np.uint64(0xFFFFFFFF)
    c = [
        ids & mask,
        ids >> np.uint64(32),
        np.full(ids.shape, event, dtype=np.uint64),
        np.full(ids.shape, domain, dtype=np.uint64),
    ]
    keys = [seed & 0xFFFFFFFF, seed >> 32]
    for _ in range(10):
        p, q = c[0] * np.uint64(0xD2511F53), c[2] * np.uint64(0xCD9E8D57)
        c = [
            (q >> np.uint64(32)) ^ c[1] ^ np.uint64(keys[0]),
            q & mask,
            (p >> np.uint64(32)) ^ c[3] ^ np.uint64(keys[1]),
            p & mask,
        ]
        keys = [(keys[0] + 0x9E3779B9) & 0xFFFFFFFF, (keys[1] + 0xBB67AE85) & 0xFFFFFFFF]
    return (np.column_stack(c).astype(np.float64) + 0.5) / 2**32


def independent_angles(energy, seed, first, count, event, budget, packed, draws=None):
    histories = np.arange(count, dtype=np.uint64) + np.uint64(first)
    out = np.tile([0.0, 0.0, energy, 6.0], (count, 1))
    pending = np.ones(count, dtype=bool)
    flight = independent_uniforms(seed, histories, event, 0) if draws is None else draws(event, 0)
    for proposal in range(budget):
        domain = 0x80000000 + (proposal // 2 if packed else proposal)
        block = (
            independent_uniforms(seed, histories, event, domain)
            if draws is None
            else draws(event, domain)
        )
        lane = 2 * (proposal % 2) if packed else 0
        cosine = 2 * block[:, lane] - 1
        ratio = 1 / (1 + energy / 510.99895069 * (1 - cosine))
        accept = 0.5 * (ratio**3 + ratio - ratio**2 * (1 - cosine**2))
        accepted = pending & (block[:, lane + 1] < accept)
        out[accepted, 0] = cosine[accepted]
        out[accepted, 1] = 2 * math.pi * (flight[accepted, 2] if packed else block[accepted, 2])
        out[accepted, 2] = energy * ratio[accepted]
        out[accepted, 3] = 0
        pending[accepted] = False
    return out


def angular_samples(device, energy, count, budget=256, packed=True, first=2**63 + 19, design=None):
    output = wp.empty(count, dtype=wp.vec4d, device=device)
    wp.launch(
        angular_probe,
        dim=count,
        inputs=[
            energy,
            wp.uint64(9713),
            wp.uint64(first),
            wp.uint32(3),
            budget,
            int(packed),
            RandomDesign() if design is None else design,
            output,
        ],
        device=device,
    )
    return output.numpy()


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("budget", [1, 2, 3, 8])
def test_compton_exact_counter_proposals_and_odd_budget(packed, budget, warp_device):
    actual = angular_samples(warp_device, 6000.0, 512, budget=budget, packed=packed)
    expected = independent_angles(6000.0, 9713, 2**63 + 19, 512, 3, budget, packed)
    np.testing.assert_array_equal(actual[:, 3], expected[:, 3])
    # ANGLE_BUDGET is the published status 6. No second proposal is permitted
    # after an odd cap, and all other status and counter coordinates are fixed.
    np.testing.assert_allclose(actual[:, :3], expected[:, :3], rtol=3e-15, atol=2e-14)
    assert np.count_nonzero(actual[:, 3] == 6) > 0


def quadrature_kn(energy, lower=-1.0, upper=1.0):
    nodes, weights = np.polynomial.legendre.leggauss(256)
    cosine = lower + (nodes + 1) * (upper - lower) / 2
    ratio = 1 / (1 + energy / 510.99895069 * (1 - cosine))
    density = ratio**2 * (ratio + 1 / ratio - (1 - cosine**2))
    measure = weights * density * (upper - lower) / 2
    return cosine, ratio * energy, measure


@pytest.mark.parametrize("energy", [20.0, 150.0, 6000.0])
@pytest.mark.parametrize("packed", [False, True])
def test_kn_angle_energy_distributions_first_two_moments_and_azimuth(
    energy, packed, warp_device, record_testsuite_property
):
    sample = angular_samples(warp_device, energy, 65536, budget=512, packed=packed)
    assert np.all(sample[:, 3] == 0)
    cosine, outgoing, measure = quadrature_kn(energy)
    norm = measure.sum()
    for column, reference in [(0, cosine), (2, outgoing)]:
        for power in (1, 2):
            exact = float(np.dot(measure, reference**power) / norm)
            second = float(np.dot(measure, reference ** (2 * power)) / norm)
            se = math.sqrt(max(0.0, second - exact**2) / len(sample))
            assert abs(float(np.mean(sample[:, column] ** power)) - exact) < 6 * se
    edges = np.array([-1.0, -0.9, -0.75, -0.5, 0.0, 0.5, 0.75, 0.9, 1.0])
    probabilities = np.array(
        [quadrature_kn(energy, a, b)[2].sum() / norm for a, b in pairwise(edges)]
    )
    observed = np.histogram(sample[:, 0], edges)[0]
    expected = probabilities * len(sample)
    expected *= observed.sum() / expected.sum()
    probability = stats.chisquare(observed, expected).pvalue
    assert probability > 1e-6
    assert stats.chisquare(np.histogram(np.zeros(len(sample)), edges)[0], expected).pvalue < 1e-6
    assert stats.kstest(sample[:, 1] / (2 * math.pi), "uniform").pvalue > 1e-6
    for trigonometric in (np.sin, np.cos):
        projected = trigonometric(sample[:, 1])
        assert abs(float(projected.mean())) < 6 / math.sqrt(2 * len(sample))
        centred_angle = sample[:, 0] - sample[:, 0].mean()
        assert abs(float(np.mean(centred_angle * projected))) < 6 * math.sqrt(
            float(np.mean(centred_angle**2)) / (2 * len(sample))
        )
    record_testsuite_property(
        f"kn_{energy}_{packed}_{warp_device}",
        str(
            {
                "angle_chi_square_p": float(probability),
                "count": len(sample),
                "mean_cosine": float(sample[:, 0].mean()),
                "mean_energy": float(sample[:, 2].mean()),
            }
        ),
    )


def compton_workspace(device, count, packed=True):
    spec = TransportSpec(
        MaterialGrid((-2.0, -2.0, 0.0), (4.0, 4.0, 2.0), (1, 1, 1), 1),
        PlanarDetector((-8.0, -8.0), (1.0, 1.0), (16, 16), 3.0),
        150.0,
        "analytic:homogeneous-Compton",
        (0.001, 150.0),
        "free-electron-compton",
        max_angle_trials=32,
        max_events=16,
        compton_sampler="packed-uniform-rejection" if packed else "legacy-rejection",
    )
    return prepare_transport(
        spec,
        material_ids=wp.array([0], dtype=wp.int32, device=device),
        absorption=wp.array([0.03, 0.03], dtype=wp.float64, device=device),
        scattering=wp.array([0.5, 0.5], dtype=wp.float64, device=device),
        max_histories=count,
        device=device,
    )


@pytest.mark.parametrize("law", [None, "stratified", "scrambled-sobol"])
def test_packed_production_walk_full_replay_shards_and_all_qmc_lanes(law, warp_device):
    with cpu_test_context():
        count = 256
        workspace = compton_workspace(warp_device, count)
        batch = HistoryBatch(9713, 2**63 + 19, count)
        if law is not None:
            batch = prepare_transport_sampling(
                workspace,
                batch=batch,
                design=SamplingDesign(law, 16, 32, layout="packed-compton-v2"),
            )
            sampling_identity = batch.sampling
            assert sampling_identity is not None
            coordinate_out = wp.empty(count, dtype=wp.vec4d, device=warp_device)
            for domain in [0, 1, 2, 3, 0x40000000, 0x40000001, 0x8000000F, 0xC000001F]:
                wp.launch(
                    inspect_random_design,
                    dim=count,
                    inputs=[
                        wp.uint64(batch.seed),
                        wp.uint64(batch.first_history),
                        wp.uint32(3),
                        wp.uint32(domain),
                        workspace._parameters.sampling,
                        coordinate_out,
                    ],
                    device=warp_device,
                )
                np.testing.assert_array_equal(
                    coordinate_out.numpy(),
                    sampling_uniforms(sampling_identity, event=3, domain=domain),
                )
            actual = angular_samples(
                warp_device, 150.0, count, budget=32, design=workspace._parameters.sampling
            )
            expected = independent_angles(
                150.0,
                batch.seed,
                batch.first_history,
                count,
                3,
                32,
                True,
                draws=lambda event, domain: sampling_uniforms(
                    sampling_identity, event=event, domain=domain
                ),
            )
            np.testing.assert_array_equal(actual[:, 3], expected[:, 3])
            np.testing.assert_allclose(actual[:, :3], expected[:, :3], rtol=3e-15, atol=2e-14)
        source = ParallelBeam((-0.3, -0.4, -1.0), (0.6, 0.8))
        density = wp.ones(1, dtype=wp.float64, device=warp_device)

        def run(token: HistoryBatch):
            n = token.count
            position = wp.empty(n, dtype=wp.vec3d, device=warp_device)
            direction = wp.empty_like(position)
            weight = wp.empty(n, dtype=wp.float64, device=warp_device)
            source_identity = sample_parallel_beam(
                source,
                batch=token,
                workspace=workspace,
                out_position=position,
                out_direction=direction,
                out_weight=weight,
            )
            token = replace(token, source_identity=source_identity)
            arrays = [
                wp.empty(n, dtype=dtype, device=warp_device)
                for dtype in (wp.int32, wp.float64, wp.float64, wp.int32, wp.int32)
            ]
            trace_histories(
                position,
                direction,
                weight,
                density,
                batch=token,
                workspace=workspace,
                out_pixel=arrays[0],
                out_score=arrays[1],
                out_energy=arrays[2],
                out_events=arrays[3],
                out_status=arrays[4],
            )
            return tuple(a.numpy() for a in arrays)

        whole = run(batch)
        replay = run(batch)
        shards = [run(batch.shard(7, rank).batch) for rank in range(7)]
        for index in range(5):
            np.testing.assert_array_equal(whole[index], replay[index])
            np.testing.assert_array_equal(whole[index], np.concatenate([s[index] for s in shards]))
        assert np.count_nonzero(whole[3] >= 2) > 5
        assert np.count_nonzero((whole[0] >= 0) & (whole[3] >= 1)) > 5
        with pytest.raises(TransportError, match="layout version"):
            prepare_transport_sampling(
                workspace,
                batch=HistoryBatch(3, 0, 128),
                design=SamplingDesign("stratified", 16, 32),
            )


@pytest.mark.parametrize("algorithm", ["scalar-box-muller", "box-muller-pairs"])
def test_gaussian_exact_absolute_counters_replay_odd_chunks_and_endpoints(algorithm, warp_device):
    with cpu_test_context():
        execution = GaussianReadNoiseExecution(algorithm)
        n, offset, sigma = 257, 2**32 - 257, 2.7
        workspace = prepare_detector(
            max_pixels=n, device=warp_device, read_noise_execution=execution
        )
        identity = ObservationIdentity(
            5417, 31, pixel_offset=offset, read_noise_algorithm=algorithm
        )
        values = np.linspace(-2.0, 3.0, n).astype(np.float32)

        def run(start, end):
            source = wp.array(values[start:end], dtype=wp.float32, device=warp_device)
            out = wp.empty_like(source)
            add_gaussian_read_noise(
                source,
                standard_deviation=sigma,
                out_signal=out,
                identity=replace(identity, pixel_offset=offset + start),
                workspace=workspace,
            )
            return out.numpy()

        whole = run(0, n)
        np.testing.assert_array_equal(whole, run(0, n))
        edges = [0, 1, 4, 5, 19, 128, 129, 256, 257]
        np.testing.assert_array_equal(
            whole, np.concatenate([run(a, b) for a, b in pairwise(edges)])
        )
        pixels = np.arange(n, dtype=np.uint64) + np.uint64(offset)
        anchors = pixels if algorithm == "scalar-box-muller" else pixels & np.uint64(0xFFFFFFFE)
        ids = (np.uint64(identity.observation_id) << np.uint64(32)) | anchors
        uniforms = independent_uniforms(
            identity.seed, ids, 0, 0x7FFFFFFF if algorithm == "scalar-box-muller" else 0xFFFFFFFF
        )
        # The detector's retained uniform_pair consumes all four Philox words
        # as two independent 53-bit open uniforms. Build this integer packing
        # independently rather than using transport's four 32-bit uniforms.
        words = np.floor(uniforms * 2**32).astype(np.uint64)
        wide = np.column_stack(
            [
                words[:, 0] * np.uint64(2**21) + (words[:, 1] >> np.uint64(11)),
                words[:, 2] * np.uint64(2**21) + (words[:, 3] >> np.uint64(11)),
            ]
        ).astype(np.float64)
        uniforms = (wide + 1.0) / (2**53 + 2.0)
        radius = np.sqrt(-2 * np.log(uniforms[:, 0]))
        angle = 2 * math.pi * uniforms[:, 1]
        normal = radius * np.cos(angle)
        if algorithm == "box-muller-pairs":
            normal[pixels % 2 == 1] = (radius * np.sin(angle))[pixels % 2 == 1]
        expected = (values.astype(np.float64) + sigma * normal).astype(np.float32)
        np.testing.assert_array_equal(whole, expected)
        if algorithm == "box-muller-pairs":
            wrong = replace(identity, read_noise_algorithm="scalar-box-muller")
            with pytest.raises(ContractError, match="policies differ"):
                add_gaussian_read_noise(
                    wp.zeros(n, dtype=wp.float32, device=warp_device),
                    standard_deviation=sigma,
                    out_signal=wp.empty(n, dtype=wp.float32, device=warp_device),
                    identity=wrong,
                    workspace=workspace,
                )


def test_gaussian_pair_marginal_joint_covariance_and_nonzero_signal(warp_device):
    with cpu_test_context():
        n, sigma = 65536, 2.7
        workspace = prepare_detector(
            max_pixels=n,
            device=warp_device,
            read_noise_execution=GaussianReadNoiseExecution("box-muller-pairs"),
        )
        source = wp.full(n, 3.0, dtype=wp.float32, device=warp_device)
        out = wp.empty_like(source)
        add_gaussian_read_noise(
            source,
            standard_deviation=sigma,
            out_signal=out,
            identity=ObservationIdentity(97191, 123, read_noise_algorithm="box-muller-pairs"),
            workspace=workspace,
        )
        sample = (out.numpy().astype(np.float64) - 3.0) / sigma
        assert stats.kstest(sample, "norm").pvalue > 1e-6
        assert abs(float(sample.mean())) < 6 / math.sqrt(n)
        assert abs(float(np.mean(sample**2)) - 1) < 6 * math.sqrt(2 / n)
        pairs = sample.reshape(-1, 2)
        assert abs(float(np.mean(pairs[:, 0] * pairs[:, 1]))) < 6 / math.sqrt(n / 2)
        assert stats.kstest(np.sum(pairs**2, axis=1), "chi2", args=(2,)).pvalue > 1e-6
        assert abs(
            float(np.mean((pairs[:, 0] ** 2 - 1) * (pairs[:, 1] ** 2 - 1)))
        ) < 12 / math.sqrt(n / 2)
        for lag in [1, 2, 7]:
            assert abs(float(np.mean(sample[:-lag] * sample[lag:]))) < 6 / math.sqrt(n - lag)


@pytest.mark.parametrize("integration", ["midpoint", "cell_gauss"])
def test_sparse_zero_cotangent_volume_vjp_against_independent_matrix(integration, warp_device):
    from dpt.geometry import DetectorGeometry, RigidTransform, compose_pose
    from dpt.projection import ProjectionSpec, prepare_projection, projection_vjp
    from dpt.validation.projection import integrate_sampled_field
    from dpt.volumes import GridSpec

    with cpu_test_context():
        grid = GridSpec((3, 3, 3), (1.2, 0.8, 1.1), (-2.0, -1.0, -1.0))
        detector = DetectorGeometry(
            (-12.0, 0.31, 0.43),
            (12.0, -0.6, -0.6),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.11, 0.13),
            (4, 5),
        )
        transform = compose_pose(RigidTransform(), (0.13, -0.08, 0.07, 0.02, -0.03, 0.01))
        matrix = np.empty((detector.pixels, grid.voxels))
        for voxel in range(grid.voxels):
            basis = np.eye(grid.voxels)[voxel]
            for pixel in range(detector.pixels):
                row, column = divmod(pixel, detector.shape[1])
                matrix[pixel, voxel] = integrate_sampled_field(
                    basis,
                    grid,
                    detector,
                    transform,
                    row,
                    column,
                    midpoint_samples=47 if integration == "midpoint" else None,
                )
        workspace = prepare_projection(
            grid,
            detector,
            ProjectionSpec(
                47,
                precision="float64",
                integration=integration,
                active_pose=False,
                active_volume=True,
                field_domain="signed_linear",
            ),
            device=warp_device,
        )
        pose = wp.array(transform.packed(), dtype=wp.float64, device=warp_device)
        rng = np.random.default_rng(19412)
        field = wp.array(
            rng.normal(size=grid.voxels).astype(np.float32), dtype=wp.float32, device=warp_device
        )
        gradient = wp.empty(grid.voxels, dtype=wp.float64, device=warp_device)
        for active in [0, 1, 5, detector.pixels]:
            covector = np.zeros(detector.pixels)
            selection = rng.choice(detector.pixels, active, replace=False)
            covector[selection] = rng.normal(size=active)
            projection_vjp(
                field,
                pose,
                cotangent=wp.array(covector, dtype=wp.float64, device=warp_device),
                out_mu=gradient,
                workspace=workspace,
                volume_gradient_precision="float64",
            )
            np.testing.assert_allclose(
                gradient.numpy(), matrix.T @ covector, rtol=3e-12, atol=3e-12
            )
