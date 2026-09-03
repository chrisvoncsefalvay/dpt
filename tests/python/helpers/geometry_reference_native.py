"""Shared exact scientific helpers for the native controls."""

# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false, reportPrivateUsage=false
import importlib
from typing import Any, Literal, cast

import numpy as np

from dpt.transport.geometry_tangents import RayTangent
from dpt.transport.sampling import _philox_words, sampling_uniforms
from helpers.physical_geometry_native import SEED
from helpers.physical_geometry_native import actual_tables as actual_tables

wp: Any = importlib.import_module("warp")

native: Any = importlib.import_module("dpt.transport.geometry_reference_kernels")


@wp.kernel
def _responses(
    rays: wp.array2d(dtype=wp.float64),
    query: wp.array2d(dtype=wp.int32),
    histories: wp.array(dtype=wp.uint64),
    materials: wp.array(dtype=wp.int32),
    voxel: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    parameters: native.Parameters,
    seed_image: wp.array(dtype=wp.float64),
    output: wp.array2d(dtype=wp.float64),
    metadata: wp.array2d(dtype=wp.int32),
):
    index = wp.tid()
    ray = native.NativeGeometryRay()
    ray.position = wp.vec3d(rays[index, 0], rays[index, 1], rays[index, 2])
    ray.direction = wp.vec3d(rays[index, 3], rays[index, 4], rays[index, 5])
    result = native.reference_response(
        ray,
        rays[index, 6],
        query[index, 0],
        query[index, 1],
        query[index, 2],
        query[index, 3],
        materials,
        voxel,
        energies,
        absorption,
        scattering,
        density,
        parameters,
        wp.uint64(SEED),
        histories[index],
        query[index, 4],
        query[index, 5],
        query[index, 6],
        seed_image,
    )
    output[index, 0] = result.value
    output[index, 1] = result.final_energy
    metadata[index, 0] = result.status
    metadata[index, 1] = result.collision_count
    metadata[index, 2] = result.terminal
    metadata[index, 3] = parameters.max_events
    for component in range(3):
        if (
            ray.position[component] != rays[index, component]
            or ray.direction[component] != rays[index, component + 3]
        ):
            metadata[index, 3] = -1


@wp.kernel
def _prefixes(
    rays: wp.array2d(dtype=wp.float64),
    query: wp.array2d(dtype=wp.int32),
    histories: wp.array(dtype=wp.uint64),
    materials: wp.array(dtype=wp.int32),
    voxel: wp.array(dtype=wp.float64),
    energies: wp.array(dtype=wp.float64),
    absorption: wp.array(dtype=wp.float64),
    scattering: wp.array(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    parameters: native.Parameters,
    output: wp.array2d(dtype=wp.float64),
    metadata: wp.array2d(dtype=wp.int32),
):
    index = wp.tid()
    ray = native.NativeGeometryRay()
    ray.position = wp.vec3d(rays[index, 0], rays[index, 1], rays[index, 2])
    ray.direction = wp.vec3d(rays[index, 3], rays[index, 4], rays[index, 5])
    result = native.prefix_step(
        ray,
        rays[index, 6],
        query[index, 1],
        query[index, 2],
        query[index, 3],
        materials,
        voxel,
        energies,
        absorption,
        scattering,
        density,
        parameters,
        wp.uint64(SEED),
        histories[index],
        query[index, 4],
        query[index, 5],
        query[index, 6],
    )
    for component in range(3):
        output[index, component] = result.ray.position[component]
    output[index, 3] = result.distance
    output[index, 4] = result.total_optical_depth
    metadata[index, 0] = result.status
    metadata[index, 1] = result.cell
    metadata[index, 2] = result.terminal
    metadata[index, 3] = result.crossings
    metadata[index, 4] = result.event
    for component in range(3):
        if (
            ray.position[component] != rays[index, component]
            or ray.direction[component] != rays[index, component + 3]
        ):
            metadata[index, 4] = -1


class _Addressed:
    """Read actual canonical Philox or resident coordinates with a mapped event."""

    layout: Literal["legacy-v1", "packed-compton-v2", "bound-compton-v3"]

    def __init__(self, spec, history, branch, offset, identity=None):
        self.layout = (
            "bound-compton-v3"
            if spec.scattering_law == "bound-electron-compton"
            else "packed-compton-v2"
            if spec.compton_sampler == "packed-uniform-rejection"
            else "legacy-v1"
        )
        self.identity = (
            f"actual canonical seed={SEED}; history={history}; branch={branch}; "
            f"offset={offset}; resident={identity}"
        )
        self._spec, self._history, self._branch, self._offset = spec, history, branch, offset
        self._resident = identity
        self._cache = {}

    def uniform(self, lineage, event, domain, lane):
        assert lineage == (self._history,)
        if event not in self._cache:
            trials = self._spec.max_angle_trials
            blocks = (trials + 1) // 2 if self.layout == "packed-compton-v2" else trials
            domains = [0, 2, 3]
            if self.layout == "bound-compton-v3":
                domains.append(0x10000000)
            domains += [0x80000000 + i for i in range(blocks)]
            domains += [0xC0000000 + i for i in range(trials)]
            addressed = np.asarray(domains, dtype=np.uint64) + np.uint64(self._branch * 65536)
            if self._resident is None:
                words = _philox_words(
                    SEED,
                    np.full(len(domains), self._history, dtype=np.uint64),
                    event + self._offset,
                    cast(Any, addressed),
                )
                values = (words.astype(np.float64) + 0.5) / 2**32
            else:
                values = np.asarray(
                    [
                        sampling_uniforms(
                            self._resident,
                            histories=[self._history],
                            event=event + self._offset,
                            domain=int(address),
                        )[0]
                        for address in addressed
                    ]
                )
            self._cache[event] = dict(zip(domains, values, strict=True))
        return float(self._cache[event][domain][lane])


def _launch(
    workspace,
    density,
    histories,
    *,
    energy=10.0,
    forced=-1,
    side=None,
    branch=2,
    offset=1,
    limit=16,
    ray=None,
    prefix=False,
    queries=None,
    seed_values=None,
):
    device = workspace.context.device
    ray = RayTangent((0.005, 0.005, 0.005), (0.0, 0.0, 1.0)) if ray is None else ray
    count = len(histories)
    coordinates = np.asarray([(*ray.origin_mm, *ray.direction, energy)] * count)
    axis, face, sign = (-1, 0, 0) if side is None else side
    query = np.asarray([[forced, axis, face, sign, branch, offset, limit]] * count, dtype=np.int32)
    if queries is not None:
        query = np.asarray(queries, dtype=np.int32)
    seed = tuple((index + 1) * (-1.0 if index % 3 == 0 else 1.0) for index in range(16))
    if seed_values is not None:
        seed = tuple(seed_values)
    output = wp.empty((count, 5 if prefix else 2), dtype=wp.float64, device=device)
    metadata = wp.empty((count, 5 if prefix else 4), dtype=wp.int32, device=device)
    inputs = [
        wp.array(coordinates, dtype=wp.float64, device=device),
        wp.array(query, dtype=wp.int32, device=device),
        wp.array(histories, dtype=wp.uint64, device=device),
        workspace.material_ids,
        workspace.voxel_density,
        workspace._energies,
        workspace.absorption,
        workspace.scattering,
        density,
        workspace._parameters,
    ]
    if not prefix:
        inputs.append(wp.array(seed, dtype=wp.float64, device=device))
    wp.launch(
        _prefixes if prefix else _responses,
        count,
        inputs=inputs,
        outputs=[output, metadata],
        device=device,
    )
    return output.numpy(), metadata.numpy(), ray, seed
