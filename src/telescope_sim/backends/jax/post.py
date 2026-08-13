"""In-graph output programs: tap + post-processing compiled to pure JAX.

``sample_batch(key=...)`` runs each output's tap + post-processor chain
on-device as one jitted, vmapped program over the batched forward
intensities, so noisy training data never round-trips to the host. This
module compiles those programs from the same ``_OutputSpec`` objects the
host path uses.

Semantics match the host-side post stage at the *distribution* level:
the noisy detector mirrors ``hcipy.NoisyDetector`` (charge = power [+
``int_phot_flux`` rescale] + dark; photon noise with the ``large_poisson``
normal-approximation switch above 1e6 counts; × flat field; + Gaussian
read noise; optional abs clamp) and reuses the *realized* flat-field
array from the bound hcipy detector so the fixed-pattern noise is
identical across host and device paths. Random draws, however, come from
JAX PRNG keys and can never bit-match the host path's numpy draws — a
deliberate fork; noisy outputs are validated statistically and are
reproducible within the jax backend for a given key.

A chain element with no in-graph equivalent (a custom tap or
post-processor, or a subsampling detector) makes the whole output
ineligible; ``compile_output_program`` raises with the element named so
the caller can fall back to host-side post by dropping ``key=``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from telescope_sim.outputs.intensity import IntensityOutputTap
from telescope_sim.post.convolve import ConvolveImagePostProcessor
from telescope_sim.post.noisy_detector import NoisyDetectorPostProcessor
from telescope_sim.post.normalization import (
    ChannelsFirst,
    MaxImageNorm,
    MaxIntensityNorm,
    PerSampleNorm,
)

# A stage maps (image, key, overrides) -> image, all traced values.
_Stage = Callable[[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]], jnp.ndarray]

_POISSON_NORMAL_THRESH = 1.0e6  # hcipy.large_poisson switch point

# Per-sample ndim of each override an in-graph program can consume; the
# batch entry point uses this to tell "one value for all samples" apart
# from "a leading batch dimension".
OVERRIDE_SAMPLE_NDIM: dict[str, int] = {"int_phot_flux": 0, "convolve_image": 2}


class OutputProgram:
    """One output's tap + post chain as a pure per-sample function.

    ``__call__(intensities, key, overrides)`` takes the per-sample dict of
    raw focal-plane intensities, a per-sample PRNG key, and a dict of
    traced override values (subset of :attr:`override_params`), and
    returns the finished image. Pure — callers vmap/jit it.
    """

    def __init__(
        self,
        *,
        fp_names: list[str],
        stages: list[_Stage],
        override_params: frozenset[str],
        needs_key: bool,
    ) -> None:
        self._fp_names = fp_names
        self._stages = stages
        self.override_params = override_params
        self.needs_key = needs_key

    def __call__(
        self,
        intensities: dict[str, jnp.ndarray],
        key: jnp.ndarray,
        overrides: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        image = jnp.stack([intensities[n] for n in self._fp_names], axis=-1)
        for i, stage in enumerate(self._stages):
            # Per-stage key fold so two random stages in one chain can
            # never share draws.
            image = stage(image, jax.random.fold_in(key, i), overrides)
        return image


def _ineligible(out_name: str, what: str) -> NotImplementedError:
    return NotImplementedError(
        f"output {out_name!r}: {what} has no in-graph equivalent, so this "
        "output cannot run on-device. Drop the key= argument to use "
        "host-side post-processing instead."
    )


def _max_intensity_norm_stage(peaks: np.ndarray, dtype: Any) -> _Stage:
    peaks_arr = jnp.asarray(np.asarray(peaks, dtype=np.float64), dtype=dtype)

    def stage(image, key, overrides):
        return image / peaks_arr[None, None, :]

    return stage


def _max_image_norm_stage() -> _Stage:
    def stage(image, key, overrides):
        maxes = image.reshape(-1, image.shape[-1]).max(axis=0)
        safe = jnp.where(maxes > 0, maxes, 1.0)
        return image / safe[None, None, :]

    return stage


def _per_sample_norm_stage() -> _Stage:
    def stage(image, key, overrides):
        flat = image.reshape(-1, image.shape[-1])
        mins = flat.min(axis=0)
        maxes = flat.max(axis=0)
        denom = jnp.where(maxes > mins, maxes - mins, 1.0)
        return (image - mins[None, None, :]) / denom[None, None, :]

    return stage


def _channels_first_stage() -> _Stage:
    def stage(image, key, overrides):
        return jnp.transpose(image, (2, 0, 1))

    return stage


def _fftconvolve_same(scene: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    """``scipy.signal.fftconvolve(scene, kernel, mode="same")`` in JAX.

    Full linear convolution via zero-padded FFTs, then the centered crop
    scipy uses (start index ``(full - out) // 2`` per axis).
    """
    full = (scene.shape[0] + kernel.shape[0] - 1, scene.shape[1] + kernel.shape[1] - 1)
    spectrum = jnp.fft.rfft2(scene, full) * jnp.fft.rfft2(kernel, full)
    conv = jnp.fft.irfft2(spectrum, full)
    r0 = (full[0] - scene.shape[0]) // 2
    c0 = (full[1] - scene.shape[1]) // 2
    return conv[r0 : r0 + scene.shape[0], c0 : c0 + scene.shape[1]]


def _convolve_stage(
    out_name: str, pp: ConvolveImagePostProcessor, dtype: Any
) -> tuple[_Stage, set[str]]:
    ref_sum = pp._reference_psf_sum
    if ref_sum is None:
        raise RuntimeError(
            f"output {out_name!r}: convolve_image was not bound by the loader "
            "(missing reference_psf_sum)."
        )
    default_scene = (
        None if pp._default_image is None else jnp.asarray(pp._default_image, dtype=dtype)
    )

    def stage(image, key, overrides):
        scene = overrides.get("convolve_image", default_scene)
        if scene is None:
            # No scene configured or supplied: passthrough (host behavior).
            return image
        kernel = image[..., 0] / ref_sum
        return _fftconvolve_same(scene, kernel)[..., None]

    return stage, {"convolve_image"}


def _noisy_detector_stage(
    out_name: str,
    pp: NoisyDetectorPostProcessor,
    dtype: Any,
) -> tuple[_Stage, set[str]]:
    detector = pp._detector
    if detector is None or pp._focal_grid is None or pp._aperture_area is None:
        raise RuntimeError(f"output {out_name!r}: noisy_detector was not bound by the loader.")
    if np.ndim(detector.subsampling) != 0 or int(round(float(detector.subsampling))) != 1:
        raise _ineligible(out_name, "noisy_detector with subsampling != 1")

    grid = pp._focal_grid
    shape = tuple(int(s) for s in grid.shape)
    weights = np.atleast_1d(np.asarray(grid.weights, dtype=np.float64))
    if weights.size > 1 and not np.all(weights == weights.flat[0]):
        raise _ineligible(out_name, "noisy_detector on a non-uniform focal grid")
    w_in = float(weights.flat[0])

    aperture_area = float(pp._aperture_area)
    default_flux = pp.int_phot_flux
    clamp = pp.clamp_nonnegative
    include_photon_noise = bool(detector.include_photon_noise)
    # Constants mirrored from the bound hcipy detector. The flat field is
    # the *realized* per-pixel array (hcipy draws it at construction when
    # given a scalar), so host and device paths share the fixed pattern.
    flat_field = jnp.asarray(
        np.broadcast_to(np.asarray(detector.flat_field, dtype=np.float64), (grid.size,)).reshape(
            shape
        ),
        dtype=dtype,
    )
    dark = jnp.asarray(
        np.broadcast_to(
            np.asarray(detector.dark_current_rate, dtype=np.float64), (grid.size,)
        ).reshape(shape),
        dtype=dtype,
    )
    read_noise = jnp.asarray(
        np.broadcast_to(np.asarray(detector.read_noise, dtype=np.float64), (grid.size,)).reshape(
            shape
        ),
        dtype=dtype,
    )

    def stage(image, key, overrides):
        power = image[..., 0] * w_in
        flux = overrides.get("int_phot_flux", default_flux)
        if flux is not None:
            total = power.sum()
            scale = jnp.where(total > 0, (flux * aperture_area) / total, 1.0)
            power = power * scale
        charge = power + dark
        if include_photon_noise:
            k_poisson, k_normal, key = jax.random.split(key, 3)
            small = jax.random.poisson(
                k_poisson, jnp.clip(charge, 0.0, _POISSON_NORMAL_THRESH), charge.shape
            ).astype(charge.dtype)
            large = jnp.round(
                charge
                + jax.random.normal(k_normal, charge.shape, dtype=charge.dtype)
                * jnp.sqrt(jnp.clip(charge, 0.0, None))
            )
            charge = jnp.where(charge > _POISSON_NORMAL_THRESH, large, small)
        charge = charge * flat_field
        k_read, key = jax.random.split(key)
        charge = charge + read_noise * jax.random.normal(k_read, charge.shape, dtype=charge.dtype)
        if clamp:
            charge = jnp.abs(charge)
        return charge[..., None]

    return stage, {"int_phot_flux"}


def compile_output_program(
    out_spec: Any, focal_planes: dict[str, Any], dtype: Any
) -> OutputProgram:
    """Compile one ``_OutputSpec`` into an :class:`OutputProgram`.

    Raises ``NotImplementedError`` (with the blocking element named) when
    the tap or any post-processor has no in-graph equivalent.
    """
    tap = out_spec.tap
    if type(tap) is not IntensityOutputTap:
        raise _ineligible(out_spec.name, f"output tap {getattr(tap, 'name', type(tap).__name__)!r}")

    stages: list[_Stage] = []
    override_params: set[str] = set()
    needs_key = False
    for pp in out_spec.post_processors:
        if type(pp) is MaxIntensityNorm:
            peaks = np.array(
                [focal_planes[n].reference_peak_intensity for n in out_spec.focal_plane_names],
                dtype=np.float64,
            )
            stages.append(_max_intensity_norm_stage(peaks, dtype))
        elif type(pp) is MaxImageNorm:
            stages.append(_max_image_norm_stage())
        elif type(pp) is PerSampleNorm:
            stages.append(_per_sample_norm_stage())
        elif type(pp) is ChannelsFirst:
            stages.append(_channels_first_stage())
        elif type(pp) is ConvolveImagePostProcessor:
            stage, params = _convolve_stage(out_spec.name, pp, dtype)
            stages.append(stage)
            override_params |= params
        elif type(pp) is NoisyDetectorPostProcessor:
            stage, params = _noisy_detector_stage(out_spec.name, pp, dtype)
            stages.append(stage)
            override_params |= params
            needs_key = True
        else:
            raise _ineligible(
                out_spec.name, f"post-processor {getattr(pp, 'name', type(pp).__name__)!r}"
            )

    return OutputProgram(
        fp_names=list(out_spec.focal_plane_names),
        stages=stages,
        override_params=frozenset(override_params),
        needs_key=needs_key,
    )


__all__ = ["OVERRIDE_SAMPLE_NDIM", "OutputProgram", "compile_output_program"]
