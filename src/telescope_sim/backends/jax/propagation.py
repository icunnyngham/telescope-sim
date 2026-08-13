"""JAX matrix-Fourier-transform propagation matching hcipy's conventions.

Implements the Fraunhofer perfect-lens propagation as a per-wavelength
matrix Fourier transform (Soummer et al. 2007), with kernels built directly
from the hcipy grid coordinate arrays so grid centering, integration
weights, and the ``1/(i·λ·f)`` normalization agree with
``hcipy.FraunhoferPropagator`` + ``hcipy.MatrixFourierTransform`` to
floating-point precision:

    E_focal(u) = (1 / (i·λ·f)) · Σ_x E_pupil(x) · w_in · exp(-i·2π/(λf)·u·x)

Both grids must be separable regular Cartesian grids (the ones
``make_pupil_grid`` / ``make_uniform_grid`` produce), which lets the 2-D
transform factor into two small matmuls per wavelength.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray


def _separable_axes(grid: Any) -> tuple[NDArray, NDArray]:
    """Extract (x_1d, y_1d) from a separable hcipy Cartesian grid.

    hcipy flattens fields with x varying fastest; ``grid.shape`` is
    ``(Ny, Nx)``. Raises if the grid is not separable (the MFT
    factorization would be invalid).
    """
    shape = tuple(int(s) for s in grid.shape)
    x2d = np.asarray(grid.x).reshape(shape)
    y2d = np.asarray(grid.y).reshape(shape)
    x1d = x2d[0, :]
    y1d = y2d[:, 0]
    if not (
        np.array_equal(x2d, np.broadcast_to(x1d, shape))
        and np.array_equal(y2d, np.broadcast_to(y1d[:, None], shape))
    ):
        raise ValueError("grid is not a separable Cartesian grid; MFT propagation requires one")
    return x1d, y1d


class FraunhoferMFT:
    """Broadband Fraunhofer propagator: pupil field → summed focal intensity.

    Kernels for every wavelength are precomputed at construction (they are
    small: ``n_lams × focal_res × pupil_res`` per axis); the per-sample
    path is a single jitted, wavelength-vmapped pair of matmuls.

    Parameters
    ----------
    pupil_grid, focal_grid
        hcipy grids (geometry authority — shared with the hcipy backend).
    filter_lams
        Wavelengths (meters) to propagate; intensities are summed over them.
    focal_length
        Lens focal length in meters; 1.0 reproduces the angular-focal-plane
        convention (focal coordinates in radians).
    dtype
        ``"float64"`` (default; the parity-first precision) or
        ``"float32"`` for half-memory kernels and faster propagation at
        single-precision accuracy. Kernels are built in the matching
        complex dtype and inputs are cast on the way in.
    """

    def __init__(
        self,
        pupil_grid: Any,
        focal_grid: Any,
        filter_lams: NDArray[np.floating],
        *,
        focal_length: float = 1.0,
        dtype: str = "float64",
    ) -> None:
        if dtype not in ("float64", "float32"):
            raise ValueError(f"dtype must be 'float64' or 'float32', got {dtype!r}")
        real_dtype = jnp.float64 if dtype == "float64" else jnp.float32
        complex_dtype = jnp.complex128 if dtype == "float64" else jnp.complex64
        px, py = _separable_axes(pupil_grid)
        fx, fy = _separable_axes(focal_grid)
        lams = np.asarray(filter_lams, dtype=np.float64)

        # Uniform pixel weight (pupil pixel area). make_pupil_grid produces
        # regular grids; guard anyway so a non-uniform grid fails loudly.
        weights = np.atleast_1d(np.asarray(pupil_grid.weights, dtype=np.float64))
        if weights.size > 1 and not np.all(weights == weights.flat[0]):
            raise ValueError("non-uniform pupil-grid weights are not supported")
        w_in = float(weights.flat[0])

        # Per-λ kernels: K1[l] (Nfy, Ny) applies the y-axis transform,
        # K2[l] (Nx, Nfx) the x-axis: E_f = K1 @ (E_p · w_in) @ K2.
        scale = 2.0 * np.pi / (focal_length * lams)  # (Nλ,)
        k1 = np.exp(-1j * scale[:, None, None] * fy[None, :, None] * py[None, None, :])
        k2 = np.exp(-1j * scale[:, None, None] * px[None, :, None] * fx[None, None, :])

        self.pupil_shape = (py.size, px.size)
        self.focal_shape = (fy.size, fx.size)
        self.filter_lams = lams
        self.real_dtype = real_dtype
        self._w_in = w_in
        # |1/(i·λ·f)|² intensity normalization per λ.
        self._int_norm = jnp.asarray(1.0 / (focal_length * lams) ** 2, dtype=real_dtype)
        self._k1 = jnp.asarray(k1, dtype=complex_dtype)
        self._k2 = jnp.asarray(k2, dtype=complex_dtype)
        self._lams = jnp.asarray(lams, dtype=real_dtype)

        @jax.jit
        def _summed_intensity(amplitude: jnp.ndarray, opd: jnp.ndarray) -> jnp.ndarray:
            def one_lam(k1_l, k2_l, lam, norm_l):
                field = amplitude * jnp.exp(1j * (2.0 * jnp.pi / lam) * opd)
                e_focal = k1_l @ (field * w_in) @ k2_l
                return norm_l * jnp.abs(e_focal) ** 2

            per_lam = jax.vmap(one_lam)(self._k1, self._k2, self._lams, self._int_norm)
            return per_lam.sum(axis=0)

        self._summed_intensity = _summed_intensity

    def summed_intensity(
        self,
        amplitude: NDArray[np.floating],
        opd: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        """Sum of per-wavelength focal intensities for a pupil field.

        Parameters
        ----------
        amplitude
            Real pupil-plane amplitude (the aperture transmission), flat or
            2-D.
        opd
            Pupil-plane optical path difference in meters, flat or 2-D.

        Returns
        -------
        (focal_res_y, focal_res_x) array in this propagator's real dtype.
        """
        amp = jnp.asarray(
            np.asarray(amplitude, dtype=np.float64).reshape(self.pupil_shape),
            dtype=self.real_dtype,
        )
        opd2d = jnp.asarray(
            np.asarray(opd, dtype=np.float64).reshape(self.pupil_shape), dtype=self.real_dtype
        )
        return np.asarray(self._summed_intensity(amp, opd2d))


__all__ = ["FraunhoferMFT"]
