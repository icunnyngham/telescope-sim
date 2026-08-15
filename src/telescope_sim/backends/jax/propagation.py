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
    coronagraph
        Optional bound coronagraph implementation, dispatched on its
        ``name``. When given, a second jitted propagation path applies
        the coronagraph train in the pupil plane before the final
        transform; ``summed_intensity(..., coronagraph=True)`` selects
        it and the plain path remains for the reference PSF.

        ``"lyot"`` (see ``coronagraphs/lyot.py``): per wavelength,

            E_lyot = L · (E − P⁻[occulter · P⁺[E]])

        with P⁺/P⁻ matrix Fourier transforms between the pupil and the
        small mask grid mirroring ``hcipy.FraunhoferPropagator``'s
        forward/backward conventions (Babinet's principle — the same
        scheme :class:`hcipy.LyotCoronagraph` uses, sharing the exact
        geometry arrays).

        ``"vortex"`` / ``"vector_vortex"`` (see
        ``coronagraphs/standard.py``): hcipy's multi-scale scheme
        replayed from the exact per-level masks the bound hcipy object
        precomputes — level 0 as an FFT filter, finer levels as λ=1 MFT
        round trips. The vortex phase is scale-invariant, so the train
        is wavelength-independent (hcipy evaluates it at unit
        wavelength too) and ONE kernel set serves the whole band. The
        vector variant runs two half-weight scalar channels at charges
        ±c (the circular-basis decomposition of the π-retardance
        plate) and averages intensities.
    """

    def __init__(
        self,
        pupil_grid: Any,
        focal_grid: Any,
        filter_lams: NDArray[np.floating],
        *,
        focal_length: float = 1.0,
        dtype: str = "float64",
        coronagraph: Any | None = None,
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
        # Science-path propagation: identical to _summed_intensity unless a
        # coronagraph is installed below. forward_fn and the focal planes'
        # per-sample path route through this; the reference PSF always uses
        # _summed_intensity (coronagraph bypassed by convention).
        self._science_intensity = _summed_intensity
        self._summed_intensity_coro = None
        if coronagraph is not None:
            kw = dict(py=py, px=px, complex_dtype=complex_dtype, real_dtype=real_dtype, w_in=w_in)
            coro_name = getattr(coronagraph, "name", None)
            if coro_name == "lyot":
                self._build_lyot_path(coronagraph, **kw)
            elif coro_name in ("vortex", "vector_vortex"):
                self._build_vortex_path(coronagraph, **kw)
            else:
                raise ValueError(
                    f"coronagraph {coro_name!r} has no jax propagation path "
                    "(supported: lyot, vortex, vector_vortex)."
                )

    def _build_lyot_path(self, lyot, *, py, px, complex_dtype, real_dtype, w_in):
        """Build the jitted Lyot-train propagation (see class docstring)."""
        mx, my = _separable_axes(lyot.mask_grid)
        spot2d = np.asarray(lyot.occulter, dtype=np.float64).reshape((my.size, mx.size))

        stop = None
        if lyot.lyot_field is not None:
            stop_arr = np.asarray(lyot.lyot_field)
            if np.iscomplexobj(stop_arr):
                raise ValueError(
                    "complex Lyot-stop fields are not supported on the 'jax' "
                    "backend; use the hcipy backend for complex stops."
                )
            stop = jnp.asarray(stop_arr.reshape(self.pupil_shape), dtype=real_dtype)

        mask_weights = np.atleast_1d(np.asarray(lyot.mask_grid.weights, dtype=np.float64))
        if mask_weights.size > 1 and not np.all(mask_weights == mask_weights.flat[0]):
            raise ValueError("non-uniform mask-grid weights are not supported")
        w_mask = float(mask_weights.flat[0])

        # Per-λ MFT kernel pairs between the pupil and the small mask grid,
        # mirroring hcipy.FraunhoferPropagator: forward kernel exp(-i·s·u·x)
        # with norm 1/(i·λ·f); backward kernel exp(+i·s·x·u) with norm
        # i/(λ·f) (the M†-weights inverse: w_mask/(λf)² times i·λ·f).
        f_c = float(lyot.focal_length)
        lams = self.filter_lams
        s_c = 2.0 * np.pi / (f_c * lams)  # (Nλ,)
        kf1 = np.exp(-1j * s_c[:, None, None] * my[None, :, None] * py[None, None, :])
        kf2 = np.exp(-1j * s_c[:, None, None] * px[None, :, None] * mx[None, None, :])
        kb1 = np.exp(1j * s_c[:, None, None] * py[None, :, None] * my[None, None, :])
        kb2 = np.exp(1j * s_c[:, None, None] * mx[None, :, None] * px[None, None, :])
        norm_fwd = 1.0 / (1j * f_c * lams)
        norm_back = 1j / (f_c * lams)

        kf1 = jnp.asarray(kf1, dtype=complex_dtype)
        kf2 = jnp.asarray(kf2, dtype=complex_dtype)
        kb1 = jnp.asarray(kb1, dtype=complex_dtype)
        kb2 = jnp.asarray(kb2, dtype=complex_dtype)
        norm_fwd = jnp.asarray(norm_fwd, dtype=complex_dtype)
        norm_back = jnp.asarray(norm_back, dtype=complex_dtype)
        spot = jnp.asarray(spot2d, dtype=real_dtype)

        @jax.jit
        def _summed_intensity_coro(amplitude: jnp.ndarray, opd: jnp.ndarray) -> jnp.ndarray:
            def one_lam(operands):
                k1_l, k2_l, lam, norm_l, kf1_l, kf2_l, kb1_l, kb2_l, nf_l, nb_l = operands
                field = amplitude * jnp.exp(1j * (2.0 * jnp.pi / lam) * opd)
                e_mask = nf_l * (kf1_l @ (field * w_in) @ kf2_l)
                e_back = nb_l * (kb1_l @ ((spot * e_mask) * w_mask) @ kb2_l)
                e_lyot = field - e_back
                if stop is not None:
                    e_lyot = stop * e_lyot
                e_focal = k1_l @ (e_lyot * w_in) @ k2_l
                return norm_l * jnp.abs(e_focal) ** 2

            per_lam = jax.vmap(one_lam)(
                (
                    self._k1,
                    self._k2,
                    self._lams,
                    self._int_norm,
                    kf1,
                    kf2,
                    kb1,
                    kb2,
                    norm_fwd,
                    norm_back,
                )
            )
            return per_lam.sum(axis=0)

        self._summed_intensity_coro = _summed_intensity_coro
        self._science_intensity = _summed_intensity_coro

    def _build_vortex_path(self, vortex, *, py, px, complex_dtype, real_dtype, w_in):
        """Build the jitted multi-scale vortex train (see class docstring)."""
        stop = None
        if vortex.lyot_field is not None:
            stop_arr = np.asarray(vortex.lyot_field)
            if np.iscomplexobj(stop_arr):
                raise ValueError(
                    "complex Lyot-stop fields are not supported on the 'jax' "
                    "backend; use the hcipy backend for complex stops."
                )
            stop = jnp.asarray(stop_arr.reshape(self.pupil_shape), dtype=real_dtype)

        channels = [
            (
                float(weight),
                self._vortex_channel(msc, py=py, px=px, complex_dtype=complex_dtype, w_in=w_in),
            )
            for weight, msc in vortex._jax_multi_scale_sources()
        ]

        @jax.jit
        def _summed_intensity_coro(amplitude: jnp.ndarray, opd: jnp.ndarray) -> jnp.ndarray:
            def one_lam(k1_l, k2_l, lam, norm_l):
                field = amplitude * jnp.exp(1j * (2.0 * jnp.pi / lam) * opd)
                total = None
                for weight, train in channels:
                    e_lyot = train(field)
                    if stop is not None:
                        e_lyot = stop * e_lyot
                    e_focal = k1_l @ (e_lyot * w_in) @ k2_l
                    contrib = weight * jnp.abs(e_focal) ** 2
                    total = contrib if total is None else total + contrib
                return norm_l * total

            per_lam = jax.vmap(one_lam)(self._k1, self._k2, self._lams, self._int_norm)
            return per_lam.sum(axis=0)

        self._summed_intensity_coro = _summed_intensity_coro
        self._science_intensity = _summed_intensity_coro

    def _vortex_channel(self, msc, *, py, px, complex_dtype, w_in):
        """One scalar multi-scale train: pupil field → Lyot-plane field.

        Replays ``hcipy.MultiScaleCoronagraph.forward`` at λ=1 from the
        bound object's precomputed per-level masks (windowed and
        correction-subtracted by hcipy itself): level 0 through its FFT
        filter (zero-pad, fftn, ifftshifted-mask multiply, ifftn,
        cutout — no Fraunhofer norm, matching ``hcipy.FourierFilter``),
        finer levels as matrix-Fourier round trips with the λ=1
        Fraunhofer norms 1/i (forward) and i (backward).
        """
        ff = msc.props[0]
        internal_shape = tuple(int(s) for s in ff.internal_grid.shape)
        cutout = ff.cutout
        tf = jnp.asarray(
            np.fft.ifftshift(np.asarray(msc.focal_masks[0]).reshape(internal_shape)),
            dtype=complex_dtype,
        )

        s = 2.0 * np.pi  # 2π/(λf) at λ = f = 1
        levels = []
        for i in range(1, len(msc.focal_masks)):
            grid = msc.props[i]._output_grid
            fx, fy = _separable_axes(grid)
            mask2d = np.asarray(msc.focal_masks[i]).reshape((fy.size, fx.size))
            w_out = float(np.atleast_1d(np.asarray(grid.weights, dtype=np.float64)).flat[0])
            levels.append(
                (
                    jnp.asarray(np.exp(-1j * s * np.outer(fy, py)), dtype=complex_dtype),
                    jnp.asarray(np.exp(-1j * s * np.outer(px, fx)), dtype=complex_dtype),
                    jnp.asarray(np.exp(1j * s * np.outer(py, fy)), dtype=complex_dtype),
                    jnp.asarray(np.exp(1j * s * np.outer(fx, px)), dtype=complex_dtype),
                    jnp.asarray(mask2d, dtype=complex_dtype),
                    w_out,
                )
            )

        def train(field: jnp.ndarray) -> jnp.ndarray:
            f = jnp.zeros(internal_shape, dtype=complex_dtype).at[cutout].set(field)
            lyot = jnp.fft.ifftn(jnp.fft.fftn(f) * tf)[cutout]
            for kf1, kf2, kb1, kb2, mask, w_out in levels:
                e_focal = -1j * (kf1 @ (field * w_in) @ kf2)
                lyot = lyot + 1j * (kb1 @ ((mask * e_focal) * w_out) @ kb2)
            return lyot

        return train

    def summed_intensity(
        self,
        amplitude: NDArray[np.floating],
        opd: NDArray[np.floating],
        *,
        coronagraph: bool = False,
    ) -> NDArray[np.float64]:
        """Sum of per-wavelength focal intensities for a pupil field.

        Parameters
        ----------
        amplitude
            Real pupil-plane amplitude (the aperture transmission), flat or
            2-D.
        opd
            Pupil-plane optical path difference in meters, flat or 2-D.
        coronagraph
            Propagate through the installed coronagraph train (requires a
            ``lyot`` object at construction). False — the reference-PSF
            convention — propagates the plain perfect-lens path.

        Returns
        -------
        (focal_res_y, focal_res_x) array in this propagator's real dtype.
        """
        if coronagraph and self._summed_intensity_coro is None:
            raise RuntimeError(
                "this propagator was built without a coronagraph; construct "
                "FraunhoferMFT with lyot=... to enable the coronagraphic path."
            )
        amp = jnp.asarray(
            np.asarray(amplitude, dtype=np.float64).reshape(self.pupil_shape),
            dtype=self.real_dtype,
        )
        opd2d = jnp.asarray(
            np.asarray(opd, dtype=np.float64).reshape(self.pupil_shape), dtype=self.real_dtype
        )
        fn = self._summed_intensity_coro if coronagraph else self._summed_intensity
        return np.asarray(fn(amp, opd2d))


__all__ = ["FraunhoferMFT"]
