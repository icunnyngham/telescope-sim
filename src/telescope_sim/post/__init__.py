"""PostProcessor implementations.

Concrete classes register themselves at import time via
``@register("post_processor", "<name>")``. Built-ins:

- ``max_intensity_norm``, ``max_image_norm``, ``per_sample_norm``,
  ``channels_first`` — basic image transforms (``normalization``)
- ``noisy_detector`` — HCIPy ``NoisyDetector`` integration
- ``convolve_image`` — convolve PSF with a caller-provided scene

Two of the above (``noisy_detector`` and ``convolve_image``) implement the
:class:`LoaderBindable` hook to receive focal-grid / aperture-area /
reference-PSF-sum from the YAML loader at sim-build time. See the
individual module docstrings for details.

``power_scale``, ``gaussian_noise``, ``fft_channels`` will follow as the
variants that exercise them come online.
"""

# Import side effects: register implementations
from telescope_sim.post import (
    convolve,  # noqa: F401
    noisy_detector,  # noqa: F401
    normalization,  # noqa: F401
)
