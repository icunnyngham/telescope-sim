"""PostProcessor implementations.

Concrete classes register themselves at import time via
``@register("post_processor", "<name>")``. Built-ins include
``max_intensity_norm``, ``max_image_norm``, ``per_sample_norm``, and
``channels_first``. ``power_scale``, ``gaussian_noise``, ``detector_noise``,
``fft_channels``, and ``convolve`` will follow as the variants that exercise
them come online.
"""

# Import side effects: register implementations
from telescope_sim.post import normalization  # noqa: F401
