"""PostProcessor implementations.

Concrete classes register themselves at import time via
``@register("post_processor", "<name>")``. Planned built-ins include
``max_intensity_norm``, ``max_image_norm``, ``per_sample_norm``,
``power_scale``, ``gaussian_noise``, ``detector_noise``, ``fft_channels``,
``channels_first``, and ``convolve``.
"""
