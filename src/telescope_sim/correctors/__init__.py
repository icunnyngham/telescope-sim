"""Corrector implementations.

Concrete classes register themselves at import time via
``@register("corrector", "<name>")``. Built-ins: ``segmented_ptt``
(per-segment piston/tip/tilt, or piston only, over aperture-defined or
partitioned segments), ``zernike`` (Zernike-mode basis), ``fourier``
(low-order Fourier-mode basis), and ``actuator_grid`` (N×N
influence-function DM with gaussian or xinetics actuator shapes and
baked-in rotation/flip misalignment). Planned: ``prebuilt`` (wraps a
pre-constructed HCIPy ``DeformableMirror``).
"""
