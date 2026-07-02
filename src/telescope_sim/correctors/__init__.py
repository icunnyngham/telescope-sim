"""Corrector implementations.

Concrete classes register themselves at import time via
``@register("corrector", "<name>")``. Built-ins: ``segmented_ptt``
(per-segment piston/tip/tilt), ``zernike`` (Zernike-mode basis), and
``actuator_grid`` (N×N influence-function DM with gaussian or xinetics
actuator shapes and baked-in rotation/flip misalignment). Planned:
``fourier`` (Fourier-mode basis) and ``prebuilt`` (wraps a
pre-constructed HCIPy ``DeformableMirror``).
"""
