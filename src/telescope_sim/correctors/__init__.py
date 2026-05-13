"""Corrector implementations.

Concrete classes register themselves at import time via
``@register("corrector", "<name>")``. Planned built-ins include ``xinetics``
(influence-function DM), ``zernike`` (Zernike-mode basis), ``fourier``
(Fourier-mode basis), ``segmented_ptt`` (per-segment piston/tip/tilt), and
``prebuilt`` (wraps a pre-constructed HCIPy ``DeformableMirror``).
"""
