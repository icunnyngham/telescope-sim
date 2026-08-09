"""Compute-backend packages.

Each subpackage registers backend-specific implementations of pipeline
stages via ``@register(kind, name, backend="<backend>")``. Importing a
backend package is what populates its registry overlay; the config loader
does this on demand when a config selects the backend.
"""
