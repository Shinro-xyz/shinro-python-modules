# FILE: trajectories/__init__.py
"""Reference path generators for smooth point-to-point motion.

Provides polynomial trajectory generators that compute position, velocity,
and acceleration profiles from boundary conditions. All generators support
arbitrary N-dimensional positions via numpy broadcasting.

Available generators:
    CubicPolynomial   — 3rd-order, position + velocity continuity
    QuinticPolynomial — 5th-order, position + velocity + acceleration continuity
"""
from .cubic_polynomial import CubicPolynomial
from .quintic_polynomial import QuinticPolynomial

__all__ = ["CubicPolynomial", "QuinticPolynomial"]
