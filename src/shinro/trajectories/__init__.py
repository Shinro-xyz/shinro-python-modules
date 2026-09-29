# FILE: trajectories/__init__.py
"""Reference path generators for smooth point-to-point motion.

Provides polynomial and harmonic trajectory generators that compute position,
velocity, and acceleration profiles from boundary conditions. All generators
support arbitrary N-dimensional positions via numpy broadcasting.

Available generators:
    CubicPolynomial   — 3rd-order, position + velocity continuity
    QuinticPolynomial — 5th-order, position + velocity + acceleration continuity
    Lissajous         — odd-harmonic rest-to-rest figure, orientable via R
    BezierCurve       — Bernstein curve over an arbitrary control-point list
    BSpline           — Cox–de Boor B-spline over a knot vector and control polygon
"""
from .b_spline import BSpline
from .bezier_curve import BezierCurve
from .cubic_polynomial import CubicPolynomial
from .lissajous import Lissajous
from .quintic_polynomial import QuinticPolynomial

__all__ = ["CubicPolynomial", "QuinticPolynomial", "Lissajous", "BezierCurve", "BSpline"]
