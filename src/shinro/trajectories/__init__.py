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
    CatmullRom        — C¹ cubic Hermite through an ordered waypoint list

Sampling helpers:
    sample_schedule / sample_segments — stacked (steps, N) position/velocity/
        acceleration schedules (the ``derivatives`` config opt-in uses them).
"""
from .b_spline import BSpline
from .bezier_curve import BezierCurve
from .catmull_rom import CatmullRom
from .cubic_polynomial import CubicPolynomial
from .lissajous import Lissajous
from .quintic_polynomial import QuinticPolynomial
from .sampling import sample_schedule, sample_segments

__all__ = [
    "CubicPolynomial",
    "QuinticPolynomial",
    "Lissajous",
    "BezierCurve",
    "BSpline",
    "CatmullRom",
    "sample_schedule",
    "sample_segments",
]
