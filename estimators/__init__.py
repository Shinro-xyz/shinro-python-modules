# FILE: estimators/__init__.py
"""State estimation algorithms for reconstructing system state from measurements.

Provides discrete-time state estimators that combine dynamics models with
sensor measurements. All estimators implement the StateEstimator ABC.

Available estimators:
    KalmanFilter        — Optimal stochastic filter (predict-update cycle)
    LuenbergerObserver  — Deterministic observer with fixed gain
"""
from .kalman_filter import KalmanFilter
from .luenberger_observer import LuenbergerObserver

__all__ = ["KalmanFilter", "LuenbergerObserver"]
