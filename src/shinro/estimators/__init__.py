# FILE: estimators/__init__.py
"""State estimation algorithms for reconstructing system state from measurements.

Provides discrete-time state estimators that combine dynamics models with
sensor measurements. All estimators implement the StateEstimator ABC.

Available estimators:
    KalmanFilter         — Optimal stochastic filter (predict-update cycle)
    ExtendedKalmanFilter — Nonlinear EKF (local linearization of f, h)
    UnscentedKF          — Nonlinear UKF (deterministic sigma-point propagation)
    LuenbergerObserver   — Deterministic observer with fixed gain
"""
from .extended_kf import ExtendedKalmanFilter
from .kalman_filter import KalmanFilter
from .luenberger_observer import LuenbergerObserver
from .unscented_kf import UnscentedKF

__all__ = ["KalmanFilter", "ExtendedKalmanFilter", "UnscentedKF", "LuenbergerObserver"]
