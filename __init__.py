# FILE: __init__.py
"""lerobot-mpc-lekiwi — Whole-body control framework for the lekiwi robot.

A modular robotics control stack with four abstract base classes and
concrete implementations for trajectory generation, control, state
estimation, and plant dynamics.

Subpackages:
    trajectories/  — Reference path generators (CubicPolynomial, QuinticPolynomial)
    controllers/   — Control algorithms (LQR, PID, MPC)
    plants/        — Robot models (ArmRobot, HolonomicMobileRobot)
    estimators/    — State estimation (KalmanFilter, LuenbergerObserver)
"""
