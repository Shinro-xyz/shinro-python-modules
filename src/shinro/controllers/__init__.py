# FILE: controllers/__init__.py
"""Control algorithms for state regulation and trajectory tracking.

Provides discrete-time controllers that compute control actions from
state feedback. All controllers implement the Controller ABC.

Available controllers:
    LQR             — Linear Quadratic Regulator (DARE-based optimal gain)
    PIDController   — Proportional-Integral-Derivative with anti-windup
    MPC_LTI         — Linear Time-Invariant MPC with OSQP QP solver
    MPC_LTI_DeltaU  — MPC with Δu (control rate) regularization
"""
from .lerobot_adapter import LeRobotDiffusionAdapter
from .lqr import LQR
from .mpc_lti import MPC_LTI_DeltaU
from .pid import PIDController
from .smc import SlidingModeController

__all__ = ["LQR", "PIDController", "MPC_LTI_DeltaU", "LeRobotDiffusionAdapter", "SlidingModeController"]
