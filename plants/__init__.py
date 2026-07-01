# FILE: plants/__init__.py
"""Robot plant models for simulation and control.

Provides concrete plant implementations that wrap robot kinematics and
optionally attach a MuJoCo physics engine for mesh-accurate simulation.

Available plants:
    ArmRobot               — 6-DOF serial-link arm with FK, Jacobian, IK
    HolonomicMobileRobot   — N-wheel holonomic base with omni-wheel kinematics
"""
from .armrobot import ArmRobot
from .holonomicmobilerobot import HolonomicMobileRobot

__all__ = ["ArmRobot", "HolonomicMobileRobot"]
