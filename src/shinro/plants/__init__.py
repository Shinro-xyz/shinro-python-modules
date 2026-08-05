# FILE: plants/__init__.py
"""Robot plant models for simulation and control.

Provides concrete plant implementations that wrap robot kinematics and
optionally attach a MuJoCo physics engine for mesh-accurate simulation.

Available plants:
    ArmRobot               — 6-DOF serial-link arm with FK, Jacobian, IK
    HolonomicMobileRobot   — N-wheel holonomic base with omni-wheel kinematics
    InvertedPendulum       — 2D inverted pendulum with analytical dynamics
    CartPole               — 4D cart-pole with coupled dynamics
    Quadrotor              — 12D quadrotor (placeholder)
"""
from .armrobot import ArmRobot
from .cartpole import CartPole
from .holonomicmobilerobot import HolonomicMobileRobot
from .inverted_pendulum import InvertedPendulum
from .quadrotor import Quadrotor

__all__ = ["ArmRobot", "HolonomicMobileRobot", "InvertedPendulum", "CartPole", "Quadrotor"]
