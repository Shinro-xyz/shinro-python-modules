from typing import Optional
from components import TrajectoryGenerator
from factories.registry import register_trajectory
from utils.array_backend import ArrayBackend, NumpyBackend
import numpy as np


@register_trajectory("quintic_segments")
class QuinticPolynomial(TrajectoryGenerator):
    def __init__(self, backend: Optional[ArrayBackend] = None):
        self.bk = backend or NumpyBackend()

    def generate(
        self,
        start_position,
        end_position,
        duration: float,
        start_vel=None,
        end_vel=None,
        start_acc=None,
        end_acc=None,
    ):
        start_vel = self.bk.zeros_like(start_position) if start_vel is None else start_vel
        start_acc = self.bk.zeros_like(start_position) if start_acc is None else start_acc
        end_vel = self.bk.zeros_like(end_position) if end_vel is None else end_vel
        end_acc = self.bk.zeros_like(end_position) if end_acc is None else end_acc

        self.T = duration
        T = duration
        M = self.bk.array([
            [0, 0, 0, 0, 0, 1],
            [T ** 5, T ** 4, T ** 3, T ** 2, T, 1],
            [0, 0, 0, 0, 1, 0],
            [5 * T ** 4, 4 * T ** 3, 3 * T ** 2, 2 * T, 1, 0],
            [0, 0, 0, 2, 0, 0],
            [20 * T ** 3, 12 * T ** 2, 6 * T, 2, 0, 0],
        ])

        b = [start_position, end_position, start_vel, end_vel, start_acc, end_acc]

        coeff_vectors = self.bk.solve(M, b)

        self.A = coeff_vectors[0]
        self.B = coeff_vectors[1]
        self.C = coeff_vectors[2]
        self.D = coeff_vectors[3]
        self.E = coeff_vectors[4]
        self.F = coeff_vectors[5]

    def position_at(self, t: float):
        t = self.bk.clip(t, 0, self.T)
        pos = self.A * t ** 5 + self.B * t ** 4 + self.C * t ** 3 + self.D * t ** 2 + self.E * t + self.F
        vel = 5 * self.A * t ** 4 + 4 * self.B * t ** 3 + 3 * self.C * t ** 2 + 2 * self.D * t + self.E
        acc = 20 * self.A * t ** 3 + 12 * self.B * t ** 2 + 6 * self.C * t + 2 * self.D
        return pos, vel, acc


@register_trajectory("quintic_segments")
class QuinticPolynomialConfigAdapter:
    @classmethod
    def from_config(cls, config, backend: Optional[ArrayBackend] = None):
        bk = backend or NumpyBackend()
        dt = config["dt"]
        schedule = []
        for seg in config["segments"]:
            n_steps = int(np.round(seg["duration"] / dt))
            p0 = bk.array(seg["start"])
            pf = bk.array(seg["end"])
            T = seg["duration"]
            start_vel = bk.array(seg.get("start_vel", [0.0, 0.0, 0.0]))
            end_vel = bk.array(seg.get("end_vel", [0.0, 0.0, 0.0]))
            start_acc = bk.array(seg.get("start_acc", [0.0, 0.0, 0.0]))
            end_acc = bk.array(seg.get("end_acc", [0.0, 0.0, 0.0]))
            traj = QuinticPolynomial(backend=bk)
            traj.generate(p0, pf, T, start_vel, end_vel, start_acc, end_acc)
            for k in range(n_steps):
                t = k * dt
                pos, _, _ = traj.position_at(t)
                schedule.append(pos)
        return bk.array(schedule)


@register_trajectory("waypoints")
class WaypointSchedule:
    @classmethod
    def from_config(cls, config, backend: Optional[ArrayBackend] = None):
        bk = backend or NumpyBackend()
        dt = config["dt"]
        schedule = []
        for wp in config["waypoints"]:
            n_steps = int(np.round(wp["duration"] / dt))
            schedule.extend([bk.array(wp["position"])] * n_steps)
        return bk.array(schedule)


@register_trajectory("phase_list")
class PhaseSchedule:
    @classmethod
    def from_config(cls, config, backend: Optional[ArrayBackend] = None):
        bk = backend or NumpyBackend()
        dt = config["dt"]
        arm_sched = []
        base_sched = []
        jaw_sched = []
        for phase in config["phases"]:
            n_steps = int(np.round(phase["duration"] / dt))
            arm = bk.array(phase["arm"])
            base = bk.array(phase["base"])
            jaw = float(phase["jaw"])
            for _ in range(n_steps):
                arm_sched.append(bk.copy(arm))
                base_sched.append(bk.copy(base))
                jaw_sched.append(jaw)
        return {"arm": bk.array(arm_sched), "base": bk.array(base_sched), "jaw": bk.array(jaw_sched)}
