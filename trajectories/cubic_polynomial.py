from typing import Optional
from components import TrajectoryGenerator
from factories.registry import register_trajectory
from utils.array_backend import ArrayBackend, NumpyBackend
import numpy as np


@register_trajectory("cubic_segments")
class CubicPolynomial(TrajectoryGenerator):
    def __init__(self, backend: Optional[ArrayBackend] = None):
        self.bk = backend or NumpyBackend()

    def generate(
        self,
        start_position,
        end_position,
        duration: float,
        start_vel,
        end_vel,
    ):
        self.a0 = start_position
        self.a1 = start_vel
        a2_numerator = 3 * (end_position - start_position) - duration * (2 * start_vel + end_vel)
        self.a2 = a2_numerator / (duration ** 2)

        a3_numerator = -2 * (end_position - start_position) + duration * (start_vel + end_vel)
        self.a3 = a3_numerator / (duration ** 3)

        self.duration = duration

    def position_at(self, t: float):
        t = self.bk.clip(t, 0, self.duration)
        pos = self.a0 + self.a1 * t + self.a2 * t ** 2 + self.a3 * t ** 3
        vel = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t ** 2
        acc = 2 * self.a2 + 6 * self.a3 * t
        return pos, vel, acc

    @classmethod
    def from_config(cls, config, backend: Optional[ArrayBackend] = None):
        bk = backend or NumpyBackend()
        dt = config["dt"]
        schedule = []
        default_vel = bk.array([0.0, 0.0, 0.0])
        for seg in config["segments"]:
            n_steps = int(np.round(seg["duration"] / dt))
            p0 = bk.array(seg["start"])
            pf = bk.array(seg["end"])
            T = seg["duration"]
            start_vel = bk.array(seg.get("start_vel", [0.0, 0.0, 0.0]))
            end_vel = bk.array(seg.get("end_vel", [0.0, 0.0, 0.0]))
            traj = cls(backend=bk)
            traj.generate(p0, pf, T, start_vel, end_vel)
            for k in range(n_steps):
                t = k * dt
                pos, _, _ = traj.position_at(t)
                schedule.append(pos)
        return bk.array(schedule)
