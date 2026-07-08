import numpy as np
import yaml
from pathlib import Path
from components import TrajectoryGenerator


class TrajectoryFactory:
    """Creates a callable trajectory schedule from a YAML config file.

    Generates a full sequence of setpoints for the simulation loop.

    Usage:
        schedule = TrajectoryFactory("trajectories/waypoints_config.yaml").create()
        setpoint = schedule[step]  # returns np.ndarray
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def create(self) -> np.ndarray | dict:
        ttype = self.config["type"]
        dt = self.config.get("dt", 0.02)

        if ttype == "waypoints":
            waypoints = self.config["waypoints"]
            schedule = []
            for wp in waypoints:
                duration = wp["duration"]
                n_steps = int(np.round(duration / dt))
                schedule.extend([np.array(wp["position"])] * n_steps)
            return np.array(schedule)

        elif ttype == "cubic_segments":
            segments = self.config["segments"]
            schedule = []
            for i, seg in enumerate(segments):
                duration = seg["duration"]
                n_steps = int(np.round(duration / dt))
                p_start = np.array(seg["start"])
                p_end = np.array(seg["end"])
                T = duration
                delta = p_end - p_start
                a0 = p_start
                a2 = 3.0 * delta / (T * T)
                a3 = -2.0 * delta / (T * T * T)
                for k in range(n_steps):
                    t_local = k * dt
                    pos = a0 + a2 * t_local**2 + a3 * t_local**3
                    schedule.append(pos.copy())
            return np.array(schedule)

        elif ttype == "phase_list":
            phases = self.config["phases"]
            arm_sched = []
            base_sched = []
            jaw_sched = []
            for phase in phases:
                n_steps = int(np.round(phase["duration"] / dt))
                arm = np.array(phase["arm"])
                base = np.array(phase["base"])
                jaw = float(phase["jaw"])
                for _ in range(n_steps):
                    arm_sched.append(arm.copy())
                    base_sched.append(base.copy())
                    jaw_sched.append(jaw)
            return {
                "arm": np.array(arm_sched),
                "base": np.array(base_sched),
                "jaw": np.array(jaw_sched),
            }

        else:
            raise ValueError(f"Unknown trajectory type: {ttype}")