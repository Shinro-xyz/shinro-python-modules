# FILE: controllers/lerobot_adapter.py (70 lines)
"""
LeRobot policy adapter — wraps any Hugging Face LeRobot policy as a Controller.

Allows swapping between classical control (LQR, MPC) and learned policies
(diffusion, ACT, pi0, smolvla, etc.) with the same --controller flag.

Usage:
    # In configs/controllers/lerobot_diffusion.toml:
    #   type = "lerobot_diffusion"
    #   policy_type = "diffusion"
    #   checkpoint = "huggingface/lerobot-diffusion-lekiwi"
    #   use_camera = false
    #
    # python -m demos.demo_base_tracking --controller lerobot_diffusion
"""

from typing import Optional, Union
from components import Controller
from factories.registry import register_controller


@register_controller("lerobot_diffusion")
class LeRobotDiffusionAdapter(Controller):
    """Wrap a LeRobot diffusion policy as a Controller.

    The policy is loaded from a Hugging Face checkpoint or local path.
    The `compute()` method converts the plant state into LeRobot's
    observation dict format, runs the policy, and returns the action.

    Accepts both numpy arrays and torch tensors for state and camera
    frames, so it works with any ArrayBackend.
    """

    def __init__(self, policy, use_camera: bool = False, device: str = "cpu"):
        self.policy = policy
        self.use_camera = use_camera
        self.device = device
        self._latest_camera_frame = None

    def update_camera(self, frame):
        """Feed a camera frame for the next policy step."""
        self._latest_camera_frame = frame

    def compute(self, state, target=None):
        """Run the LeRobot policy on the current state.

        Args:
            state: Plant state vector (n_x,). Can be numpy or torch tensor.
            target: Ignored for learned policies.

        Returns:
            Action vector (n_u,) as numpy array.
        """
        import torch

        if isinstance(state, torch.Tensor):
            obs = {"observation.state": state.float().unsqueeze(0)}
        else:
            obs = {"observation.state": torch.from_numpy(state).float().unsqueeze(0)}

        if self.use_camera and self._latest_camera_frame is not None:
            frame = self._latest_camera_frame
            if isinstance(frame, torch.Tensor):
                frame_tensor = frame.float()
            else:
                frame_tensor = torch.from_numpy(frame).float()
            if frame_tensor.ndim == 3:
                frame_tensor = frame_tensor.permute(2, 0, 1)
            obs["observation.images.cam"] = frame_tensor.unsqueeze(0)
            self._latest_camera_frame = None

        obs = {k: v.to(self.device) for k, v in obs.items()}
        with torch.no_grad():
            action = self.policy.select_action(obs)

        return action.squeeze(0).cpu().numpy()

    def reset(self):
        """Reset the policy's internal state."""
        self.policy.reset()
        self._latest_camera_frame = None

    @classmethod
    def from_config(cls, config):
        """Load a LeRobot policy from a config dict.

        Config fields:
            policy_type:   Type of policy ("diffusion", "act", "pi0", etc.)
            checkpoint:    Hugging Face repo ID or local path
            use_camera:    Whether to expect camera observations (default: false)
            device:        Torch device (default: "cpu")
        """
        policy_type = config["policy_type"]
        checkpoint = config["checkpoint"]
        use_camera = config.get("use_camera", False)
        device = config.get("device", "cpu")

        from lerobot.policies import make_policy
        policy = make_policy(policy_type, pretrained_path=checkpoint)
        policy.to(device)
        policy.eval()

        return cls(policy, use_camera=use_camera, device=device)