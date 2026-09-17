from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]

JOINT_NUM = 21
POLICY_HISTORY_LEN = 1
CRITIC_HISTORY_LEN = 3
POLICY_OBS_DIM = 72
CRITIC_OBS_DIM = 225

JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]
LEFT_JOINT_INDICES = [0, 1, 2, 3, 4, 5, 13, 14, 15, 16]
RIGHT_JOINT_INDICES = [6, 7, 8, 9, 10, 11, 17, 18, 19, 20]

# Lens110 URDF uses +X axes for roll joints and +Z axes for yaw joints on both sides.
# Mirroring across the sagittal plane therefore swaps left/right and flips roll/yaw signs.
SIGN_FLIP_JOINT_INDICES = [1, 2, 5, 7, 8, 11, 12, 14, 15, 18, 19]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Apply left-right symmetry augmentation for Lens110 21DOF observations and actions."""
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)

        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(obs["policy"])

        obs_aug["critic"][:batch_size] = obs["critic"][:]
        obs_aug["critic"][batch_size : 2 * batch_size] = _transform_critic_obs_left_right(obs["critic"])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size : 2 * batch_size] = _switch_lens110_joints_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _transform_policy_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    _validate_last_dim(obs, POLICY_OBS_DIM, "policy observations")
    obs = obs.clone()
    end_idx = 0

    end_idx = _flip_repeated_vectors(obs, end_idx, 3, POLICY_HISTORY_LEN, [-1, 1, -1])
    end_idx = _flip_repeated_vectors(obs, end_idx, 3, POLICY_HISTORY_LEN, [1, -1, 1])
    end_idx = _flip_repeated_vectors(obs, end_idx, 3, POLICY_HISTORY_LEN, [1, -1, -1])
    end_idx = _switch_repeated_joints(obs, end_idx, POLICY_HISTORY_LEN)
    end_idx = _switch_repeated_joints(obs, end_idx, POLICY_HISTORY_LEN)
    _switch_repeated_joints(obs, end_idx, POLICY_HISTORY_LEN)

    return obs


def _transform_critic_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    _validate_last_dim(obs, CRITIC_OBS_DIM, "critic observations")
    obs = obs.clone()
    end_idx = 0

    end_idx = _flip_repeated_vectors(obs, end_idx, 3, CRITIC_HISTORY_LEN, [1, -1, 1])
    end_idx = _flip_repeated_vectors(obs, end_idx, 3, CRITIC_HISTORY_LEN, [-1, 1, -1])
    end_idx = _flip_repeated_vectors(obs, end_idx, 3, CRITIC_HISTORY_LEN, [1, -1, 1])
    end_idx = _flip_repeated_vectors(obs, end_idx, 3, CRITIC_HISTORY_LEN, [1, -1, -1])
    end_idx = _switch_repeated_joints(obs, end_idx, CRITIC_HISTORY_LEN)
    end_idx = _switch_repeated_joints(obs, end_idx, CRITIC_HISTORY_LEN)
    _switch_repeated_joints(obs, end_idx, CRITIC_HISTORY_LEN)

    return obs


def _flip_repeated_vectors(
    obs: torch.Tensor,
    start_idx: int,
    dim: int,
    history_len: int,
    signs: list[int],
) -> int:
    signs_tensor = torch.tensor(signs, device=obs.device, dtype=obs.dtype)
    end_idx = start_idx
    for _ in range(history_len):
        next_idx = end_idx + dim
        obs[:, end_idx:next_idx] = obs[:, end_idx:next_idx] * signs_tensor
        end_idx = next_idx
    return end_idx


def _switch_repeated_joints(obs: torch.Tensor, start_idx: int, history_len: int) -> int:
    end_idx = start_idx
    for _ in range(history_len):
        next_idx = end_idx + JOINT_NUM
        obs[:, end_idx:next_idx] = _switch_lens110_joints_left_right(obs[:, end_idx:next_idx])
        end_idx = next_idx
    return end_idx


def _switch_lens110_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Mirror Lens110 joint data in lab joint order.

    Joint order:
    left leg 0-5, right leg 6-11, torso_yaw 12,
    left arm 13-16, right arm 17-20.
    """
    _validate_last_dim(joint_data, JOINT_NUM, "Lens110 joint data")
    joint_data_switched = torch.zeros_like(joint_data)

    joint_data_switched[..., LEFT_JOINT_INDICES] = joint_data[..., RIGHT_JOINT_INDICES]
    joint_data_switched[..., RIGHT_JOINT_INDICES] = joint_data[..., LEFT_JOINT_INDICES]
    joint_data_switched[..., 12] = joint_data[..., 12]
    joint_data_switched[..., SIGN_FLIP_JOINT_INDICES] *= -1.0

    return joint_data_switched


def _validate_last_dim(tensor: torch.Tensor, expected_dim: int, name: str) -> None:
    if tensor.shape[-1] != expected_dim:
        raise ValueError(
            f"Expected {name} last dimension to be {expected_dim}, got {tensor.shape[-1]}. "
            "Update legged_lab/tasks/locomotion/amp/mdp/symmetry/lens110.py if Lens110 "
            "joint order or observation terms changed."
        )
