from __future__ import annotations

from typing import Any

import torch

PUSHT_STATE_API = {
    "tee_xy": "float tensor [N, 2], current T-block world x/y in meters",
    "tee_yaw": "float tensor [N], current T-block yaw in radians",
    "goal_xy": "float tensor [N, 2], target world x/y in meters",
    "goal_yaw": "float tensor [N], target yaw in radians",
    "tcp_xy": "float tensor [N, 2], robot tool-center-point world x/y in meters",
    "overlap": "float tensor [N], current T/goal overlap in [0, 1]",
    "previous_overlap": "float tensor [N], overlap at the preceding environment step",
    "step_fraction": "float tensor [N], elapsed episode fraction in [0, 1]",
}


def planar_yaw(raw_pose: torch.Tensor) -> torch.Tensor:
    """Return yaw from ManiSkill [x, y, z, qw, qx, qy, qz] poses."""
    return 2.0 * torch.atan2(raw_pose[..., 6], raw_pose[..., 3])


def pusht_state_from_env(
    env: Any,
    previous_overlap: torch.Tensor,
    step_fraction: torch.Tensor | float,
) -> dict[str, torch.Tensor]:
    """Adapt a PushTEnv instance to the stable LLM-generated reward contract."""
    tee_pose = env.tee.pose.raw_pose
    goal_pose = env.goal_tee.pose.raw_pose
    tcp_pose = env.agent.tcp.pose.raw_pose
    overlap = env.pseudo_render_intersection()
    fraction = torch.as_tensor(step_fraction, dtype=overlap.dtype, device=overlap.device)
    if fraction.ndim == 0:
        fraction = fraction.expand_as(overlap)
    return {
        "tee_xy": tee_pose[..., :2],
        "tee_yaw": planar_yaw(tee_pose),
        "goal_xy": goal_pose[..., :2],
        "goal_yaw": planar_yaw(goal_pose),
        "tcp_xy": tcp_pose[..., :2],
        "overlap": overlap,
        "previous_overlap": previous_overlap,
        "step_fraction": fraction,
    }
