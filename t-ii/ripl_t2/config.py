from __future__ import annotations

from dataclasses import dataclass

from ripl.config import EvalConfig, parse_config


@dataclass
class PushTFailureEvalConfig(EvalConfig):
    failure_mode_name: str = "discovery"
    x_rel_min: float = -0.10
    x_rel_max: float = 0.10
    y_rel_min: float = -0.10
    y_rel_max: float = 0.20
    theta_deg_min: float = 0.0
    theta_deg_max: float = 360.0
    save_trajectories: bool = True


def parse_failure_eval_config(argv: list[str] | None = None) -> PushTFailureEvalConfig:
    return parse_config(PushTFailureEvalConfig, argv)
