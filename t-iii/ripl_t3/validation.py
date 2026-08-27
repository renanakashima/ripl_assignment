from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

FORBIDDEN_CALLS = {
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
}
FORBIDDEN_NAMES = {
    "builtins",
    "ctypes",
    "importlib",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
}
FORBIDDEN_TORCH_ATTRIBUTES = {
    "classes",
    "compile",
    "distributed",
    "from_file",
    "hub",
    "jit",
    "load",
    "multiprocessing",
    "ops",
    "package",
    "save",
    "serialization",
    "utils",
}
GENERATED_CODE_ERRORS = (
    ArithmeticError,
    AssertionError,
    AttributeError,
    IndexError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass
class ValidationReport:
    passed: bool = True
    checks: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def check(self, name: str, condition: bool, message: str) -> None:
        self.checks[name] = bool(condition)
        if not condition:
            self.passed = False
            self.errors.append(message)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GeneratedCodeVisitor(ast.NodeVisitor):
    def __init__(self, expected_function: str) -> None:
        self.errors: list[str] = []
        self.expected_function = expected_function

    def visit_Import(self, node: ast.Import) -> None:
        self.errors.append(f"Imports are forbidden (line {node.lineno})")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.errors.append(f"Imports are forbidden (line {node.lineno})")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
            self.errors.append(f"Forbidden name {node.id!r} at line {node.lineno}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            self.errors.append(f"Dunder attribute access at line {node.lineno}")
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "torch"
            and node.attr in FORBIDDEN_TORCH_ATTRIBUTES
        ):
            self.errors.append(f"Forbidden torch capability {node.attr!r} at line {node.lineno}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            self.errors.append(f"Forbidden call {node.func.id!r} at line {node.lineno}")
        if isinstance(node.func, ast.Name) and node.func.id == self.expected_function:
            self.errors.append(f"Recursive calls are forbidden at line {node.lineno}")
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.errors.append(
            f"Loops are forbidden; use batched tensor operations (line {node.lineno})"
        )

    def visit_While(self, node: ast.While) -> None:
        self.errors.append(
            f"Loops are forbidden; use batched tensor operations (line {node.lineno})"
        )


def inspect_source(source: str, expected_function: str) -> tuple[ast.Module | None, list[str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return None, [f"Syntax error: {error}"]
    visitor = GeneratedCodeVisitor(expected_function)
    visitor.visit(tree)
    top_level_functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    if top_level_functions != [expected_function]:
        visitor.errors.append(
            f"Expected exactly one function named {expected_function!r}; got {top_level_functions}"
        )
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            visitor.errors.append("Only the required function may appear at module scope")
    return tree, visitor.errors


def load_generated_function(source: str, expected_function: str):
    tree, errors = inspect_source(source, expected_function)
    if errors or tree is None:
        raise ValueError("; ".join(errors))
    safe_builtins = {
        "abs": abs,
        "bool": bool,
        "float": float,
        "int": int,
        "max": max,
        "min": min,
        "range": range,
    }
    namespace: dict[str, Any] = {"__builtins__": safe_builtins, "torch": torch}
    exec(compile(tree, "<llm-generated>", "exec"), namespace)  # noqa: S102
    return namespace[expected_function]


def _example_state(batch_size: int = 16) -> dict[str, torch.Tensor]:
    goal_xy = torch.tensor([-0.156, -0.100]).repeat(batch_size, 1)
    return {
        "tee_xy": goal_xy + torch.linspace(-0.08, 0.08, batch_size)[:, None].repeat(1, 2),
        "tee_yaw": torch.linspace(0.0, 2.0 * torch.pi, batch_size),
        "goal_xy": goal_xy,
        "goal_yaw": torch.full((batch_size,), torch.deg2rad(torch.tensor(300.0))),
        "tcp_xy": goal_xy + 0.04,
        "overlap": torch.linspace(0.0, 1.0, batch_size),
        "previous_overlap": torch.linspace(0.0, 0.9, batch_size),
        "step_fraction": torch.linspace(0.0, 1.0, batch_size),
    }


def validate_reward_code(source: str) -> ValidationReport:
    report = ValidationReport()
    _, errors = inspect_source(source, "compute_dense_reward")
    report.check("static_policy", not errors, "; ".join(errors))
    if errors:
        return report
    try:
        function = load_generated_function(source, "compute_dense_reward")
        result = function(_example_state())
    except GENERATED_CODE_ERRORS as error:
        report.check("executes", False, f"Reward execution failed: {type(error).__name__}: {error}")
        return report
    report.check("returns_dict", isinstance(result, dict), "Reward must return a dictionary")
    if not isinstance(result, dict):
        return report
    report.check("has_reward", "reward" in result, "Reward result must contain `reward`")
    tensors_ok = all(torch.is_tensor(value) and value.shape == (16,) for value in result.values())
    report.check("batched_components", tensors_ok, "Every reward component must be a [N] tensor")
    finite = tensors_ok and all(torch.isfinite(value).all().item() for value in result.values())
    report.check("finite", finite, "Reward components must be finite")

    if "reward" in result and torch.is_tensor(result["reward"]) and result["reward"].shape == (16,):
        perfect = _example_state(4)
        perfect["tee_xy"] = perfect["goal_xy"].clone()
        perfect["tee_yaw"] = perfect["goal_yaw"].clone()
        perfect["overlap"] = torch.ones(4)
        perfect["previous_overlap"] = torch.full((4,), 0.95)
        poor = _example_state(4)
        poor["tee_xy"] = poor["goal_xy"] + 0.20
        poor["tee_yaw"] = poor["goal_yaw"] + torch.pi
        poor["overlap"] = torch.zeros(4)
        poor["previous_overlap"] = torch.zeros(4)
        try:
            perfect_reward = function(perfect)["reward"].mean()
            poor_reward = function(poor)["reward"].mean()
            report.check(
                "goal_preferred",
                bool(perfect_reward > poor_reward),
                "A successful aligned state must score above a displaced, misaligned state",
            )
        except GENERATED_CODE_ERRORS as error:
            report.check("goal_preferred", False, f"Goal-preference check failed: {error}")
    return report


def _inside_failure_region(
    samples: dict[str, torch.Tensor], region: dict[str, float]
) -> torch.Tensor:
    theta_deg = torch.rad2deg(samples["theta_rad"]) % 360.0
    theta_min = region["theta_deg_min"] % 360.0
    span = region["theta_deg_max"] - region["theta_deg_min"]
    if abs(span) >= 360.0:
        theta_inside = torch.ones_like(theta_deg, dtype=torch.bool)
    elif span >= 0 and region["theta_deg_max"] <= 360.0:
        theta_inside = (theta_deg >= theta_min) & (theta_deg <= region["theta_deg_max"])
    else:
        theta_max = region["theta_deg_max"] % 360.0
        theta_inside = (theta_deg >= theta_min) | (theta_deg <= theta_max)
    return (
        (samples["x_rel"] >= region["x_rel_min"])
        & (samples["x_rel"] <= region["x_rel_max"])
        & (samples["y_rel"] >= region["y_rel_min"])
        & (samples["y_rel"] <= region["y_rel_max"])
        & theta_inside
    )


def validate_sampler_code(source: str, region: dict[str, float]) -> ValidationReport:
    report = ValidationReport()
    _, errors = inspect_source(source, "sample_initial_poses")
    report.check("static_policy", not errors, "; ".join(errors))
    if errors:
        return report
    try:
        function = load_generated_function(source, "sample_initial_poses")
        generator = torch.Generator(device="cpu").manual_seed(20260827)
        samples = function(4096, torch.device("cpu"), generator)
    except GENERATED_CODE_ERRORS as error:
        report.check(
            "executes", False, f"Sampler execution failed: {type(error).__name__}: {error}"
        )
        return report
    report.check("returns_dict", isinstance(samples, dict), "Sampler must return a dictionary")
    if not isinstance(samples, dict):
        return report
    required = {"x_rel", "y_rel", "theta_rad"}
    report.check("required_fields", set(samples) == required, f"Sampler fields must be {required}")
    shapes_ok = all(
        torch.is_tensor(samples.get(key)) and samples[key].shape == (4096,) for key in required
    )
    report.check("batched_fields", shapes_ok, "Every sampler field must be a [num_envs] tensor")
    if not shapes_ok:
        return report
    in_domain = bool(
        (samples["x_rel"] >= -0.10).all()
        and (samples["x_rel"] <= 0.10).all()
        and (samples["y_rel"] >= -0.10).all()
        and (samples["y_rel"] <= 0.20).all()
        and (samples["theta_rad"] >= 0.0).all()
        and (samples["theta_rad"] < 2.0 * torch.pi).all()
    )
    report.check(
        "nominal_domain", in_domain, "Sampler emitted states outside PushT's nominal domain"
    )
    failure_mass = float(_inside_failure_region(samples, region).float().mean())
    report.check(
        "failure_biased",
        failure_mass >= 0.60,
        f"Only {failure_mass:.3f} of samples lie in the T-II failure region; require >= 0.60",
    )
    report.check(
        "retains_exploration",
        failure_mass < 0.995,
        "Sampler collapsed entirely into the failure region; retain nominal-domain exploration",
    )
    report.warnings.append(f"Observed failure-region sample mass: {failure_mass:.3f}")
    return report
