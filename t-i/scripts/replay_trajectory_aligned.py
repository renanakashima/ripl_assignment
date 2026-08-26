#!/usr/bin/env python3
"""Run ManiSkill GPU replay with the recorded-state index aligned to actions.

ManiSkill trajectories contain T actions and T + 1 environment states. After
executing action t, the environment must therefore be restored to state t + 1
before action t + 1. Some ManiSkill releases restore state t in the GPU replay
path, which makes subsequent visual observations and actions misaligned.
"""

from __future__ import annotations

import inspect

OLD_RESTORE = "env.base_env.set_state_dict(env_states_batch[t])"
NEW_RESTORE = "env.base_env.set_state_dict(env_states_batch[t + 1])"


def align_parallel_replay_source(source: str) -> str:
    """Return replay source with exactly one recorded-state index corrected."""
    occurrences = source.count(OLD_RESTORE)
    if occurrences != 1:
        raise RuntimeError(
            "Expected exactly one unaligned GPU replay state restore, "
            f"found {occurrences}. Inspect the installed ManiSkill replay implementation."
        )
    return source.replace(OLD_RESTORE, NEW_RESTORE)


def main() -> None:
    from mani_skill.trajectory import replay_trajectory

    source = inspect.getsource(replay_trajectory.replay_parallelized_sim)
    aligned_source = align_parallel_replay_source(source)
    exec(  # noqa: S102 - constrained, verified replacement in trusted installed source
        compile(aligned_source, replay_trajectory.__file__, "exec"),
        replay_trajectory.__dict__,
    )
    print("Applied PushT GPU replay alignment: env_states[t] -> env_states[t + 1]")
    replay_trajectory.main(replay_trajectory.parse_args())


if __name__ == "__main__":
    main()
