import pytest

from scripts.replay_trajectory_aligned import (
    NEW_RESTORE,
    OLD_RESTORE,
    align_parallel_replay_source,
)


def test_align_parallel_replay_source_corrects_state_index():
    source = f"before\n    {OLD_RESTORE}\nafter\n"

    corrected = align_parallel_replay_source(source)

    assert OLD_RESTORE not in corrected
    assert NEW_RESTORE in corrected


@pytest.mark.parametrize("source", ["no restore", f"{OLD_RESTORE}\n{OLD_RESTORE}"])
def test_align_parallel_replay_source_rejects_unknown_implementation(source):
    with pytest.raises(RuntimeError, match="exactly one"):
        align_parallel_replay_source(source)
