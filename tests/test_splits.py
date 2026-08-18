import pandas as pd
import pytest

from ocular import splits


def make_frame(n_groups: int = 20, per_group: int = 50) -> pd.DataFrame:
    rows = []
    for g in range(n_groups):
        group = f"study1/p{g:03d}"
        for i in range(per_group):
            rows.append(
                {
                    "key": f"{group}_{i}",
                    "group": group,
                    "label": "blink" if i % 2 else "non-blink",
                }
            )
    return pd.DataFrame(rows)


def test_splits_never_share_a_recording():
    """The whole point of splitting by group. This is the leakage guard."""
    frame = make_frame()
    split = splits.make(frame)

    assert not set(split.train) & set(split.val)
    assert not set(split.train) & set(split.test)
    assert not set(split.val) & set(split.test)


def test_every_recording_is_used():
    frame = make_frame()
    split = splits.make(frame)
    covered = set(split.train) | set(split.val) | set(split.test)
    assert covered == set(frame["group"])


def test_assigned_frame_has_no_group_crossing_splits():
    frame = make_frame()
    assigned = splits.make(frame).assign(frame)

    per_group = assigned.groupby("group")["split"].nunique()
    assert (per_group == 1).all()


def test_split_is_deterministic():
    frame = make_frame()
    assert splits.make(frame, seed=7) == splits.make(frame, seed=7)


def test_different_seeds_give_different_splits():
    frame = make_frame()
    assert splits.make(frame, seed=1) != splits.make(frame, seed=2)


def test_adding_a_recording_leaves_the_others_alone():
    """Hash ordering means new data does not reshuffle existing assignments."""
    small = make_frame(n_groups=20)
    large = make_frame(n_groups=21)

    before = splits.make(small).assign(small)
    after = splits.make(large).assign(large)

    shared = set(small["group"])
    before_map = before.drop_duplicates("group").set_index("group")["split"]
    after_map = after.drop_duplicates("group").set_index("group")["split"]

    moved = [g for g in shared if before_map[g] != after_map[g]]
    # A handful may shift as the split boundaries move, but not most of them
    assert len(moved) < len(shared) / 2


def test_split_proportions_are_roughly_respected():
    frame = make_frame(n_groups=100)
    split = splits.make(frame, val_fraction=0.2, test_fraction=0.2)

    assert len(split.test) == 20
    assert len(split.val) == 20
    assert len(split.train) == 60


def test_too_few_recordings_is_an_error():
    frame = make_frame(n_groups=2)
    with pytest.raises(ValueError, match="at least 3"):
        splits.make(frame)


def test_fractions_that_leave_no_training_data_are_an_error():
    frame = make_frame(n_groups=4)
    with pytest.raises(ValueError, match="cannot fill"):
        splits.make(frame, val_fraction=0.5, test_fraction=0.5)


def test_assign_drops_unlisted_groups():
    frame = make_frame(n_groups=5)
    split = splits.Split(train=("study1/p000",), val=("study1/p001",), test=("study1/p002",))
    assigned = split.assign(frame)

    assert set(assigned["group"]) == {"study1/p000", "study1/p001", "study1/p002"}
