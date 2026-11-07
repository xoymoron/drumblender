import copy

import numpy as np
import pytest
import torch

from drumblender.control import ControlSAE
from drumblender.control_protocol import (
    WARMUP_KIND,
    fork_warmup,
    paired_budget,
    resumed_seconds,
    select_cohort,
    validate_atoms,
)
from drumblender.intervention import InterventionSAE


def warmup():
    return dict(
        kind=WARMUP_KIND,
        identity={"checkpoint": "new"},
        cache_sha256="cache",
        state_dict=ControlSAE().state_dict(),
        warmup_seconds=1800.0,
        step=10,
        seed=7,
    )


def test_fresh_fork_is_identical_independent_and_has_no_retired_atom():
    artifact = warmup()
    main, comparison = fork_warmup(artifact, {"checkpoint": "new"}, "cache", "cpu")
    for a, b in zip(main.parameters(), comparison.parameters()):
        assert torch.equal(a, b) and a.data_ptr() != b.data_ptr()
    with torch.no_grad():
        main.encoder.bias[22] = 1e6
    mask = torch.ones(1, 12, dtype=torch.bool)
    _, codes = main.reconstruct(torch.zeros(1, 12, 256), mask)
    assert codes[..., 22].gt(0).all()
    assert not torch.equal(main.encoder.bias, comparison.encoder.bias)
    assert isinstance(main, InterventionSAE)


@pytest.mark.parametrize(
    "change", ["kind", "identity", "cache_sha256", "reader_state_dict"]
)
def test_fork_rejects_old_curriculum_or_wrong_checkpoint(change):
    artifact = warmup()
    artifact[change] = "old"
    with pytest.raises(ValueError):
        fork_warmup(artifact, {"checkpoint": "new"}, "cache", "cpu")


def test_training_budget_includes_complete_warmup():
    assert paired_budget(1800, 120, 180) == (5400, 9000)
    with pytest.raises(ValueError):
        paired_budget(7200, 120, 180)
    with pytest.raises(ValueError):
        paired_budget(float("nan"), 120, 180)


def test_resume_budget_survives_reboot_and_charges_interruption_time():
    clock = {"utc_started": 1000, "started_monotonic": 90000}
    assert resumed_seconds(clock, 100, 1600) == 600
    assert resumed_seconds(clock, 700, 1600) == 700
    with pytest.raises(ValueError):
        resumed_seconds(clock, 100, 999)


def test_train_support_selection_has_no_historical_id_rule():
    activity = np.zeros((20, 512))
    activity[:, [22, 69, 5, 17]] = 1
    rows = [{"pack": str(i % 2)} for i in range(20)]
    assert select_cohort(activity, rows) == [5, 17, 22, 69]
    assert validate_atoms([22, 17], [5, 17, 22, 69]) == [22, 17]
    for invalid in ([22, 22], [-1, 17], [512, 17], [1, 17]):
        with pytest.raises(ValueError):
            validate_atoms(invalid, [5, 17, 22, 69])


def test_warmup_is_not_mutated_by_fork():
    artifact = warmup()
    before = copy.deepcopy(artifact["state_dict"])
    main, _ = fork_warmup(artifact, {"checkpoint": "new"}, "cache", "cpu")
    with torch.no_grad():
        main.encoder.bias.add_(1)
    assert all(torch.equal(before[k], artifact["state_dict"][k]) for k in before)
