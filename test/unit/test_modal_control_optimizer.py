import pytest
import torch

from drumblender.tasks.modal_control import ModalControlTask


@pytest.mark.parametrize("stage", ["modal", "noise", "joint", "patterns"])
def test_staged_optimizer_schedules_all_trainable_parameters(stage):
    model = ModalControlTask(
        stage=stage, channels=16, context_tokens=2, features=32, top_k=2,
    )
    configured = model.configure_optimizers()
    assert isinstance(configured, dict)
    optimizer = configured["optimizer"]
    schedule = configured["lr_scheduler"]
    assert type(optimizer) is torch.optim.Adam
    assert optimizer.defaults["weight_decay"] == 0
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    assert {id(p) for p in optimized} == expected
    assert len(optimized) == len(expected)
    assert {group["lr"] for group in optimizer.param_groups} == {1e-4}
    assert schedule["monitor"] == "validation/loss"
    scheduler = schedule["scheduler"]
    assert scheduler.optimizer is optimizer
    initial_rates = [group["lr"] for group in optimizer.param_groups]
    for _ in range(12):
        scheduler.step(1.0)
    assert [group["lr"] for group in optimizer.param_groups] == [
        rate / 2 for rate in initial_rates
    ]
