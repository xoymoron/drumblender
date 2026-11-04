"""Keep the server DDP default and explicit local overrides compatible."""

from pathlib import Path

import pytest
import yaml

from scripts.train_modal_control import resolve_devices


def test_server_config_uses_two_devices_with_unchanged_effective_batch():
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "cfg/modal_control.yaml").read_text()
    )
    options = resolve_devices(config["trainer"])
    assert options == {"accelerator": "gpu", "devices": 2, "strategy": "ddp"}
    assert (
        config["data"]["batch_size"] * 2 * config["trainer"]["accumulate_grad_batches"]
        == 64
    )


def test_cpu_override_defaults_to_one_process():
    assert resolve_devices({"accelerator": "gpu", "devices": 2}, "cpu") == {
        "accelerator": "cpu",
        "devices": 1,
        "strategy": "auto",
    }


@pytest.mark.parametrize("devices,want", [("1", 1), ("0,1", [0, 1])])
def test_explicit_gpu_devices_override_the_config(devices, want):
    result = resolve_devices({"accelerator": "gpu", "devices": 2}, devices=devices)
    assert result["devices"] == want
    assert result["strategy"] == ("auto" if want == 1 else "ddp")


@pytest.mark.parametrize("devices", ["0", "-1", "0,0", "-1,0"])
def test_invalid_devices_fail_before_starting_workers(devices):
    with pytest.raises(ValueError):
        resolve_devices({}, devices=devices)


def test_cpu_rejects_gpu_id_lists():
    with pytest.raises(ValueError):
        resolve_devices({}, "cpu", "0,1")
