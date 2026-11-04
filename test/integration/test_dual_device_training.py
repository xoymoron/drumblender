"""Exercise real DDP gradient synchronization using two CPU worker processes."""

import pytest
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from drumblender.data.modal_control import collate_modal_control
from drumblender.tasks.modal_control import ModalControlTask


class SyntheticPercussion(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        length = 513 + index * 8
        clock = torch.arange(length).float()
        audio = (clock * 0.1).sin()[None] * 0.1
        tracks = torch.zeros(3, 2, 5)
        tracks[0] = 0.1
        tracks[1] = torch.linspace(0.1, 0.01, 5)
        return audio, tracks, length


class CheckSynchronizedWeights(pl.Callback):
    def on_fit_start(self, trainer, pl_module):
        torch.set_num_threads(1)
        self.initial = torch.cat(
            [p.detach().flatten() for p in pl_module.parameters() if p.requires_grad]
        ).clone()

    def on_train_end(self, trainer, pl_module):
        assert trainer.world_size == 2
        parameters = torch.cat(
            [p.detach().flatten() for p in pl_module.parameters() if p.requires_grad]
        )
        assert torch.isfinite(parameters).all()
        assert not torch.equal(parameters, self.initial)
        copies = [torch.empty_like(parameters) for _ in range(trainer.world_size)]
        dist.all_gather(copies, parameters)
        torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=0)


@pytest.mark.parametrize("stage", ["modal", "noise", "joint", "patterns"])
def test_two_process_training_synchronizes_model_weights(stage, tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    pl.seed_everything(20261106, workers=True)
    model = ModalControlTask(
        stage=stage, channels=16, context_tokens=2, features=32, top_k=2
    )
    loader = DataLoader(
        SyntheticPercussion(), batch_size=2, collate_fn=collate_modal_control
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=2,
        strategy="ddp_spawn",
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
        callbacks=[CheckSynchronizedWeights()],
    )
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
