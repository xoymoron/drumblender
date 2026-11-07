import numpy as np
import torch
from torch import nn

from drumblender.control import ControlSAE, FrameCache, latent_delta
from drumblender.models.joint_resynthesis import (
    JointResynthesisModel,
    _residual_mlp,
    _sample_frames,
)
from scripts.control_modal import ModalAwareSAE, probe_head


def test_crop_halo_matches_full_clip_and_masks_neighbor(tmp_path):
    rng = np.random.default_rng(21)
    values = rng.standard_normal((281, 256), dtype=np.float32)
    np.save(tmp_path / "train.npy", values)
    cache = FrameCache(
        tmp_path, "train", [{"samples": 279 * 128 + 37}, {"samples": 128}]
    )
    sae = ControlSAE(features=12)
    full = torch.from_numpy(values[:280].copy())[None]
    full_mask = torch.ones((1, 280), dtype=torch.bool)
    full_out, _ = sae.reconstruct(full, full_mask)
    x, weight = cache.batch(np.array([1]))
    out, _ = sae.reconstruct(x, weight > 0)
    torch.testing.assert_close(out[0, 8:32], full_out[0, 256:])
    assert weight[0, 31].item() == 37 / 128
    assert not weight[0, 32:].any()


def test_scaled_edit_matches_difference_of_decoder_outputs():
    sae = ControlSAE(features=12)
    h = torch.randn(1, 41, 256)
    mask = torch.ones(1, 41, dtype=torch.bool)
    _, codes = sae.reconstruct(h, mask)
    sigma = torch.linspace(0.2, 2, 256)
    idx = int(codes.square().sum((0, 1)).argmax())
    edited = codes.clone()
    edited[..., idx] *= 1.75
    expected = (sae.render_codes(edited) - sae.render_codes(codes)) * sigma
    torch.testing.assert_close(0.75 * latent_delta(sae, codes, idx, sigma), expected)


def test_modal_teacher_matches_frozen_head_and_backpropagates_only_to_input():
    class Head(nn.Module):
        _correction = JointResynthesisModel._correction

        def __init__(self):
            super().__init__()
            self.modal_h = nn.Linear(256, 64)
            self.modal_u = nn.Linear(64, 64)
            self.modal_norm = nn.LayerNorm(64)
            self.modal_ffn = _residual_mlp(64)
            self.modal_out = nn.Linear(64, 2)

    head = Head().requires_grad_(False)
    h = torch.randn(1, 11, 256, requires_grad=True)
    pos = torch.tensor([[0.0, 5.25, 10.0]])
    u = torch.randn(1, 3, 64)
    observed = torch.ones((1, 1, 3), dtype=torch.bool)
    expect = head._correction(
        u[:, None], head.modal_h(_sample_frames(h, pos[0])), observed
    )
    pred = probe_head(head, h, pos, u)
    torch.testing.assert_close(
        pred, expect[:, 0] * torch.tensor([0.03, np.log(4)], dtype=h.dtype)
    )
    pred.square().sum().backward()
    assert h.grad.abs().sum() > 0
    assert all(p.grad is None for p in head.parameters())


def test_modal_sae_does_not_reserve_historical_feature_ids():
    sae = ModalAwareSAE()
    with torch.no_grad():
        sae.encoder.bias[22] = 1e6
    hidden = torch.zeros(1, 12, 256)
    _, codes = sae.reconstruct(hidden, torch.ones(1, 12, dtype=torch.bool))
    assert codes[..., 22].gt(0).all()
