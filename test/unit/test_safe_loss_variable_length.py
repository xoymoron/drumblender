import auraloss
import pytest
import torch
import torch.nn.functional as F

from drumblender.loss import LengthAwareMRSTFTLoss, SafeScaleInvariantMSSLoss


def _objective(branch):
    class ZeroSpectral(torch.nn.Module):
        def forward(self, pred, target):
            return (pred - target).sum() * 0

    def spectral():
        if branch in ("envelope", "prior"):
            return ZeroSpectral()
        return auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[32], hop_sizes=[8], win_lengths=[32],
        )

    return SafeScaleInvariantMSSLoss(
        sample_rate=1000, base_mss=spectral(), si_mss=spectral(),
        si_enabled=branch == "si", amp_weight=0.0,
        env_weight=1.0 if branch == "envelope" else 0.0,
        envelope_windows_ms=(20.0, 40.0),
        prior_enabled=branch == "prior", prior_n_fft=32,
        prior_hop_length=8, prior_num_bands=4,
        prior_weight_inc=0.3, prior_weight_curv=0.2,
        prior_weight_floor=0.1, prior_weight_onset_band=0.4,
        prior_decay_start_ms=0, prior_min_decay_frames=0,
        prior_short_ms_start=5, prior_short_ms_end=20,
    )


@pytest.mark.parametrize("branch", ["base", "si", "envelope", "prior"])
def test_safe_loss_is_mean_of_valid_clips_regardless_of_storage_padding(branch):
    objective = _objective(branch)
    generator = torch.Generator().manual_seed(20261106)
    target = torch.randn(2, 1, 192, generator=generator) * 0.03
    pred = target * 0.6 + torch.randn(2, 1, 192, generator=generator) * 0.01
    lengths = torch.tensor([96, 192])
    individual = torch.stack([
        objective(pred[i:i + 1, :, :n], target[i:i + 1, :, :n], lengths=[n])
        for i, n in enumerate(lengths.tolist())
    ]).mean()
    assert individual > 0
    padded_pred = F.pad(pred, (0, 96), value=30.0).requires_grad_()
    padded_target = F.pad(target, (0, 96), value=-20.0)
    padded_pred.data[0, :, 96:] = 40.0
    padded_target[0, :, 96:] = -50.0
    actual = objective(padded_pred, padded_target, lengths=lengths)
    torch.testing.assert_close(actual, individual)
    actual.backward()
    assert torch.isfinite(padded_pred.grad).all()
    assert not padded_pred.grad[0, :, 96:].any()
    assert not padded_pred.grad[1, :, 192:].any()


@pytest.mark.parametrize("length", [1, 8, 16])
def test_safe_loss_short_clips_keep_true_duration_and_finite_gradients(length):
    objective = SafeScaleInvariantMSSLoss(
        prior_enabled=True, prior_weight_onset_band=0.1,
    )
    pred = torch.full((1, 1, length), 0.05, requires_grad=True)
    target = torch.full_like(pred, 0.1)
    actual = objective(pred, target, lengths=[length])
    padded = objective(F.pad(pred, (0, 1500)), F.pad(target, (0, 1500)),
                       lengths=[length])
    torch.testing.assert_close(actual, padded)
    assert torch.isfinite(actual)
    actual.backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("objective", [
    LengthAwareMRSTFTLoss(mrstft=torch.nn.L1Loss()),
    SafeScaleInvariantMSSLoss(base_mss=torch.nn.L1Loss(), si_mss=torch.nn.L1Loss()),
])
@pytest.mark.parametrize("lengths", [[0, 8], [-1, 8], [9, 8], [4.5, 8.0], [4], [[4, 8]]])
def test_losses_reject_invalid_length_metadata(objective, lengths):
    audio = torch.ones(2, 1, 8)
    with pytest.raises((TypeError, ValueError)):
        objective(audio, audio, lengths=lengths)
