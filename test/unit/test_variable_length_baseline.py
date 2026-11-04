"""Regression tests for full-clip variable-length baseline training."""

from pathlib import Path
from types import SimpleNamespace

import auraloss
import jsonargparse
import pytest
import torch
import torch.nn.functional as F
import yaml

from drumblender.callbacks import LogAudioCallback
from drumblender.data.collate import pad_audio_params_collate
from drumblender.loss import MRSTFTWithLogRMSAuxLoss, MRSTFTWithSmoothL1AuxLoss
from drumblender.models.encoders import ModalAmpParameters
from drumblender.models.soundstream import (
    SoundStreamAttentionEncoder,
    SoundStreamEncoder,
)
from drumblender.synths import ModalSynth
from drumblender.synths.transient import TransientTCN
from drumblender.tasks import DrumBlender


class _EnvelopeSynth(torch.nn.Module):
    def forward(self, features, length):
        return F.interpolate(features.mean(-1)[:, None], size=length,
                             mode="linear", align_corners=False)[:, 0]


@pytest.mark.parametrize("branch", ["noise", "transient"])
def test_real_audio_encoder_reconstruction_is_independent_of_collate_padding(branch):
    torch.manual_seed(20261106)
    kwargs = {}
    if branch == "noise":
        kwargs.update(noise_autoencoder=SoundStreamEncoder(1, 2, 4, strides=(2, 2),
                                                          transpose_output=True),
                      noise_autoencoder_accepts_audio=True, noise_synth=_EnvelopeSynth())
    else:
        kwargs.update(transient_autoencoder=SoundStreamAttentionEncoder(
                          1, 2, 4, strides=(2, 2)),
                      transient_autoencoder_accepts_audio=True,
                      transient_synth=TransientTCN(hidden_channels=4, num_layers=2,
                          kernel_size=3, film_conditioning=True, film_embedding_size=4),
                      transient_parallel=True)
    model = DrumBlender(modal_synth=ModalSynth(), loss_fn=torch.nn.L1Loss(), **kwargs).eval()
    a = (torch.randn(1, 513), _modal_item(513, 9)[1], 513)
    b = (torch.randn(1, 1025), _modal_item(1025, 17)[1], 1025)
    batch = pad_audio_params_collate([a, b])
    actual = model(batch[0], batch[1], lengths=batch[2], frame_lengths=batch[3])
    expected = model(a[0][None], a[1][None])
    torch.testing.assert_close(actual[:1, :, :513], expected)
    assert not actual[0, :, 513:].any()


def test_generic_loss_uses_mean_of_valid_clips_without_padding_dilution():
    model = DrumBlender(modal_synth=ModalSynth(), loss_fn=torch.nn.L1Loss())
    a = (torch.ones(1, 32), torch.zeros(3, 2, 3), 32)
    b = (torch.full((1, 96), 3.0), torch.zeros(3, 2, 7), 96)
    loss, _, _ = model._do_step(pad_audio_params_collate([a, b]))
    torch.testing.assert_close(loss, torch.tensor(2.0))


@pytest.mark.parametrize("sample_lengths", [(1, 64), (41, 99)])
def test_variable_length_training_uses_valid_audio_and_finite_gradients(sample_lengths):
    model = DrumBlender(
        modal_synth=ModalSynth(), loss_fn=torch.nn.L1Loss(),
        noise_autoencoder=SoundStreamEncoder(1, 2, 4, strides=(2, 2), transpose_output=True),
        noise_autoencoder_accepts_audio=True, noise_synth=_EnvelopeSynth(),
        transient_autoencoder=SoundStreamAttentionEncoder(1, 2, 4, strides=(2, 2)),
        transient_autoencoder_accepts_audio=True,
        transient_synth=TransientTCN(hidden_channels=4, num_layers=1, kernel_size=3,
                                    film_conditioning=True, film_embedding_size=4),
        transient_parallel=True,
    ).train()
    items = [(torch.randn(1, n), _modal_item(n, max(1, n // 8))[1], n)
             for n in sample_lengths]
    audio, params, lengths, frames = pad_audio_params_collate(items)
    audio.requires_grad_()
    loss, reconstructed, _ = model._do_step((audio, params, lengths, frames))
    loss.backward()
    assert torch.isfinite(loss)
    assert reconstructed.shape == audio.shape
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert torch.isfinite(audio.grad).all()
    assert not audio.grad[0, :, sample_lengths[0]:].any()


@pytest.mark.parametrize("length", [1, 2, 23])
def test_soundstream_short_nonempty_audio_has_finite_forward_and_gradient(length):
    encoder = SoundStreamEncoder(1, 2, 4, strides=(2, 2, 4, 8))
    audio = torch.ones(1, 1, length, requires_grad=True)
    encoded = encoder(audio)
    assert encoded.shape == (1, 4, 1)
    assert torch.isfinite(encoded).all()
    encoded.sum().backward()
    assert torch.isfinite(audio.grad).all()


def _modal_item(samples, frames):
    params = torch.zeros(3, 2, frames)
    params[0] = 0.25
    params[1] = torch.exp(-torch.linspace(0, 8, frames))
    # Silent frames are part of the trajectory, not collate padding.
    params[1, :, -3:] = 0
    return torch.zeros(1, samples), params, samples


def _configured_baseline_loss():
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--loss", type=torch.nn.Module)
    path = Path(__file__).resolve().parents[2] / "cfg" / "loss" / "mss.yaml"
    args = parser.parse_object({"loss": yaml.safe_load(path.read_text())})
    return parser.instantiate_classes(args).loss


def test_collate_preserves_exact_modal_frame_lengths():
    batch = pad_audio_params_collate([_modal_item(1024, 17), _modal_item(4096, 65)])
    assert len(batch) == 4
    torch.testing.assert_close(batch[3], torch.tensor([17, 65]))
    assert batch[1].shape == (2, 3, 2, 65)


@pytest.mark.parametrize("sample_lengths", [(1024, 4096), (2048, 2048)])
@pytest.mark.parametrize("with_transient", [False, True])
def test_batched_modal_reconstruction_matches_single_items(sample_lengths, with_transient):
    items = [_modal_item(sample_lengths[0], 17), _modal_item(sample_lengths[1], 65)]
    transient = (TransientTCN(hidden_channels=4, num_layers=2, kernel_size=3)
                 if with_transient else None)
    model = DrumBlender(modal_synth=ModalSynth(), loss_fn=torch.nn.L1Loss(),
                       modal_autoencoder=ModalAmpParameters(num_modes=2, ones=True),
                       transient_synth=transient, transient_parallel=True).eval()
    _, batched, _ = model._do_step(pad_audio_params_collate(items))
    for idx, (audio, params, length) in enumerate(items):
        expected = model(audio.unsqueeze(0), params.unsqueeze(0))
        torch.testing.assert_close(batched[idx:idx + 1, :, :length], expected)


def test_four_entry_modal_batch_survives_device_transfer_and_audio_callback():
    from lightning_fabric.utilities.apply_func import move_data_to_device

    batch = pad_audio_params_collate([_modal_item(1024, 17), _modal_item(4096, 65)])
    batch = move_data_to_device(batch, torch.device("cpu"))
    model = DrumBlender(modal_synth=ModalSynth(), loss_fn=torch.nn.L1Loss())
    callback = LogAudioCallback(on_train=False, on_val=True, on_test=False)
    trainer = SimpleNamespace(is_global_zero=True)
    callback.setup(trainer, model, "validate")
    callback.on_validation_batch_start(trainer, model, batch, 0)
    _, reconstructed, target = model._do_step(batch)
    callback.on_validation_batch_end(trainer, model, None, batch, 0)
    torch.testing.assert_close(callback.saved_targets["val"][0], target)
    torch.testing.assert_close(callback.saved_reconstructions["val"][0], reconstructed)


def test_test_metrics_receive_valid_lengths_from_four_entry_batch(monkeypatch):
    batch = pad_audio_params_collate([_modal_item(1024, 17), _modal_item(4096, 65)])
    model = DrumBlender(modal_synth=ModalSynth(), loss_fn=torch.nn.L1Loss(),
                       test_metrics=torch.nn.ModuleDict({"mae": torch.nn.L1Loss()}))
    metric_lengths = []

    def score(metrics, pred, target):
        metric_lengths.append(pred.shape[-1])
        return {"mae": metrics["mae"](pred, target)}

    monkeypatch.setattr("drumblender.tasks.drumblender.score_reconstruction", score)
    monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)
    model.test_step(batch, 0)
    assert metric_lengths == [1024, 4096]


def _spectral_pair(samples=4096):
    t = torch.arange(samples, dtype=torch.float32)
    target = (torch.sin(t * 0.13) * torch.exp(-t / samples)).view(1, 1, -1)
    pred = target * 0.7 + 0.08 * torch.cos(t * 0.37).view(1, 1, -1)
    return pred, target


@pytest.mark.parametrize("padding_factor", [1, 4, 19])
def test_default_mrstft_ignores_extra_padding(padding_factor):
    objective = _configured_baseline_loss()
    pred, target = _spectral_pair()
    model = DrumBlender(modal_synth=ModalSynth(), loss_fn=objective)
    # Test the actual loss route selected by the task's configured objective.
    length = pred.shape[-1]
    assert model._loss_accepts_lengths
    expected = objective(pred, target, lengths=[length])
    actual = objective(F.pad(pred, (0, length * padding_factor)),
                       F.pad(target, (0, length * padding_factor)), lengths=[length])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("kind", ["plain", "rms", "smooth"])
def test_spectral_loss_is_mean_of_individually_cropped_pairs(kind):
    mrstft = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[64, 128], hop_sizes=[16, 32], win_lengths=[64, 128])
    if kind == "rms":
        objective = MRSTFTWithLogRMSAuxLoss(amp_weight=0, mrstft=mrstft)
    elif kind == "smooth":
        objective = MRSTFTWithSmoothL1AuxLoss(smooth_l1_weight=0, mrstft=mrstft)
    else:
        from drumblender.loss import LengthAwareMRSTFTLoss
        objective = LengthAwareMRSTFTLoss(mrstft=mrstft)
    pred, target = _spectral_pair(1024)
    pred = torch.cat([F.pad(pred[:, :, :512], (0, 512)), pred * 12])
    target = torch.cat([F.pad(target[:, :, :512], (0, 512)), target * 12])
    pred.requires_grad_()
    expected = (mrstft(pred[:1, :, :512], target[:1, :, :512]) +
                mrstft(pred[1:], target[1:])) / 2
    actual = objective(pred, target, lengths=[512, 1024])
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(pred.grad).all()
    assert torch.count_nonzero(pred.grad[0, :, 512:]) == 0
    # The same item reduction must apply even without explicit lengths.
    uniform_expected = (mrstft(pred[:1], target[:1]) + mrstft(pred[1:], target[1:])) / 2
    torch.testing.assert_close(objective(pred, target), uniform_expected)


@pytest.mark.parametrize("samples", [1, 31, 64])
def test_default_mrstft_supports_clips_shorter_than_reflect_padding(samples):
    objective = _configured_baseline_loss()
    target = torch.ones(1, 1, samples)
    pred = (target * 0.8).requires_grad_()
    actual = objective(pred, target, lengths=[samples])
    padded = objective(F.pad(pred, (0, 4096)), F.pad(target, (0, 4096)),
                       lengths=[samples])
    torch.testing.assert_close(actual, padded)
    actual.backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("samples", [5, 16, 23])
@pytest.mark.parametrize("batch_size, channels", [(1, 1), (3, 1), (2, 2)])
def test_transient_conditioning_preserves_batch_and_time(samples, batch_size, channels):
    torch.manual_seed(20261106)
    model = TransientTCN(in_channels=channels + 2, hidden_channels=4, num_layers=1,
                         kernel_size=3, transient_conditioning=True,
                         transient_conditioning_channels=2,
                         transient_conditioning_length=16)
    x = torch.randn(batch_size, channels, samples)
    condition = F.pad(model.transient_conditioning[..., :samples],
                      (0, max(0, samples - 16))).expand(batch_size, -1, -1)
    expected = model.tcn(torch.cat([x, condition], dim=1))
    actual = model(x)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert model.transient_conditioning.grad is not None
