"""Check the shared seed at stochastic inference and validation boundaries."""

import pytest
import torch

from drumblender.tasks.modal_control import ModalControlTask, spectral_loss


@pytest.fixture
def noise_reference():
    model = ModalControlTask(
        stage="noise", channels=16, context_tokens=2, features=32, top_k=2
    ).eval()
    model.noise_updates.fill_(1)
    audio = (0.1 * torch.sin(torch.arange(256) * 0.1))[None, None]
    params = torch.zeros(1, 3, 2, 5)
    params[:, 0] = 0.1
    params[:, 1] = 0.2
    reference = model.encode_reference(audio, params)
    return model, audio, params, reference


def test_default_edit_uses_the_shared_excitation_seed(noise_reference):
    model, _, _, reference = noise_reference
    default = model.edit_reference(reference)
    explicit = model.edit_reference(reference, seed=20261106)
    repeated = model.edit_reference(reference)
    torch.testing.assert_close(default["audio"], explicit["audio"], rtol=0, atol=0)
    torch.testing.assert_close(default["audio"], repeated["audio"], rtol=0, atol=0)


def test_validation_uses_the_shared_excitation_seed(noise_reference):
    model, audio, params, reference = noise_reference
    modal = model.modal_synth(reference.parameters, audio.shape[-1])
    noise = model._render_noise(reference.noise_bands, audio.shape[-1], seed=20261106)
    expected = spectral_loss(modal + noise, audio)
    expected += model.hparams.residual_weight * spectral_loss(noise, audio - modal)
    actual, _ = model._example_loss(audio, params, training=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
