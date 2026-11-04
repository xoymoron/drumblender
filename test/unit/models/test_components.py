import torch

from drumblender.models.components import FiLM, GatedActivation, TFiLM


def test_film_supports_single_valid_sample_in_training():
    film = FiLM(4, 3).train()
    x = torch.ones(1, 3, 1, requires_grad=True)
    embedding = torch.ones(1, 4, requires_grad=True)
    output = film(x, embedding)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(embedding.grad).all()


def test_singleton_film_gradient_uses_the_statistics_from_its_forward():
    film = FiLM(3, 2).train()
    with torch.no_grad():
        film.net.weight.zero_()
        film.net.bias.copy_(torch.tensor([2.0, 3.0, 0.0, 0.0]))
        film.norm.running_mean.copy_(torch.tensor([1.0, 2.0]))
        film.norm.running_var.copy_(torch.tensor([4.0, 9.0]))
    audio = torch.ones(1, 2, 1, requires_grad=True)
    singleton = film(audio, torch.ones(1, 3))
    # Another valid-length group updates BN buffers before the shared backward.
    film(torch.arange(20.0).reshape(1, 2, 10) * 10, torch.ones(1, 3))
    singleton.sum().backward()
    expected = torch.tensor([2.0 / (4.0 + 1e-5) ** 0.5,
                             3.0 / (9.0 + 1e-5) ** 0.5]).reshape(1, 2, 1)
    torch.testing.assert_close(audio.grad, expected)


def test_film_correctly_forwards_input():
    batch_size = 11
    in_channels = 13
    seq_len = 31
    film_embedding_size = 7

    film = FiLM(film_embedding_size, in_channels)
    x = torch.testing.make_tensor(
        batch_size, in_channels, seq_len, device="cpu", dtype=torch.float32
    )
    film_embedding = torch.testing.make_tensor(
        batch_size,
        film_embedding_size,
        device="cpu",
        dtype=torch.float32,
        requires_grad=True,
    )

    y = film(x, film_embedding)
    assert y.shape == (batch_size, in_channels, seq_len)

    (dy_dx,) = torch.autograd.grad(y.sum().square(), film_embedding)
    assert (dy_dx.abs() > 0.0).all()


def test_film_can_toggle_batch_norm(mocker):
    spy_batch_norm_init = mocker.spy(torch.nn.BatchNorm1d, "__init__")
    spy_batch_norm_forward = mocker.spy(torch.nn.BatchNorm1d, "forward")

    batch_size = 7
    in_channels = 13
    seq_len = 37
    film_embedding_size = 5

    x = torch.testing.make_tensor(
        batch_size, in_channels, seq_len, device="cpu", dtype=torch.float32
    )
    film_embedding = torch.testing.make_tensor(
        batch_size, film_embedding_size, device="cpu", dtype=torch.float32
    )

    # Test with batch norm
    film = FiLM(film_embedding_size, in_channels, use_batch_norm=True)
    film(x, film_embedding)
    assert spy_batch_norm_init.call_count == 1
    assert spy_batch_norm_forward.call_count == 1

    # Test without batch norm
    film = FiLM(film_embedding_size, in_channels, use_batch_norm=False)
    film(x, film_embedding)
    assert spy_batch_norm_init.call_count == 1
    assert spy_batch_norm_forward.call_count == 1


def test_gated_activation_correctly_forwards_input():
    batch_size = 11
    out_channels = 17
    seq_len = 23

    ga = GatedActivation()
    x = torch.testing.make_tensor(
        batch_size, out_channels * 2, seq_len, device="cpu", dtype=torch.float32
    )

    y = ga(x)
    assert y.shape == (batch_size, out_channels, seq_len)


def test_gated_activation_gates_input():
    batch_size = 11
    out_channels = 17
    seq_len = 23

    ga = GatedActivation()
    x_1 = torch.testing.make_tensor(
        batch_size, out_channels, seq_len, device="cpu", dtype=torch.float32
    )
    x_2 = torch.testing.make_tensor(
        batch_size,
        out_channels,
        seq_len,
        device="cpu",
        dtype=torch.float32,
        low=-1e8,
        high=-1e8,
    )
    x = torch.cat([x_1, x_2], dim=1)

    y = ga(x)
    assert y.shape == (batch_size, out_channels, seq_len)

    assert y.abs().sum() == 0.0


def test_tfilm_correctly_forwards_input():
    batch_size = 3
    channels = 11
    block_size = 16
    seq_len = block_size * 10

    tfilm = TFiLM(
        channels=channels,
        block_size=block_size,
    )

    x = torch.testing.make_tensor(
        batch_size, channels, seq_len, device="cpu", dtype=torch.float32
    )

    y = tfilm(x)
    assert y.shape == (batch_size, channels, seq_len)
