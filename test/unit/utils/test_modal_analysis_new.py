"""Regression tests for observable partials and synthesis-safe envelopes."""

import numpy as np
import pytest

from drumblender.utils.modal_analysis_new import CQTModalAnalysis


def tone(frequency, seconds=0.5, sample_rate=16000, phase=0.3):
    time = np.arange(round(seconds * sample_rate)) / sample_rate
    return (0.4 * np.sin(2 * np.pi * frequency * time + phase)).astype(np.float32)


def analyzer(**kwargs):
    return CQTModalAnalysis(
        16000,
        hop_length=128,
        fmin=30,
        n_bins=192,
        bins_per_octave=24,
        short_window=1024,
        long_window=4096,
        **kwargs,
    )


def test_silence_has_no_modes_and_finite_output():
    result = analyzer().analyze(np.zeros(8000, dtype=np.float32))
    assert result.frequencies.shape == (0, 63)
    assert np.isfinite(result.amplitudes).all()


def test_two_simultaneous_peaks_remain_two_stationary_tracks():
    result = analyzer(num_modes=2).analyze(tone(220) + tone(1500))
    assert result.frequencies.shape[0] == 2
    medians = np.sort(np.median(result.frequencies, axis=1))
    np.testing.assert_allclose(medians, [220, 1500], atol=2)
    assert np.max(np.ptp(result.frequencies[:, 8:-8], axis=1)) < 5


def test_amplitude_can_recover_after_beating_minimum():
    result = analyzer(num_modes=2, backend="stft").analyze(
        tone(1000, 1) + tone(1020, 1)
    )
    medians = np.sort(np.median(result.frequencies, axis=1))
    np.testing.assert_allclose(medians, [1000, 1020], atol=2)
    assert np.all(result.observed[:, 15:-15].mean(axis=1) > 0.8)


def test_delayed_mode_keeps_time_and_positive_frequency_in_silent_frames():
    wave = np.zeros(16000, dtype=np.float32)
    wave[4800:9600] = tone(1800, 0.3)
    result = analyzer(num_modes=1, backend="stft").analyze(wave)
    active = result.amplitudes[0] > result.amplitudes[0].max() * 0.5
    assert result.frame_times[np.flatnonzero(active)[0]] > 0.25
    assert result.frame_times[np.flatnonzero(active)[-1]] < 0.65
    assert np.all(result.frequencies[0] > 1700)


def test_mode_limit_is_not_hardcoded_to_64():
    model = analyzer(num_modes=160)
    assert model.num_modes == 160
    assert model.max_peaks >= 160


def test_empty_or_nonfinite_audio_is_rejected():
    model = analyzer()
    with pytest.raises(ValueError):
        model.analyze(np.array([], dtype=np.float32))
    with pytest.raises(ValueError):
        model.analyze(np.array([np.nan], dtype=np.float32))


def test_torch_batch_with_unequal_mode_counts_is_padded():
    torch = pytest.importorskip("torch")
    batch = torch.from_numpy(np.stack([tone(440), np.zeros(8000, np.float32)]))
    frequencies, amplitudes, phases = analyzer(num_modes=1)(batch)
    assert frequencies.shape == amplitudes.shape == phases.shape == (2, 1, 63)
    assert torch.count_nonzero(amplitudes[1]) == 0


def test_cached_silent_fmin_padding_cannot_change_synthesized_pitch():
    torch = pytest.importorskip("torch")
    from drumblender.synths.modal import modal_synth

    amplitudes = torch.tensor([[[0.0, 0.4, 0.4, 0.0, 0.0]]])
    reference = torch.full_like(amplitudes, 0.3)
    padded = torch.tensor([[[0.001, 0.3, 0.3, 0.001, 0.001]]])
    np.testing.assert_allclose(
        modal_synth(padded, amplitudes, 1024).numpy(),
        modal_synth(reference, amplitudes, 1024).numpy(),
        atol=2e-6,
    )


def test_internal_zero_amp_frame_does_not_create_an_audible_frequency_sweep():
    torch = pytest.importorskip("torch")
    from drumblender.synths.modal import modal_synth

    sample_rate = 16000
    sample_count = 900
    frequencies = torch.tensor([[[200.0, 20.0, 800.0]]], dtype=torch.float64)
    amplitudes = torch.tensor([[[0.5, 0.0, 0.5]]], dtype=torch.float64)
    actual = modal_synth(
        frequencies * (2 * torch.pi / sample_rate), amplitudes, sample_count
    )

    position = (np.arange(sample_count) + 0.5) * 3 / sample_count - 0.5
    carrier_hz = np.where(position < 1.0, 200.0, 800.0)
    envelope = np.interp(position, [0.0, 1.0, 2.0], [0.5, 0.0, 0.5])
    expected = envelope * np.sin(np.cumsum(carrier_hz * 2 * np.pi / sample_rate))
    np.testing.assert_allclose(actual.numpy()[0], expected, atol=1e-10)


def test_reference_renderer_and_torch_agree_across_a_missing_peak():
    torch = pytest.importorskip("torch")
    from drumblender.synths.modal import modal_synth
    from drumblender.utils.modal_analysis_new import ModalAnalysisResult, render_modal

    sample_rate, sample_count, frames = 16000, 1000, 5
    frequencies = np.array([[200, 200, 200, 800, 800]], np.float32)
    amplitudes = np.array([[0.5, 0.5, 0, 0.5, 0.5]], np.float32)
    phases = np.zeros_like(frequencies)
    frame_times = (
        (np.arange(frames) + 0.5) * sample_count / frames - 0.5
    ) / sample_rate
    result = ModalAnalysisResult(
        frequencies,
        amplitudes,
        phases,
        amplitudes != 0,
        np.ones_like(frequencies),
        np.zeros_like(frequencies, np.int8),
        frame_times,
        np.ones(1),
        1,
    )
    reference = render_modal(result, sample_count, sample_rate)
    torch_audio = modal_synth(
        torch.from_numpy(frequencies[None]) * (2 * torch.pi / sample_rate),
        torch.from_numpy(amplitudes[None]),
        sample_count,
    )[0].numpy()
    np.testing.assert_allclose(reference, torch_audio, atol=4e-5)


def test_missing_observation_does_not_invent_a_frequency_curve():
    model = analyzer(min_length=3, min_streak=2)
    low = np.array([[200, 0.4, 0, 4, 1, 1]], dtype=np.float32)
    high = np.array([[208, 0.4, 0, 4, 1, 1]], dtype=np.float32)
    empty = np.empty((0, 6), np.float32)
    times = np.arange(7) * 128 / 16000
    tracks = model._track_peaks([low] * 3 + [empty] + [high] * 3, times)
    result = model._pack_tracks(tracks, times)
    assert result.frequencies.shape[0] == 1
    assert not result.observed[0, 3]
    assert result.amplitudes[0, 3] == 0
    assert result.frequencies[0, 3] == result.frequencies[0, 2]


def test_placeholder_gaps_never_count_as_observations_or_reconnect_dead_tracks():
    model = analyzer(min_length=3, min_streak=2)
    peak = np.array([[220, 0.4, 0, 4, 1, 1]], dtype=np.float32)
    empty = np.empty((0, 6), np.float32)
    times = np.arange(12) * 128 / 16000
    tracks = model._track_peaks([peak] + [empty] * 11, times)
    result = model._pack_tracks(tracks, times)
    assert result.frequencies.shape[0] == 0
    tracks = model._track_peaks([peak] * 3 + [empty] * 6 + [peak] * 3, times)
    result = model._pack_tracks(tracks, times)
    assert result.frequencies.shape[0] == 2
    np.testing.assert_array_equal(result.observed.sum(axis=1), [3, 3])
    assert np.all(result.amplitudes[:, 3:9] == 0)


def test_more_than_64_resolved_modes_can_survive():
    frequencies = np.linspace(300, 7200, 80)
    time = np.arange(8000) / 16000
    audio = np.sin(2 * np.pi * frequencies[:, None] * time + 0.3).sum(axis=0) / 160
    result = analyzer(backend="stft", num_modes=96).analyze(audio)
    assert 64 < len(result.frequencies) <= 96


def test_synth_silent_slots_and_chunking_preserve_gradients():
    torch = pytest.importorskip("torch")
    from drumblender.synths.modal import modal_synth

    frequency = torch.full((1, 3, 5), 0.3, requires_grad=True)
    amplitude = torch.tensor(
        [[[0, 0.4, 0.4, 0, 0], [0, 0, 0, 0, 0], [0.1, 0.2, 0.3, 0.2, 0.1]]],
        requires_grad=True,
    )
    output = modal_synth(frequency, amplitude, 100, mode_chunk_size=1)
    torch.testing.assert_close(
        output, modal_synth(frequency, amplitude, 100, mode_chunk_size=32)
    )
    output.square().mean().backward()
    assert torch.isfinite(frequency.grad).all()
    assert torch.isfinite(amplitude.grad).all()
    assert frequency.grad[0, 0, 1:3].abs().sum() > 0


def test_reference_renderer_matches_synth_frame_alignment():
    torch = pytest.importorskip("torch")
    from drumblender.synths.modal import modal_synth
    from drumblender.utils.modal_analysis_new import render_modal

    audio = tone(220, seconds=0.503)
    result = analyzer().analyze(audio)
    frequency, amplitude, phase = [
        torch.from_numpy(values[None]) for values in result.parameters()
    ]
    actual = modal_synth(
        frequency * (2 * torch.pi / 16000), amplitude, len(audio), phase
    )[0].numpy()
    expected = render_modal(result, len(audio), 16000)
    np.testing.assert_allclose(actual, expected, atol=6e-5)
    assert np.corrcoef(audio, expected)[0, 1] > 0.97


def test_strong_attack_does_not_remove_quiet_tonal_tail():
    audio = tone(220, seconds=1) * 0.0005
    audio[0] = 0.8
    result = analyzer(num_modes=None).analyze(audio)
    candidates = np.flatnonzero(np.abs(np.median(result.frequencies, axis=1) - 220) < 2)
    assert any(result.observed[index, -30:].mean() > 0.8 for index in candidates)


def test_isolated_short_hf_burst_uses_short_window_envelope():
    sample_rate = 48000
    audio = np.zeros(sample_rate, np.float32)
    audio[9600:10560] = tone(6500, 0.02, sample_rate)
    result = CQTModalAnalysis(sample_rate, num_modes=1).analyze(audio)
    amplitude = result.amplitudes[0]
    assert amplitude.max() > 0.2
    active = result.frame_times[amplitude > amplitude.max() * 0.5]
    assert active[-1] - active[0] < 0.055
    assert np.any(result.source[0] == 3)


def test_mode_just_above_fmin_is_not_lost_at_cqt_boundary():
    result = CQTModalAnalysis(48000, num_modes=1).analyze(tone(20.1, 1, 48000))
    np.testing.assert_allclose(np.median(result.frequencies[0, 50:140]), 20.1, atol=0.5)
    assert result.observed[0, 50:140].mean() > 0.8


def test_new_builder_defaults_do_not_reuse_legacy_cache_directory():
    pytest.importorskip("torchaudio")
    from scripts.build_modal_features import parse_args

    old = parse_args([], default_backend="legacy")
    new = parse_args([], default_backend="hybrid")
    assert old.out_dir != new.out_dir
    assert old.num_modes == 64 and new.num_modes == 128
    assert old.min_length == 10 and new.min_length == 4
    assert new.threshold_db == -90
