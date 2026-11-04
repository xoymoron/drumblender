"""
LightningModule for drum synthesis
"""
import inspect
from typing import Callable, Literal, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import WandbLogger

from drumblender.metrics import score_reconstruction
from drumblender.utils.lengths import validate_lengths


class DrumBlender(pl.LightningModule):
    """
    LightningModule for kick synthesis from a modal frequency input

    # TODO: Alot of these are currently optional to help with testing and devlopment,
    # but they should be required in the future

    Args:
        modal_synth (nn.Module): Synthesis model takes modal parameters and generates
            audio
        loss_fn (Union[Callable, nn.Module]): Loss function to use for training
        noise_synth (Optional[nn.Module]): Receives noise parameters and generates
            noise audio signal
        transient_synth (Optional[nn.Module]): Receives audio plus transient parameters
            and generates transient audio signal
        modal_autoencoder (Optional[nn.Module]): Receives main embedding and
            generates modal parameters
        noise_autoencoder (Optional[nn.Module]): Receives main embedding and
            generates noise parameters
        transient_autoencoder (Optional[nn.Module]): Receives main embedding and
            generates transient parameters
        encoder (Optional[nn.Module]): Receives audio and generates main embedding
        float32_matmul_precision(Literal["medium", "high", "highest", None]): Sets
            the precision of float32 matmul operations.
    """

    def __init__(
        self,
        modal_synth: nn.Module,
        loss_fn: Union[Callable, nn.Module],
        noise_synth: Optional[nn.Module] = None,
        transient_synth: Optional[nn.Module] = None,
        modal_autoencoder: Optional[nn.Module] = None,
        noise_autoencoder: Optional[nn.Module] = None,
        transient_autoencoder: Optional[nn.Module] = None,
        encoder: Optional[nn.Module] = None,
        transient_parallel: bool = False,
        transient_takes_noise: bool = False,
        modal_autoencoder_accepts_audio: bool = False,
        noise_autoencoder_accepts_audio: bool = False,
        transient_autoencoder_accepts_audio: bool = False,
        test_metrics: Optional[torch.nn.ModuleDict] = None,
        float32_matmul_precision: Literal["medium", "high", "highest", None] = None,
    ):
        super().__init__()

        self.modal_synth = modal_synth
        self.loss_fn = loss_fn
        self.noise_synth = noise_synth
        self.transient_synth = transient_synth
        self.modal_autoencoder = modal_autoencoder
        self.noise_autoencoder = noise_autoencoder
        self.transient_autoencoder = transient_autoencoder
        self.encoder = encoder
        self.transient_parallel = transient_parallel
        self.modal_autoencoder_accepts_audio = modal_autoencoder_accepts_audio
        self.noise_autoencoder_accepts_audio = noise_autoencoder_accepts_audio
        self.transient_autoencoder_accepts_audio = transient_autoencoder_accepts_audio
        self.transient_takes_noise = transient_takes_noise

        if float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(float32_matmul_precision)

        if test_metrics is not None:
            self.metrics = test_metrics

        # ### HIGHLIGHT: Rotate the validation step-loss logging batch index so
        # we do not repeatedly log only the first validation sample.
        self._val_step_log_counter = 0
        self._val_step_log_batch_idx = 0

        # Track whether the configured loss supports an optional `lengths` kwarg.
        self._loss_accepts_lengths = self._callable_accepts_kwarg(
            self.loss_fn, "lengths"
        )

    def forward(self, original: torch.Tensor, params: torch.Tensor,
                lengths=None, frame_lengths=None):
        """Render valid sample/frame extents; restore padding only at the output.

        Equal extents share a forward pass. Different extents never enter an
        encoder or a FiLM normalization layer as collate-padded observations.
        """
        if original.ndim != 3 or original.shape[1] != 1 or min(original.shape) < 1:
            raise ValueError("Expected nonempty mono audio [B,1,T]")
        if params.ndim != 4 or params.shape[0] != original.shape[0] or min(params.shape) < 1:
            raise ValueError("Expected nonempty modal parameters [B,P,M,F]")
        batch_size, _, total_length = original.shape
        lengths = validate_lengths(lengths, batch_size=batch_size,
                                   max_length=total_length, device=original.device)
        frame_lengths = validate_lengths(frame_lengths, batch_size=batch_size,
                                         max_length=params.shape[-1], device=params.device)
        if lengths is None and frame_lengths is None:
            return self._forward_unpadded(original, params)

        samples = lengths.tolist() if lengths is not None else [total_length] * batch_size
        frames = (frame_lengths.tolist() if frame_lengths is not None
                  else [params.shape[-1]] * batch_size)
        groups = {}
        for index, extent in enumerate(zip(samples, frames)):
            groups.setdefault(extent, []).append(index)
        results = [None] * batch_size
        for (sample_count, frame_count), indices in groups.items():
            valid_lengths = (lengths[indices] if lengths is not None else None)
            rendered = self._forward_unpadded(
                original[indices, :, :sample_count],
                params[indices, ..., :frame_count], lengths=valid_lengths,
            )
            if rendered.ndim == 2:
                rendered = rendered[:, None]
            if rendered.shape != (len(indices), 1, sample_count):
                raise ValueError("Synthesizer output must match the requested valid audio shape")
            rendered = torch.nn.functional.pad(rendered, (0, total_length - sample_count))
            for local_index, original_index in enumerate(indices):
                results[original_index] = rendered[local_index:local_index + 1]
        return torch.cat(results, dim=0)

    def _forward_unpadded(self, original, params, lengths=None):
        """Encode and synthesize a group with identical valid time axes."""
        # Main embeddings
        embedding = None
        if self.encoder is not None:
            embedding = self._call_module_with_optional_lengths(
                self.encoder,
                original,
                lengths=lengths,
            )

        # Modal parameter autoencoder
        modal_params = params
        if self.modal_autoencoder is not None:
            if self.modal_autoencoder_accepts_audio:
                modal_params = self._call_module_with_optional_lengths(
                    self.modal_autoencoder,
                    original,
                    params,
                    lengths=lengths,
                )
            else:
                modal_params, _ = self.modal_autoencoder(embedding, params)

        # Noise parameters
        noise_params = None
        if self.noise_autoencoder is not None:
            if self.noise_autoencoder_accepts_audio:
                noise_params = self._call_module_with_optional_lengths(
                    self.noise_autoencoder,
                    original,
                    lengths=lengths,
                )
            else:
                noise_params, _ = self.noise_autoencoder(embedding)

        # Transient parameters
        transient_params = None
        if self.transient_autoencoder is not None:
            if self.transient_autoencoder_accepts_audio:
                transient_params = self._call_module_with_optional_lengths(
                    self.transient_autoencoder,
                    original,
                    lengths=lengths,
                )
            else:
                transient_params, _ = self.transient_autoencoder(embedding)

        num_samples = original.shape[-1]
        y_hat = self.modal_synth(modal_params, num_samples)
        noise = None
        if self.noise_synth is not None:
            noise = self.noise_synth(noise_params, num_samples)[:, None]
            if self.transient_takes_noise:
                y_hat = y_hat + noise
        if self.transient_synth is not None:
            transient = self.transient_synth(y_hat, transient_params)
            y_hat = y_hat + transient if self.transient_parallel else transient
        if noise is not None and not self.transient_takes_noise:
            y_hat = y_hat + noise
        return y_hat

    @staticmethod
    def _callable_accepts_kwarg(fn: Callable, kwarg_name: str) -> bool:
        target = fn.forward if isinstance(fn, nn.Module) else fn
        try:
            sig = inspect.signature(target)
        except (TypeError, ValueError):
            return False

        if kwarg_name in sig.parameters:
            return True

        return any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )

    def _call_module_with_optional_lengths(self, module, *args, lengths=None):
        if lengths is not None and self._callable_accepts_kwarg(module, "lengths"):
            return module(*args, lengths=lengths)
        return module(*args)

    @staticmethod
    def _make_time_mask(lengths: torch.Tensor, T: int, device):
        t = torch.arange(T, device=device)[None, :]
        mask = (t < lengths[:, None]).to(torch.float32)
        return mask[:, None, :]

    def _do_step(self, batch):
        frame_lengths = None
        if len(batch) == 2:
            original, params = batch
            lengths = None
        elif len(batch) == 3:
            original, params, lengths = batch
        elif len(batch) == 4:
            original, params, lengths, frame_lengths = batch
        else:
            raise ValueError("Expected batch to contain 2, 3, or 4 entries")

        lengths = validate_lengths(lengths, batch_size=original.shape[0],
                                   max_length=original.shape[-1], device=original.device)
        y_hat = self(original, params, lengths=lengths, frame_lengths=frame_lengths)

        original_masked = original
        y_hat_masked = y_hat
        if lengths is not None:
            T = original.shape[-1]
            mask = self._make_time_mask(lengths, T, original.device)
            original_masked = original * mask
            y_hat_masked = y_hat * mask

        if self._loss_accepts_lengths:
            # Let length-aware losses apply masking internally to avoid duplicating
            # padded-tail masking work before expensive spectral losses.
            loss = self.loss_fn(y_hat, original, lengths=lengths)
        elif lengths is not None:
            loss = torch.stack([
                self.loss_fn(y_hat[i:i + 1, :, :length], original[i:i + 1, :, :length])
                for i, length in enumerate(lengths.tolist())
            ]).mean()
        else:
            loss = self.loss_fn(y_hat, original)
        # ### HIGHLIGHT: Return the masked target for length-safe metric computation.
        return loss, y_hat_masked, original_masked

    def training_step(self, batch, batch_idx: int):
        loss, y_hat, target = self._do_step(batch)
        # ### HIGHLIGHT: Log a dedicated per-step training loss for denser WandB curves.
        self.log(
            "train/loss_step",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        # ### HIGHLIGHT: Keep the epoch-aggregated loss for long-term trend monitoring.
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        # ### HIGHLIGHT: Return detached audio snapshots for callback-side train audio logging.
        out = {"loss": loss}
        if batch_idx == 0:
            out["train_audio_target"] = target.detach().cpu()
            out["train_audio_reconstruction"] = y_hat.detach().cpu()
        return out

    # ### HIGHLIGHT: Choose a rotating validation batch index for step-loss logging.
    def on_validation_start(self) -> None:
        num_val_batches = getattr(self.trainer, "num_val_batches", 0)
        if isinstance(num_val_batches, (list, tuple)):
            n_batches = int(num_val_batches[0]) if len(num_val_batches) > 0 else 0
        else:
            n_batches = int(num_val_batches or 0)

        if n_batches <= 0:
            self._val_step_log_batch_idx = 0
            return

        self._val_step_log_batch_idx = int(self._val_step_log_counter % n_batches)
        self._val_step_log_counter += 1

    def validation_step(self, batch, batch_idx: int):
        loss, _, _ = self._do_step(batch)
        # ### HIGHLIGHT: Use one canonical validation epoch metric for logging,
        # LR scheduling, early stopping, and checkpoint selection.
        self.log(
            "validation/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        # ### HIGHLIGHT: Align validation step-loss x-axis with training global_step in WandB
        # and rotate which validation batch is used for this step-level snapshot.
        if batch_idx == self._val_step_log_batch_idx:
            if isinstance(self.logger, WandbLogger):
                if self.trainer.is_global_zero:
                    # ### HIGHLIGHT: Keep W&B step monotonic when mixing manual and Lightning logs.
                    # W&B may already have advanced its internal step by 1 from other logs.
                    step = int(self.global_step)
                    run = self.logger.experiment
                    try:
                        run_step = int(getattr(run, "step", -1))
                        if step <= run_step:
                            step = run_step + 1
                    except Exception:
                        pass
                    payload = {
                        "validation/loss_step": float(loss.detach().cpu()),
                        "validation/loss_step_batch_idx": int(batch_idx),
                    }
                    self.logger.experiment.log(payload, step=step)
            else:
                self.log(
                    "validation/loss_step",
                    loss,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    logger=True,
                )

        return loss

    def test_step(self, batch, batch_idx: int):
        loss, y_hat, target = self._do_step(batch)
        self.log("test/loss", loss, batch_size=target.shape[0])
        if hasattr(self, "metrics"):
            lengths = batch[2] if len(batch) >= 3 else [target.shape[-1]] * target.shape[0]
            scores = []
            for i, length in enumerate(lengths):
                n = int(length)
                scores.append(score_reconstruction(
                    self.metrics, y_hat[i:i+1, :, :n], target[i:i+1, :, :n]))
            for key in scores[0]:
                value = sum(row[key] for row in scores) / len(scores)
                self.log(key, value, batch_size=len(scores), on_step=False,
                         on_epoch=True, sync_dist=True)
        return loss
