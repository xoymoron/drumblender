from typing import Optional

import torch

from drumblender.models.tcn import TCN


class TransientTCN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        out_channels: int = 1,
        dilation_base: int = 2,
        dilation_blocks: Optional[int] = None,
        num_layers: int = 8,
        kernel_size: int = 13,
        film_conditioning: bool = False,
        film_embedding_size: Optional[int] = None,
        film_batch_norm: bool = True,
        transient_conditioning: bool = False,
        transient_conditioning_channels: int = 32,
        transient_conditioning_length: int = 24000,
    ):
        super().__init__()
        self.tcn = TCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dilation_base=dilation_base,
            dilation_blocks=dilation_blocks,
            num_layers=num_layers,
            kernel_size=kernel_size,
            film_conditioning=film_conditioning,
            film_embedding_size=film_embedding_size,
            film_batch_norm=film_batch_norm,
        )

        if transient_conditioning:
            p = (
                torch.randn(
                    1, transient_conditioning_channels, transient_conditioning_length
                )
                / transient_conditioning_channels
            )
            self.transient_conditioning = torch.nn.Parameter(p, requires_grad=True)

    def forward(self, x: torch.Tensor, embedding: Optional[torch.Tensor] = None):
        if hasattr(self, "transient_conditioning"):
            cond = self.transient_conditioning[..., :x.size(-1)]
            cond = torch.nn.functional.pad(cond, (0, x.size(-1) - cond.size(-1)))
            cond = cond.expand(x.size(0), -1, -1)
            x = torch.cat([x, cond], dim=1)

        x = self.tcn(x, embedding)
        return x
