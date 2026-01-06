from enum import Enum


import torch
from torch.nn import functional as F


AdapterMode = Enum("AdapterMode", ["ENCODER", "DECODER"])


class Adapter(torch.nn.Module):
    def __init__(
        self,
        U: torch.Tensor,
        mean: torch.Tensor,
        device: str,
        mode: AdapterMode,
    ) -> None:
        super().__init__()

        d, k = U.shape

        assert mean.shape[0] == d

        self.mat_encoder = U.T.unsqueeze(2).unsqueeze(3).to(device)
        self.mat_decoder = U.unsqueeze(2).unsqueeze(3).to(device)

        if k == 0:
            # when k=0, conv2d with empty weight causes error
            # so we use zero matrix instead
            self.mat_encoder = torch.zeros((d, d, 1, 1), device=device).float()
            self.mat_decoder = torch.zeros((d, d, 1, 1), device=device).float()

        self.mean = mean.reshape((1, -1, 1, 1)).to(device)

        self.mode = mode

    def forward(self, x) -> torch.Tensor:
        if self.mode == AdapterMode.ENCODER:
            return self.encode(x)
        elif self.mode == AdapterMode.DECODER:
            return self.decode(x)
        else:
            raise ValueError(f"[mode={self.mode}] doesn't exist!")

    def encode(self, x):
        x = x - self.mean
        x = F.conv2d(x, self.mat_encoder)
        return x

    def decode(self, x):
        x = F.conv2d(x, self.mat_decoder)
        x = x + self.mean
        return x
