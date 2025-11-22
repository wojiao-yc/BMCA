import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 16,
        bias: bool = False,
        freeze_main: bool = True,
    ):
        super().__init__()
        self.main = nn.Linear(in_features, out_features, bias=bias)
        if freeze_main:
            for p in self.main.parameters():
                p.requires_grad = False
        self.r = r
        if r > 0:
            self.A = nn.Linear(in_features, r, bias=False)
            self.B = nn.Linear(r, out_features, bias=False)
            nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.B.weight)
            self.scaling = alpha / r
        else:
            self.A = None
            self.B = None
            self.scaling = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r > 0:
            return self.main(x) + self.B(self.A(x)) * self.scaling
        else:
            return self.main(x)