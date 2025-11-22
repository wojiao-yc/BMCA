from typing import Optional, Tuple
import torch
import torch.nn as nn
from .LoRALinear import LoRALinear

class CrossAttentionAdapter(nn.Module):
    """单向 cross-attn：Q 来自 stream A，K/V 来自 stream B"""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_kv: Optional[int] = None,
        lora_r: int = 8,
        lora_alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_kv = d_kv or (d_model // n_heads)
        self.scale = self.d_kv ** -0.5

        self.q = LoRALinear(d_model, n_heads * self.d_kv, r=lora_r, alpha=lora_alpha)
        self.k = LoRALinear(d_model, n_heads * self.d_kv, r=lora_r, alpha=lora_alpha)
        self.v = LoRALinear(d_model, n_heads * self.d_kv, r=lora_r, alpha=lora_alpha)
        self.o = LoRALinear(n_heads * self.d_kv, d_model, r=lora_r, alpha=lora_alpha)

        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(d_model)
        self.gate = nn.Parameter(torch.zeros(1))  # learnable scalar gate

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        # q_tokens: [B, Lq, D], kv_tokens: [B, Lk, D]
        B, Lq, D = q_tokens.shape
        Lk = kv_tokens.shape[1]

        qn = self.norm_q(q_tokens)
        q = self.q(qn).view(B, Lq, self.n_heads, self.d_kv).transpose(1, 2)  # [B,H,Lq,d]
        k = self.k(kv_tokens).view(B, Lk, self.n_heads, self.d_kv).transpose(1, 2)
        v = self.v(kv_tokens).view(B, Lk, self.n_heads, self.d_kv).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,H,Lq,Lk]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        ctx = attn @ v  # [B,H,Lq,d]
        ctx = ctx.transpose(1, 2).contiguous().view(B, Lq, self.n_heads * self.d_kv)
        out = self.o(ctx)

        # Residual with gated injection
        out = q_tokens + torch.tanh(self.gate) * out
        return out


class CoProcessingBridge(nn.Module):
    """Projects both streams to shared dim, exchanges info both ways, and unprojects back."""

    def __init__(
        self,
        d_vit: int,
        d_brain: int,
        n_heads: int = 8,
        d_shared: Optional[int] = None,
        lora_r: int = 8,
        lora_alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_shared = d_shared or d_vit
        self.v_proj = nn.Linear(d_vit, self.d_shared, bias=False)
        self.b_proj = nn.Linear(d_brain, self.d_shared, bias=False)
        self.v2b = CrossAttentionAdapter(
            self.d_shared, n_heads, lora_r=lora_r, lora_alpha=lora_alpha, dropout=dropout
        )
        self.b2v = CrossAttentionAdapter(
            self.d_shared, n_heads, lora_r=lora_r, lora_alpha=lora_alpha, dropout=dropout
        )
        self.v_unproj = nn.Linear(self.d_shared, d_vit, bias=False)
        self.b_unproj = nn.Linear(self.d_shared, d_brain, bias=False)

    def forward(
        self, vit_tokens: torch.Tensor, brain_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        V = self.v_proj(vit_tokens)
        B = self.b_proj(brain_tokens)
        V_new = self.b2v(V, B)
        B_new = self.v2b(B, V)
        vit_tokens = vit_tokens + self.v_unproj(V_new - V)
        brain_tokens = brain_tokens + self.b_unproj(B_new - B)
        return vit_tokens, brain_tokens