import torch.nn as nn
import torch.nn.functional as F

class SimpleTransformerLayer(nn.Module):
    def __init__(self, d_model=1280, n_heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # x: [B, L, D]
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        ff = self.linear2(F.gelu(self.linear1(x)))
        x = x + self.dropout(ff)
        x = self.norm2(x)
        return x

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, depth=3, d_model=1280, n_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(d_model, n_heads, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return self.norm(x)