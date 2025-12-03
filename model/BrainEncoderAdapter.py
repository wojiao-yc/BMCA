import torch
import torch.nn as nn
from layers.Simple_Transformer import SimpleTransformerEncoder
from layers.TS_conv import Enc_eeg_256

class BrainEncoderAdapter(nn.Module):
    """
    EEG 经 tsconv patch embedding + Medformer encoder（支持逐层 step）

    注意：这里假设你在别处已经实现了：
      - Config
      - Medformer(Config)
    """

    def __init__(
        self,
        seq_len: int = 250,
        enc_in: int = 63,
        d_model_med: int = 1280,  # 对齐 ViT-H/14 的 hidden width
        depth: int = 3,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.d_model_med = d_model_med

        # 2) EEG patch embedding → [B,1,d_model_med]
        self.patch_embed = Enc_eeg_256()

        self.encoder = SimpleTransformerEncoder(
            depth=depth,
            d_model=d_model_med,
            n_heads=8,
            dropout=0.1
        )
        self.layers = self.encoder.layers
        self.norm = self.encoder.norm

    # --- 只做 tsconv patch embedding ---
    def embed(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        eeg: [B, 63, 250]
        return: [B, L_b, d_model_med]  (这里 L_b = 1)
        """
        brain_tokens = self.patch_embed(eeg)  # [B, 1, d_model_med]
        # print("brain_tokens after patch_embed:", brain_tokens.shape)
        return brain_tokens

    # --- 单层前进（给 FusedEEGViT 调用） ---
    def step_layer(self, x: torch.Tensor, layer_idx: int, attn_mask=None) -> torch.Tensor:
        layer = self.layers[layer_idx]
        # print(f"[Medformer layer {layer_idx}] input x:", x.shape)

        out = layer(x, attn_mask=attn_mask)
        if isinstance(out, (tuple, list)):
            out0 = out[0]
        else:
            out0 = out

        # print(f"[Medformer layer {layer_idx}] output out0:", out0.shape)
        return out0