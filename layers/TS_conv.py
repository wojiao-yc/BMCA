from einops.layers.torch import Rearrange
import torch.nn as nn

class PatchEmbedding256(nn.Module):
    """EEG patch embedding → 256 patches × 1280 dim"""
    def __init__(self, emb_size=1280):
        super().__init__()

        # 1) spatial conv: (63 → 1)
        self.spatial = nn.Conv2d(1, 64, kernel_size=(63, 1))

        # 2) temporal pool: (250 → 16)
        self.temporal = nn.AvgPool2d(kernel_size=(1, 15), stride=(1, 15))

        # 3) upsample height: (1 → 16)
        self.expand_h = nn.ConvTranspose2d(64, 64, kernel_size=(16, 1), stride=(16, 1))

        # 4) project to CLIP token dim 1280
        self.proj = nn.Conv2d(64, emb_size, kernel_size=1)

        # 5) rearrange → 256 tokens
        self.to_token = Rearrange("b c h w -> b (h w) c")

    def forward(self, x):
        """
        x: (B, 63, 250)
        return: (B, 256, 1280)
        """

        x = x.unsqueeze(1)               # (B,1,63,250)
        x = self.spatial(x)              # (B,64,1,250)
        x = self.temporal(x)             # (B,64,1,16)
        x = self.expand_h(x)             # (B,64,16,16)
        x = self.proj(x)                 # (B,1280,16,16)
        x = self.to_token(x)             # (B,256,1280)
        return x


class Enc_eeg_256(nn.Module):
    """Full EEG encoder producing 256×1280 tokens"""
    def __init__(self):
        super().__init__()
        self.patch = PatchEmbedding256(emb_size=1280)

    def forward(self, x):
        return self.patch(x)   # (B,256,1280)


class PatchEmbedding1(nn.Module):
    """EEG embedding → 1 × 1280"""
    def __init__(self, emb_size=1280):
        super().__init__()

        # ↓ 与 256×1280 版本完全一致
        self.spatial = nn.Conv2d(1, 64, kernel_size=(63, 1))
        self.temporal = nn.AvgPool2d((1, 15), (1, 15))
        self.expand_h = nn.ConvTranspose2d(64, 64, kernel_size=(16, 1), stride=(16, 1))
        self.proj = nn.Conv2d(64, emb_size, kernel_size=1)

        # 全局池化 → 1 token
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        x: (B, 63, 250)
        return: (B, 1, 1280)
        """

        x = x.unsqueeze(1)
        x = self.spatial(x)
        x = self.temporal(x)
        x = self.expand_h(x)
        x = self.proj(x)            # (B,1280,16,16)

        x = self.global_pool(x)     # (B,1280,1,1)
        x = x.squeeze(-1).transpose(1,2)  # (B,1,1280)

        return x


class Enc_eeg_1(nn.Module):
    """Full EEG encoder producing 1×1280"""
    def __init__(self):
        super().__init__()
        self.patch = PatchEmbedding1(emb_size=1280)

    def forward(self, x):
        return self.patch(x)      # (B,1,1280)

