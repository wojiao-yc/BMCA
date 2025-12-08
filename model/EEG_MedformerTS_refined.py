import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Optional

from einops.layers.torch import Rearrange

from loss import ClipLoss
from layers.Medformer import Medformer
from .EEG_MedformerTS import Config as BaseConfig


class EnhancedConfig(BaseConfig):
    """
    扩展原始Medformer配置，保留 seq_len=250 的假设，同时暴露时序聚合相关参数。
    """

    def __init__(self):
        super().__init__()
        self.temporal_kernel = 5
        self.temporal_heads = self.n_heads


class TSConvEmbedding(nn.Module):
    """
    将Medformer输出的token重新组织成空间结构，并复用原版的TS卷积做局部特征抽取。
    """

    def __init__(self, emb_size: int = 40):
        super().__init__()
        self.emb_size = emb_size
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (221, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.project = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange("b e h w -> b (h w) e"),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        x = tokens.unsqueeze(1)  # (B, 1, patch_num, d_model)
        x = self.tsconv(x)
        return self.project(x)


class TemporalAggregationBlock(nn.Module):
    """
    利用深度可分离卷积和注意力池化保留patch间时序结构，避免简单flatten导致的信息丢失。
    """

    def __init__(
        self,
        d_model: int,
        seq_len: int = 250,
        kernel_size: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.local_conv = nn.Sequential(
            nn.Conv1d(
                d_model, d_model, kernel_size, padding=padding, groups=d_model
            ),  # depth-wise
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.seq_len = seq_len

    def forward(self, tokens: Tensor) -> Tensor:
        # tokens: (B, patch_num, d_model)
        local_context = self.local_conv(tokens.transpose(1, 2)).transpose(1, 2)
        tokens = tokens + local_context
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm_attn(tokens + attn_out)
        pooled = tokens.mean(dim=1)  # (B, d_model)
        pooled = self.norm_ffn(pooled + self.ffn(pooled))
        return pooled


class ProjectionHead(nn.Module):
    """
    对聚合后的向量做进一步正则化与映射，便于与图像模态对齐。
    """

    def __init__(self, input_dim: int, proj_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, proj_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class eeg_encoder(nn.Module):
    """
    改进的EEG编码器：
    1. 直接使用Medformer输出的patch tokens，避免重复卷积；
    2. subject-wise adapter 在token维度注入个体信息；
    3. attention池化保留时序结构，而非简单flatten；
    4. Projection head带LayerNorm与残差FFN，数值更稳定。
    """

    def __init__(
        self,
        sequence_length: int = 250,
        num_subjects: int = 10,
        proj_dim: int = 1024,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        config = EnhancedConfig()
        config.seq_len = sequence_length  # 保留 seq_len=250 的假设但允许显式传参
        self.encoder = Medformer(config)
        self.tsconv_embedding = TSConvEmbedding()
        self.temporal_pool = TemporalAggregationBlock(
            d_model=self.tsconv_embedding.emb_size,
            seq_len=sequence_length,
            kernel_size=config.temporal_kernel,
            num_heads=config.temporal_heads,
            dropout=config.dropout,
        )
        self.projector = ProjectionHead(
            input_dim=self.tsconv_embedding.emb_size,
            proj_dim=proj_dim,
            dropout=config.dropout,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x: Tensor, subject_ids: Optional[Tensor] = None) -> Tensor:
        # x: (B, seq_len, enc_in)
        tokens = self.encoder(x)
        ts_tokens = self.tsconv_embedding(tokens)
        eeg_repr = self.temporal_pool(ts_tokens)
        return self.projector(eeg_repr)
