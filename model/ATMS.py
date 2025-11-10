import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops.layers.torch import Rearrange

from loss import ClipLoss

# Custom transformer components
from layers.Medformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import MedformerLayer
from layers.Embed import ListPatchEmbedding



class Config:
    """Configuration class for model hyperparameters"""
    def __init__(self, depth=1):
        self.task_name = 'classification'
        self.seq_len = 250                      # Input sequence length
        self.pred_len = 250                     # Prediction length
        self.output_attention = False           # Whether to output attention weights
        self.d_model = 250                      # Model dimension
        self.embed = 'timeF'                    # Time encoding method
        self.freq = 'h'                         # Time frequency
        self.dropout = 0.1 * depth / 5          # Dropout ratio (scaled by depth)
        self.factor = 1                         # Attention scaling factor
        self.n_heads = depth                    # Number of attention heads
        self.e_layers = depth                   # Number of encoder layers
        self.d_ff = 64 * depth                  # Feedforward network dimension
        self.activation = 'gelu'                # Activation function
        self.enc_in = 63                        # Encoder input dimension
        
        # Additional Medformer-specific configs
        self.single_channel = False
        self.patch_len_list = "2,4,8"           # Patch lengths for embedding
        self.augmentations = "flip,shuffle,frequency,jitter,mask,drop"  # Data augmentations
        self.no_inter_attn = False              # Whether to use inter-attention
        self.num_class = 250                    # Number of output classes


class Medformer(nn.Module):
    """
    Medformer model for EEG classification
    Based on Transformer architecture with O(L^2) complexity
    Paper reference: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """
    def __init__(self, configs):
        super(Medformer, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in
        self.single_channel = configs.single_channel
        
        # Patch embedding configuration
        patch_len_list = list(map(int, configs.patch_len_list.split(",")))
        stride_list = patch_len_list
        seq_len = configs.seq_len
        patch_num_list = [
            int((seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]
        augmentations = configs.augmentations.split(",")
        
        # Embedding layer
        self.enc_embedding = ListPatchEmbedding(
            configs.enc_in,
            configs.d_model,
            patch_len_list,
            stride_list,
            configs.dropout,
            augmentations,
            configs.single_channel,
        )
        
        # Transformer encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MedformerLayer(
                        len(patch_len_list),
                        configs.d_model,
                        configs.n_heads,
                        configs.dropout,
                        configs.output_attention,
                        configs.no_inter_attn,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        
        # Classification head
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * sum(patch_num_list) * 
                (1 if not self.single_channel else configs.enc_in),
                configs.num_class,
            )

    def classification(self, x_enc, x_mark_enc):
        """Forward pass for classification task"""
        # Embedding
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        if self.single_channel:
            enc_out = torch.reshape(enc_out, (-1, self.enc_in, *enc_out.shape[-2:]))

        # Output processing
        output = self.act(enc_out)  # Apply non-linearity
        output = self.dropout(output)
        return output

    def forward(self, x_enc, x_mark_enc=None):
        """Main forward pass"""
        return self.classification(x_enc, x_mark_enc)



class PatchEmbedding(nn.Module):
    """EEG signal patch embedding module"""
    def __init__(self, emb_size=40):
        super().__init__()
        # Temporal-spatial convolution from ShallowNet
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

        # Projection to embedding space
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass"""
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.tsconv(x)  # Temporal-spatial convolution
        x = self.projection(x)  # Project to embedding space
        return x


class ResidualAdd(nn.Module):
    """Residual connection wrapper"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        """Forward pass with residual connection"""
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    """Flattening layer for classification head"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Flatten input tensor"""
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    """EEG encoder module"""
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )


class Proj_eeg(nn.Sequential):
    """Projection head for EEG features"""
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class ATMS(nn.Module):
    """ATMS (Adaptive Transformer for Multi-Subject EEG) model"""
    def __init__(self, sequence_length=250, num_subjects=10, joint_train=False, configs=None):
        super(ATMS, self).__init__()
        default_config = Config(configs.depth)
        
        # Model components
        self.encoder = Medformer(default_config)
        self.subject_wise_linear = nn.ModuleList([
            nn.Linear(default_config.d_model, sequence_length) 
            for _ in range(num_subjects)
        ])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()
        
        # CLIP-like loss components
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x, subject_ids):
        """Forward pass"""
        x = self.encoder(x)
        eeg_embedding = self.enc_eeg(x)
        out = self.proj_eeg(eeg_embedding)
        return out