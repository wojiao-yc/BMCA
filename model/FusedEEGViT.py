import torch
import torch.nn as nn
from typing import Tuple, Iterable
import open_clip
from .BrainEncoderAdapter import BrainEncoderAdapter
from layers.CrossAttention_Bridge import CoProcessingBridge
from loss import ClipLoss
import numpy as np

class FusedEEGViT(nn.Module):
    """
    open_clip ViT-H/14 + EEG (Medformer) 双流协同处理

    关键点：
      - open_clip visual (ViT-H/14) 冻结，只用作视觉编码和对比目标
      - EEG → tsconv (Enc_eeg_1) → Medformer encoder
      - 在指定的 (vit_layer, med_layer) 对上，做双向 cross-attention bridge
      - 最终得到和 ViT-H/14 hidden 维度一致的 fused embedding，并通过 gate 和 z_pure 融合
    """

    def __init__(
        self,
        # bridge_pairs: Iterable[Tuple[int, int]] = ((6, 0), (12, 1), (18, 2)),
        bridge_pairs: Iterable[Tuple[int, int]] = ((6, 0), ),
        n_heads_bridge: int = 2,
        lora_r: int = 8,
        med_depth: int = 3,
        model_type: str = "ViT-H-14",
        pretrained: str = "laion2b_s32b_b79k",
    ):
        super().__init__()

        # 1) 创建 open_clip 模型，只用 visual 分支
        clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_type, pretrained=pretrained
        )
        self.visual = clip_model.visual  # VisionTransformer

        # 冻结 visual
        for p in self.visual.parameters():
            p.requires_grad = False
        self.visual.eval()

        self.vit_hidden = getattr(self.visual, "width", None)
        if self.vit_hidden is None:
            self.vit_hidden = self.visual.transformer.width

        self.vit_final = getattr(self.visual, "output_dim", None)
        if self.vit_final is None:
            # 旧版本可能叫 embed_dim
            self.vit_final = getattr(self.visual, "embed_dim", None)

        assert self.vit_hidden == 1280, f"Expected final embedding dim 1280, got {self.vit_hidden}"

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

        # 2) Brain 侧：tsconv + Medformer（d_model_med = vit_hidden）
        self.brain_adapter = BrainEncoderAdapter(
            seq_len=250,
            enc_in=63,
            d_model_med=self.vit_hidden,
            depth=med_depth,
        )
        self.brain_dim = self.brain_adapter.d_model_med
        self.med_num_layers = len(self.brain_adapter.layers)
        self.brain_attn_pool = nn.MultiheadAttention(
            embed_dim=1280,
            num_heads=8,
            batch_first=True,
        )
        self.brain_proj = nn.Linear(1280, 1280)


        # 3) (vit_layer, med_layer) 的映射表
        bridge_pairs = list(bridge_pairs)
        self.bridge_pairs = bridge_pairs

        for _, m_idx in bridge_pairs:
            assert 0 <= m_idx < self.med_num_layers, (
                f"med_layer index {m_idx} 超出 Medformer depth {self.med_num_layers}"
            )

        self.vit_bridge_layers = sorted(set(v for v, _ in bridge_pairs))

        self.vit_to_med = {}
        for v_idx, m_idx in bridge_pairs:
            if v_idx in self.vit_to_med:
                raise ValueError(
                    f"vit_layer {v_idx} 在 bridge_pairs 里出现多次，"
                    "如果你确实想在同一层上多次 bridge，需要自己扩展实现。"
                )
            self.vit_to_med[v_idx] = m_idx

        # 4) 为所有 vit_bridge_layers 准备 bridge module（双向 cross-attn）
        self.bridges = nn.ModuleDict()
        for vit_layer in self.vit_bridge_layers:
            key = str(vit_layer)
            self.bridges[key] = CoProcessingBridge(
                d_vit=self.vit_hidden,
                d_brain=self.brain_dim,
                n_heads=n_heads_bridge,
                lora_r=lora_r,
                lora_alpha=16,
                dropout=0.0,
            )

        # 5) gating for final fused embedding vs original CLIP embedding
        self.gate_head = nn.Sequential(
            nn.LayerNorm(self.vit_final),
            nn.Linear(self.vit_final, self.vit_final // 4),
            nn.GELU(),
            nn.Linear(self.vit_final // 4, 1),
            nn.Sigmoid(),  # per-sample scalar g in [0,1]
        )

        self.final_norm = nn.LayerNorm(self.vit_hidden)

    # ---------------- ViT token embedding (open_clip visual) ----------------
    def vit_embed_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Run open_clip visual patch embedding + positional + ln_pre
        返回 tokens（encoder 之前）: [B, 1+N_patch, D]
        对 ViT-H/14 而言：N_patch = 256
        """
        
        v = self.visual
        x = v.conv1(pixel_values)        # [B, D, H/patch, W/patch] = [B, D, 16,16]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B,D,256]
        x = x.permute(0, 2, 1)           # [B,256,D]

        class_emb = v.class_embedding.to(x.dtype)
        class_emb = class_emb + torch.zeros(
            x.shape[0], 1, x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )  # [B,1,D]
        x = torch.cat([class_emb, x], dim=1)  # [B,257,D]

        pos = v.positional_embedding.to(x.dtype)
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
        x = x + pos
        # print("x:", x.shape)
        x = v.ln_pre(x)  # [B,257,D]
        return x

    # ---------------- ViT + Medformer 联合前进（open_clip） ----------------
    def run_encoder_with_bridges(
        self,
        tokens: torch.Tensor,
        brain_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        tokens: [B, 1+256, D_vit]
        brain_tokens: [B, L_b, D_brain]

        open_clip 的 transformer 接受形状 [L,B,D]，所以需要 permute。
        """
        vit_enc = self.visual.transformer
        med_layers = self.brain_adapter.layers
        med_norm = self.brain_adapter.norm

        current_med_idx = 0
        # print("tokens:", tokens.shape)
        # 遍历 ViT 的每一层
        for i, blk in enumerate(vit_enc.resblocks):
            vit_layer_id = i + 1  # 1-based

            # 1) ViT 正常前进一层（需要 permute 成 [L,B,D]）
            x = tokens.permute(1, 0, 2)   # [L,B,D]
            x = blk(x)                    # [L,B,D]
            tokens = x.permute(1, 0, 2)   # [B,L,D]

            # 2) 检查是否在这个层后插 bridge
            if vit_layer_id in self.vit_to_med:
                target_med_idx = self.vit_to_med[vit_layer_id]

                # 2.1 先把 Medformer 跑到 target_med_idx
                while current_med_idx <= target_med_idx and current_med_idx < len(
                    med_layers
                ):
                    brain_tokens = self.brain_adapter.step_layer(
                        brain_tokens, current_med_idx, attn_mask=None
                    )
                    current_med_idx += 1

                # 2.2 当前 ViT patch tokens 和 brain_tokens 做一次双向 cross-attn bridge
                cls_tok, patch_tok = tokens[:, :1, :], tokens[:, 1:, :]
                bridge = self.bridges[str(vit_layer_id)]
                patch_tok, brain_tokens = bridge(patch_tok, brain_tokens)
                tokens = torch.cat([cls_tok, patch_tok], dim=1)

        # 3) ViT 全部层跑完后，把 Medformer 剩余层也跑完
        while current_med_idx < len(med_layers):
            brain_tokens = self.brain_adapter.step_layer(
                brain_tokens, current_med_idx, attn_mask=None
            )
            current_med_idx += 1

        if med_norm is not None:
            brain_tokens = med_norm(brain_tokens)

        # open_clip 的 ln_post 是 LayerNorm，可以直接作用在所有 tokens 上
        tokens = self.visual.ln_post(tokens)  # [B,257,D]
        return tokens, brain_tokens

    # ---------------- ViT 纯路径（无 EEG，无 bridge） ----------------
    def vit_forward_pure(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        用 open_clip visual 做一个“纯视觉”前向，
        不插入 EEG bridge，用于得到 z_pure 作为对比目标。
        """
        tokens = self.vit_embed_tokens(pixel_values)  # [B,257,D]
        vit_enc = self.visual.transformer

        x = tokens
        x = vit_enc(x)
        # for blk in vit_enc.resblocks:
        #     xb = x.permute(1, 0, 2)  # [L,B,D]
        #     xb = blk(xb)
        #     x = xb.permute(1, 0, 2)  # [B,L,D]

        x = self.visual.ln_post(x)  # [B,257,D]
        cls = x[:, 0]               # [B,D]
        return cls

    # ---------------- pooling & forward ----------------
    @staticmethod
    def _cls_pool(tokens: torch.Tensor) -> torch.Tensor:
        return tokens[:, 0]  # [B, D]

    def forward(self, pixel_values: torch.Tensor, eeg: torch.Tensor) -> dict:
        """
        pixel_values: [B,3,224,224] (建议先用 self.preprocess 预处理)
        eeg: [B,63,250]
        返回：
          - z_pure: 冻结的纯 CLIP CLS embedding
          - z_fused: 注入 EEG 后的 CLS embedding
          - z_gate: gate 后的最终 embedding
          - gate:   每个样本的 scalar gate
        """
        # 1) 纯 CLIP embedding（对比目标，不参与梯度）
        with torch.no_grad():
            z_pure = self.vit_forward_pure(pixel_values)  # [B,D]
            z_pure_detached = z_pure.detach()

        # 2) EEG → tsconv patch tokens（Medformer 输入）
        brain_tokens = self.brain_adapter.embed(eeg)  # [B,L_b,D_brain]

        # 3) ViT tokens before encoder
        vit_tokens = self.vit_embed_tokens(pixel_values)  # [B,1+256,D_vit]
        # print("vit_tokens", vit_tokens.shape)
        # 4) ViT + Medformer 联合前进 + 多次桥接
        vit_tokens, brain_tokens = self.run_encoder_with_bridges(
            vit_tokens, brain_tokens
        )
        B, L, D = brain_tokens.shape
        q = torch.zeros(B, 1, D).to(brain_tokens.device)
        # Attention pooled embedding
        out, _ = self.brain_attn_pool(q, brain_tokens, brain_tokens)   # [B,1,1280]
        brain_tokens = self.brain_proj(out.squeeze(1))  # [B,1280]
        # 5) Fused embedding from class token
        z_fused = self._cls_pool(vit_tokens)  # [B,D]
        z_fused = self.final_norm(z_fused)

        # 6) gate：在 z_fused 和 z_pure 之间做加权

        z_fused = z_fused @ self.visual.proj
        z_pure_detached = z_pure_detached @ self.visual.proj
        brain_tokens = brain_tokens @ self.visual.proj

        g = self.gate_head(z_fused).squeeze(-1)  # [B]
        z_gate = g.unsqueeze(-1) * brain_tokens + (1.0 - g).unsqueeze(-1) * z_pure_detached

        return {
            "z_pure": z_pure_detached,
            "z_fused": z_fused,
            "z_gate": z_gate,
            "gate": g,
            "brain_tokens": brain_tokens,
        }