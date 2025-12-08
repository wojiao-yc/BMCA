import os
os.environ['http_proxy'] = 'http://127.0.0.1:7896'
os.environ['https_proxy'] = 'http://127.0.0.1:7896'
os.environ['all_proxy'] = 'socks5://127.0.0.1:7897'

from typing import Iterable, List, Optional, Tuple
import open_clip
import argparse
import math
import torch
import random
import itertools
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from PIL import Image
from transformers import CLIPVisionModel
from data_preparing.blureegdatasets import EEGDataset
from model.EEG_MedformerTS import eeg_encoder
import csv


class ImageFeatureAttention(nn.Module):
    """Lightweight self-attention block to refine CLIP image embeddings."""

    def __init__(self, embed_dim=1024, num_heads=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, img_features: torch.Tensor) -> torch.Tensor:
        # Treat each feature vector as a single token and refine it with attention + FFN
        x = img_features.unsqueeze(1)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm(attn_out + x)
        x = x + self.ffn(x)
        return x.squeeze(1)


def train_one_step(
    text_features_all,
    img_features_all,
    dataloader,
    optimizer=None,
    device=None,
    model=None,
    img_attention=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = eeg_encoder()

    if img_attention is None:
        raise ValueError("img_attention module must be provided for training")

    model.to(device)
    img_attention.to(device)
    model.train()
    img_attention.train()
    text_features_all = text_features_all.to(device).float()
    img_features_all = (img_features_all[::10]).to(device).float()
    attended_img_features_all = img_attention(img_features_all).to(device).float()

    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.5  # Weight for attention-image vs attention-EEG losses
    features_list = []

    for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)

        optimizer.zero_grad()
        batch_size = eeg_data.size(0)

        eeg_features = model(eeg_data)
        eeg_features = img_attention(eeg_features)
        attended_img_features = img_attention(img_features)
        features_list.append(eeg_features)
        logit_scale = model.logit_scale

        img_reconstruction_loss = model.loss_func(attended_img_features, img_features, logit_scale)
        eeg_alignment_loss = model.loss_func(attended_img_features, eeg_features, logit_scale)
        # eeg_alignment_loss = model.loss_func(img_features, eeg_features, logit_scale)
        loss = alpha * img_reconstruction_loss + (1 - alpha) * eeg_alignment_loss
        # loss =  eeg_alignment_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        logits_img = logit_scale * eeg_features @ attended_img_features_all.T
        logits_single = logits_img
        predicted = torch.argmax(logits_single, dim=1)

        batch_size = predicted.shape[0]
        total += batch_size
        correct += (predicted == labels).sum().item()

        del eeg_data, labels, text, text_features, img, img_features

    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total
    return average_loss, accuracy, torch.cat(features_list, dim=0)


def evaluate_model(
    text_features_all,
    img_features_all,
    dataloader,
    optimizer=None,
    device=None,
    model=None,
    img_attention=None,
    k=None,
):
    """Evaluation function with k-way classification."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = eeg_encoder().to(device)

    if img_attention is None:
        raise ValueError("img_attention module must be provided for evaluation")

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model.eval()
    img_attention.to(device)
    img_attention.eval()
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    attended_img_features_all = img_attention(img_features_all).to(device).float()

    total_loss = 0
    correct = 0
    correct_attended = 0
    total = 0
    alpha = 0.5
    top5_correct_count = 0
    top5_attended_correct_count = 0
    all_labels = set(range(text_features_all.size(0)))

    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()

            eeg_features = model(eeg_data)
            # attended_img_features = img_attention(img_features)
            attended_img_features = img_features
            logit_scale = model.logit_scale

            img_reconstruction_loss = model.loss_func(attended_img_features, img_features, logit_scale)
            eeg_alignment_loss = model.loss_func(attended_img_features, eeg_features, logit_scale)
            loss = img_reconstruction_loss * alpha + eeg_alignment_loss * (1 - alpha)
            total_loss += loss.item()

            for idx, label in enumerate(labels):
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k - 1) + [label.item()]

                selected_img_features = img_features_all[selected_classes]
                selected_attended_img_features = attended_img_features_all[selected_classes]

                logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                logits_single = logits_img
                predicted_label = selected_classes[torch.argmax(logits_single).item()]

                if predicted_label == label.item():
                    correct += 1

                logits_attended = logit_scale * eeg_features[idx] @ selected_attended_img_features.T
                predicted_label_attended = selected_classes[torch.argmax(logits_attended).item()]
                if predicted_label_attended == label.item():
                    correct_attended += 1

                if k in [50, 100, 200]:
                    _, top5_indices = torch.topk(logits_single, 5, largest=True)
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:
                        top5_correct_count += 1
                    _, top5_attended_indices = torch.topk(logits_attended, 5, largest=True)
                    if label.item() in [selected_classes[i] for i in top5_attended_indices.tolist()]:
                        top5_attended_correct_count += 1
                total += 1

    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total
    attended_accuracy = correct_attended / total
    top5_acc = top5_correct_count / total if k in [50, 100, 200] else 0
    top5_attended_acc = top5_attended_correct_count / total if k in [50, 100, 200] else 0
    return average_loss, accuracy, top5_acc, attended_accuracy, top5_attended_acc


# =====================================================
#  带多模式输出与结果保存的训练循环（参照第一部分）
# =====================================================
def main_train_loop(
    mode_name,
    epochs,
    model,
    img_attention,
    train_dataloader,
    test_dataloader,
    optimizer,
    device,
    text_features_train_all,
    text_features_test_all,
    img_features_train_all,
    img_features_test_all,
    save_root,
    subject,
):
    if img_attention is None:
        raise ValueError("img_attention must be provided to the training loop")

    if optimizer is None:
        optimizer = AdamW(
            itertools.chain(model.parameters(), img_attention.parameters()),
            lr=3e-4,
        )

    # ========= 结果保存路径：/home/.../BMCA/data/joint-space/mode/subject =========
    save_dir = os.path.join(save_root, mode_name, subject)
    os.makedirs(save_dir, exist_ok=True)

    csv_file = os.path.join(save_dir, "train_log.csv")
    log_file = os.path.join(save_dir, "log.txt")

    # 初始化 CSV 日志
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "test_loss",
            "test_acc",
            "att_test_acc",
            "top5_acc",
            "att_top5_acc",
            "v2_acc",
            "v2_att_acc",
            "v4_acc",
            "v4_att_acc",
            "v10_acc",
            "v10_att_acc",
            "v50_acc",
            "v50_att_acc",
            "v50_top5_acc",
            "v50_att_top5_acc",
            "v100_acc",
            "v100_att_acc",
            "v100_top5_acc",
            "v100_att_top5_acc",
        ])

    for epoch in range(epochs):
        # ===================== Train =====================
        train_loss, train_accuracy, features_tensor = train_one_step(
            text_features_train_all,
            img_features_train_all,
            train_dataloader,
            optimizer,
            device,
            model,
            img_attention,
        )

        # ===================== Eval: k=200 =====================
        test_loss, test_accuracy, top5_acc, attended_test_accuracy, attended_top5_acc = evaluate_model(
            text_features_test_all,
            img_features_test_all,
            test_dataloader,
            optimizer,
            device,
            model,
            img_attention,
            k=200,
        )

        # 多阈值评估 k={2,4,10,50,100}
        _, v2_acc, _, v2_att_acc, _ = evaluate_model(
            text_features_test_all,
            img_features_test_all,
            test_dataloader,
            optimizer,
            device,
            model,
            img_attention,
            k=2,
        )
        _, v4_acc, _, v4_att_acc, _ = evaluate_model(
            text_features_test_all,
            img_features_test_all,
            test_dataloader,
            optimizer,
            device,
            model,
            img_attention,
            k=4,
        )
        _, v10_acc, _, v10_att_acc, _ = evaluate_model(
            text_features_test_all,
            img_features_test_all,
            test_dataloader,
            optimizer,
            device,
            model,
            img_attention,
            k=10,
        )
        _, v50_acc, v50_top5_acc, v50_att_acc, v50_att_top5_acc = evaluate_model(
            text_features_test_all,
            img_features_test_all,
            test_dataloader,
            optimizer,
            device,
            model,
            img_attention,
            k=50,
        )
        _, v100_acc, v100_top5_acc, v100_att_acc, v100_att_top5_acc = evaluate_model(
            text_features_test_all,
            img_features_test_all,
            test_dataloader,
            optimizer,
            device,
            model,
            img_attention,
            k=100,
        )

        # ===================== 打印信息 =====================
        msg1 = (
            f"[{mode_name}] {subject} | Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}, "
            f"Att Test Acc: {attended_test_accuracy:.4f}, "
            f"Top5 Acc: {top5_acc:.4f}, Att Top5 Acc: {attended_top5_acc:.4f}"
        )
        print(msg1)

        msg2 = (
            f"[{mode_name}] {subject} | Epoch {epoch + 1}/{epochs} - "
            f"v2 Acc: {v2_acc:.4f} (att: {v2_att_acc:.4f}) - "
            f"v4 Acc: {v4_acc:.4f} (att: {v4_att_acc:.4f}) - "
            f"v10 Acc: {v10_acc:.4f} (att: {v10_att_acc:.4f}) - "
            f"v50 Acc: {v50_acc:.4f} (att: {v50_att_acc:.4f}, "
            f"top5: {v50_top5_acc:.4f}, att_top5: {v50_att_top5_acc:.4f}) - "
            f"v100 Acc: {v100_acc:.4f} (att: {v100_att_acc:.4f}, "
            f"top5: {v100_top5_acc:.4f}, att_top5: {v100_att_top5_acc:.4f})"
        )
        print(msg2)

        # 写入 txt 日志
        with open(log_file, "a") as f:
            f.write(msg1 + "\n")
            f.write(msg2 + "\n")

        # 写入 CSV
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_loss,
                train_accuracy,
                test_loss,
                test_accuracy,
                attended_test_accuracy,
                top5_acc,
                attended_top5_acc,
                v2_acc,
                v2_att_acc,
                v4_acc,
                v4_att_acc,
                v10_acc,
                v10_att_acc,
                v50_acc,
                v50_att_acc,
                v50_top5_acc,
                v50_att_top5_acc,
                v100_acc,
                v100_att_acc,
                v100_top5_acc,
                v100_att_top5_acc,
            ])

        # 每 5 轮保存一次模型（同时保存 EEG 编码器和注意力模块）
        if (epoch + 1) % 5 == 0:
            model_path = os.path.join(save_dir, f"epoch{epoch + 1}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "img_attention_state_dict": img_attention.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                model_path,
            )
            print(f"==== 模型保存 {model_path} ====")


# =====================================================
#  3种模式 + 循环所有 subject（与第一部分一致的多模式结构）
# =====================================================
def run_all_modes():
    data_path = "/mnt/dataset0/ldy/datasets/THINGS_EEG/Preprocessed_data_250Hz"
    # 按需求指定新的保存路径
    # save_root = "/home/wenxiao/workspace/qhy/BMCA/data/joint-space"
    save_root = "/home/wenxiao/workspace/qhy/BMCA/data/VE-space"
    os.makedirs(save_root, exist_ok=True)

    subjects = [f"sub-{i:02d}" for i in range(1, 11)]

    # 保留第二部分原有 hyper-parameters
    lr = 3e-4
    epochs = 100
    batch_size = 256

    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

    # 三种模式名称（用于文件夹名）
    modes = {
        # 1: "in-subject",
        2: "joint-subject",
        # 3: "inter-subject",
    }

    for mode_id, mode_name in modes.items():
        print(f"\n========== 运行模式: {mode_name} ==========\n")

        for test_sub in subjects:
            # ---- 选择训练 subject ----
            if mode_id == 1:      # 单 subject 训练+测试
                train_subjects = [test_sub]
            elif mode_id == 2:    # 所有 subject 训练，一个测试
                train_subjects = subjects
            elif mode_id == 3:    # 其他9个 subject 训练，一个测试
                train_subjects = [s for s in subjects if s != test_sub]
            else:
                raise ValueError("mode must be 1,2,3")

            print(f"[{mode_name}] train={train_subjects}  test={test_sub}")

            # ---- 数据加载 ----
            train_dataset = EEGDataset(
                data_path,
                adap_subject=test_sub,
                subjects=train_subjects,
                train=True,
            )
            test_dataset = EEGDataset(
                data_path,
                adap_subject=test_sub,
                subjects=[test_sub],
                train=False,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=True,
                drop_last=True,
            )

            text_features_train_all = train_dataset.text_features
            text_features_test_all = test_dataset.text_features
            img_features_train_all = train_dataset.img_features
            img_features_test_all = test_dataset.img_features

            feature_dim = img_features_train_all.shape[-1]
            img_attention = ImageFeatureAttention(embed_dim=feature_dim)

            # ---- 模型与优化器 ----
            model = eeg_encoder()
            optimizer = AdamW(
                itertools.chain(model.parameters(), img_attention.parameters()),
                lr=lr,
            )

            # ---- 训练 + 结果保存 ----
            main_train_loop(
                mode_name,
                epochs,
                model,
                img_attention,
                train_loader,
                test_loader,
                optimizer,
                device,
                text_features_train_all,
                text_features_test_all,
                img_features_train_all,
                img_features_test_all,
                save_root,
                test_sub,
            )

        print(f"==== {mode_name} 模式已完成 ====\n")

    print("======= 所有三种模式训练已完成 =======\n")


def main():
    run_all_modes()


if __name__ == "__main__":
    main()
