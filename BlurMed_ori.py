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

def train_one_step(text_features_all, img_features_all, dataloader, optimizer=None, device=None, model=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = eeg_encoder()

    model.to(device)
    model.train()
    text_features_all = text_features_all.to(device).float()
    img_features_all = (img_features_all[::10]).to(device).float()

    total_loss = 0
    correct = 0
    total = 0
    features_list = []

    for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)

        optimizer.zero_grad()
        batch_size = eeg_data.size(0)

        eeg_features = model(eeg_data)
        features_list.append(eeg_features)
        logit_scale = model.logit_scale

        eeg_alignment_loss = model.loss_func(img_features, eeg_features, logit_scale)
        loss = eeg_alignment_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        logits_img = logit_scale * eeg_features @ img_features_all.T
        logits_single = logits_img
        predicted = torch.argmax(logits_single, dim=1)

        batch_size = predicted.shape[0]
        total += batch_size
        correct += (predicted == labels).sum().item()

        del eeg_data, labels, text, text_features, img, img_features

    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total
    return average_loss, accuracy, torch.cat(features_list, dim=0)


def evaluate_model(text_features_all, img_features_all, dataloader, optimizer=None, device=None, model=None, k=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = eeg_encoder().to(device)

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model.eval()
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()

    total_loss = 0
    correct = 0
    total = 0
    top5_correct_count = 0
    all_labels = set(range(text_features_all.size(0)))

    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()

            eeg_features = model(eeg_data)
            logit_scale = model.logit_scale

            eeg_alignment_loss = model.loss_func(eeg_features, eeg_features, logit_scale)
            loss = eeg_alignment_loss
            total_loss += loss.item()

            for idx, label in enumerate(labels):
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k - 1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]

                logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                logits_single = logits_img
                predicted_label = selected_classes[torch.argmax(logits_single).item()]

                if predicted_label == label.item():
                    correct += 1

                if k in [50, 100, 200]:
                    _, top5_indices = torch.topk(logits_single, 5, largest=True)
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:
                        top5_correct_count += 1
                total += 1

    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total
    top5_acc = top5_correct_count / total if k in [50, 100, 200] else 0
    return average_loss, accuracy, top5_acc

# =====================================================
#  训练循环
# =====================================================
def main_train_loop(mode_name, epochs, model, train_loader, test_loader, optimizer, device,
                    text_features_train_all, text_features_test_all,
                    img_features_train_all, img_features_test_all,
                    save_root, subject):

    save_dir = os.path.join(save_root, mode_name, subject)
    os.makedirs(save_dir, exist_ok=True)

    csv_file = os.path.join(save_dir, "train_log.csv")

    # 初始化日志
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc",
                         "top5_acc", "v2_acc", "v4_acc", "v10_acc", "v50_acc", "v100_acc"])

    # =============================
    # epoch 循环
    # =============================
    for epoch in range(epochs):

        train_loss, train_acc, _ = train_one_step(
            text_features_train_all,
            img_features_train_all,
            train_loader,
            optimizer,
            device,
            model,
        )

        test_loss, test_acc, top5_acc = evaluate_model(
            text_features_test_all,
            img_features_test_all,
            test_loader,
            optimizer,
            device,
            model,
            k=200,
        )

        # 多阈值评估 k={2,4,10,50,100}
        _, v2_acc, _ = evaluate_model(text_features_test_all, img_features_test_all, test_loader, optimizer, device, model, k=2)
        _, v4_acc, _ = evaluate_model(text_features_test_all, img_features_test_all, test_loader, optimizer, device, model, k=4)
        _, v10_acc, _ = evaluate_model(text_features_test_all, img_features_test_all, test_loader, optimizer, device, model, k=10)
        _, v50_acc, _ = evaluate_model(text_features_test_all, img_features_test_all, test_loader, optimizer, device, model, k=50)
        _, v100_acc, _ = evaluate_model(text_features_test_all, img_features_test_all, test_loader, optimizer, device, model, k=100)

        print(
            f"[{mode_name}] {subject} Epoch {epoch+1}/{epochs} | "
            f"Train Loss {train_loss:.4f}  Train Acc {train_acc:.4f}  "
            f"Test Acc {test_acc:.4f}"
        )

        # 写入 CSV
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, train_loss, train_acc, test_loss, test_acc, top5_acc,
                v2_acc, v4_acc, v10_acc, v50_acc, v100_acc
            ])

        # 每5轮保存模型
        if (epoch + 1) % 5 == 0:
            model_path = os.path.join(save_dir, f"epoch{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"==== 模型保存 {model_path} ====")


# =====================================================
#  3种模式 + 循环所有 subject
# =====================================================
def run_all_modes():

    data_path = "/mnt/dataset0/ldy/datasets/THINGS_EEG/Preprocessed_data_250Hz"
    save_root = "/home/wenxiao/workspace/qhy/BMCA/data"
    os.makedirs(save_root, exist_ok=True)

    subjects = [f"sub-{i:02d}" for i in range(1, 11)]

    lr = 3e-4
    epochs = 50
    batch_size = 128

    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

    # 三种模式名称（用于文件夹名）
    modes = {
        # 1: "in-subject",
        # 2: "joint-subject",
        3: "inter-subject",
    }

    for mode_id, mode_name in modes.items():

        print(f"\n========== 运行模式: {mode_name} ==========\n")

        for test_sub in subjects:

            # ---- 选择训练 subject ----
            if mode_id == 1:     # 单 subject 训练+测试
                train_subjects = [test_sub]
            elif mode_id == 2:   # 所有 subject 训练，一个测试
                train_subjects = subjects
            elif mode_id == 3:   # 其他9个 subject 训练，一个测试
                train_subjects = [s for s in subjects if s != test_sub]
            else:
                raise ValueError("mode must be 1,2,3")

            print(f"[{mode_name}] train={train_subjects}  test={test_sub}")

            # ---- 数据加载 ----
            train_dataset = EEGDataset(data_path, adap_subject=test_sub, subjects=train_subjects, train=True)
            test_dataset = EEGDataset(data_path, adap_subject=test_sub, subjects=[test_sub], train=False)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)

            # ---- 模型与优化器 ----
            model = eeg_encoder()
            optimizer = AdamW(model.parameters(), lr=lr)

            # ---- 训练 ----
            main_train_loop(
                mode_name,
                epochs,
                model,
                train_loader,
                test_loader,
                optimizer,
                device,
                train_dataset.text_features,
                test_dataset.text_features,
                train_dataset.img_features,
                test_dataset.img_features,
                save_root,
                test_sub
            )

        print(f"==== {mode_name} 模式已完成 ====\n")

    print("======= 所有三种模式训练已完成 =======\n")


if __name__ == "__main__":
    run_all_modes()
