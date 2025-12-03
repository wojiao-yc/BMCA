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
from data_preparing.eegdatasets import EEGDataset
from model.EEG_MedformerTS import eeg_encoder
# from model.custom_pipeline import *

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
    
def train_one_step(text_features_all, img_features_all, dataloader, optimizer=None, device=None, model=None, img_attention=None):
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


def evaluate_model(text_features_all, img_features_all, dataloader, optimizer=None, device=None, model=None, img_attention=None, k=None):
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
    total = 0
    alpha = 0.5
    top5_correct_count = 0
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

                if k in [50, 100, 200]:
                    _, top5_indices = torch.topk(logits_single, 5, largest=True)
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:
                        top5_correct_count += 1
                total += 1

    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total
    top5_acc = top5_correct_count / total if k in [50, 100, 200] else 0
    return average_loss, accuracy, top5_acc

def main_train_loop(epochs, model, img_attention, train_dataloader, test_dataloader, optimizer, device, text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all):
    if img_attention is None:
        raise ValueError("img_attention must be provided to the training loop")

    if optimizer is None:
        optimizer = AdamW(itertools.chain(model.parameters(), img_attention.parameters()), lr=3e-4)
    
    log_file = "log.txt"

    for epoch in range(epochs):
        train_loss, train_accuracy, features_tensor = train_one_step(
            text_features_train_all,
            img_features_train_all,
            train_dataloader,
            optimizer,
            device,
            model,
            img_attention,
        )

        test_loss, test_accuracy, top5_acc = evaluate_model(
            text_features_test_all,
            img_features_test_all,
            test_dataloader,
            optimizer,
            device,
            model,
            img_attention,
            k=200,
        )
        _, v2_acc, _ = evaluate_model(
            text_features_test_all,
            img_features_test_all,
            test_dataloader,
            optimizer,
            device,
            model,
            img_attention,
            k=2,
        )
        _, v4_acc, _ = evaluate_model(
            text_features_test_all,
            img_features_test_all,
            test_dataloader,
            optimizer,
            device,
            model,
            img_attention,
            k=4,
        )
        _, v10_acc, _ = evaluate_model(
            text_features_test_all,
            img_features_test_all,
            test_dataloader,
            optimizer,
            device,
            model,
            img_attention,
            k=10,
        )
        _, v50_acc, v50_top5_acc = evaluate_model(
            text_features_test_all,
            img_features_test_all,
            test_dataloader,
            optimizer,
            device,
            model,
            img_attention,
            k=50,
        )
        _, v100_acc, v100_top5_acc = evaluate_model(
            text_features_test_all,
            img_features_test_all,
            test_dataloader,
            optimizer,
            device,
            model,
            img_attention,
            k=100,
        )

        msg = (f"Epoch {epoch + 1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, "
              f"Top5 Accuracy: {top5_acc:.4f}")
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

        msg = (f"Epoch {epoch + 1}/{epochs} - "
              f"v2 Accuracy: {v2_acc:.4f} - v4 Accuracy: {v4_acc:.4f} - "
              f"v10 Accuracy: {v10_acc:.4f} - v50 Accuracy: {v50_acc:.4f} - "
              f"v100 Accuracy: {v100_acc:.4f}")
        
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")


def main():
    """Main function to parse arguments and run training."""
    data_path = '/mnt/dataset0/ldy/datasets/THINGS_EEG/Preprocessed_data_250Hz'
    lr = 3e-4
    epochs = 500
    batch_size = 256
    sub = 'sub-01'
    subjects = [f'sub-{i:02d}' for i in range(1, 11)]

    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    train_dataset = EEGDataset(data_path, adap_subject=sub, subjects=[sub], train=True)
    test_dataset = EEGDataset(data_path, adap_subject=sub, subjects=[sub], train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)

    text_features_train_all = train_dataset.text_features
    text_features_test_all = test_dataset.text_features
    img_features_train_all = train_dataset.img_features
    img_features_test_all = test_dataset.img_features

    feature_dim = img_features_train_all.shape[-1]
    img_attention = ImageFeatureAttention(embed_dim=feature_dim)

    model = eeg_encoder()
    optimizer = AdamW(itertools.chain(model.parameters(), img_attention.parameters()), lr=lr)

    results = main_train_loop(
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
    )

main()