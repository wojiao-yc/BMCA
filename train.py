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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from data_preparing.eegdatasets_joint_subjects import EEGDataset

from model.custom_pipeline import *
from model.FusedEEGViT import FusedEEGViT

def set_trainable_parameters(model: FusedEEGViT) -> FusedEEGViT:
    """只训练 bridge + gate + brain adapter，其余保持冻结。"""

    # visual 在 __init__ 里已经冻结

    # 启用 bridges、gate_head、final_norm 的参数
    for m in [model.bridges, model.gate_head, model.final_norm]:
        for p in m.parameters():
            p.requires_grad = True

    # Brain 侧：根据需要选择性冻结/解冻
    for name, p in model.brain_adapter.named_parameters():
        p.requires_grad = True

    return model


def train_one_step(text_features_all, img_features_all, dataloader, optimizer=None, device=None, model=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = FusedEEGViT()

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # _, _, preprocess = open_clip.create_model_and_transforms(
    #     model_name="ViT-H-14",
    #     pretrained="laion2b_s32b_b79k",
    # )
    
    model = set_trainable_parameters(model)
    model.to(device)
    model.train()
    text_features_all = text_features_all.to(device).float()
    img_features_all = (img_features_all[::10]).to(device).float()

    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.99  # Weight for image vs text loss
    features_list = []  # For storing features if needed
    
    for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
        print("batch_idx:", batch_idx)
        # Move data to device
        eeg_data = eeg_data.to(device)
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)
        
        # pixel_values = preprocess(img).unsqueeze(0).to(device)
        pixel_values = img.to(device)

        optimizer.zero_grad()
        
        # Prepare subject IDs
        batch_size = eeg_data.size(0)
        
        # Forward pass
        out = model(pixel_values, eeg_data)
        eeg_features = out["z_gate"].float()
        features_list.append(eeg_features)
        logit_scale = model.logit_scale
        
        # Compute losses
        img_loss = model.loss_func(eeg_features, img_features, logit_scale)
        text_loss = model.loss_func(eeg_features, text_features, logit_scale)
        loss = alpha * img_loss + (1 - alpha) * text_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics calculation
        total_loss += loss.item()
        logits_img = logit_scale * eeg_features @ img_features_all.T
        logits_single = logits_img
        predicted = torch.argmax(logits_single, dim=1)
        
        batch_size = predicted.shape[0]
        total += batch_size
        correct += (predicted == labels).sum().item()
        
        # Clean up
        del eeg_data, labels, text, text_features, img, img_features
        
    # Compute epoch metrics
    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total
    return average_loss, accuracy, torch.cat(features_list, dim=0)

    
def evaluate_model(text_features_all, img_features_all, dataloader, device=None, model=None, k=None):
    """Evaluation function with k-way classification"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = FusedEEGViT().to(device)
        
    model.eval()
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.99
    top5_correct_count = 0
    
    # Get all unique categories
    all_labels = set(range(text_features_all.size(0)))
    
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
            # Move data to device
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            
            # Prepare subject IDs
            batch_size = eeg_data.size(0)
            pixel_values = img
            # Forward pass
            out = model(pixel_values, eeg_data)
            eeg_features = out["brain_tokens"]
        
            logit_scale = model.logit_scale
            
            # Compute losses
            img_loss = model.loss_func(eeg_features, img_features, logit_scale)
            text_loss = model.loss_func(eeg_features, text_features, logit_scale)
            loss = img_loss * alpha + text_loss * (1 - alpha)
            total_loss += loss.item()
            
            # k-way classification evaluation
            for idx, label in enumerate(labels):
                # Select k-1 negative classes plus the correct one
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]
                
                if k == 200:
                    # Full evaluation (200-way)
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    
                    if predicted_label == label.item():
                        correct += 1
                    
                    # Top-5 accuracy calculation
                    _, top5_indices = torch.topk(logits_single, 5, largest=True)
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:
                        top5_correct_count += 1
                    total += 1
                
                elif k in [50, 100]:
                    # Medium evaluation (50 or 100-way)
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    
                    if predicted_label == label.item():
                        correct += 1
                    
                    # Top-5 accuracy calculation
                    _, top5_indices = torch.topk(logits_single, 5, largest=True)
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:
                        top5_correct_count += 1
                    total += 1
                
                elif k in [2, 4, 10]:
                    # Small evaluation (2, 4 or 10-way)
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    
                    if predicted_label == label.item():
                        correct += 1
                    total += 1
                
                else:
                    print("Error: Invalid k value")
            
            # Clean up
            del eeg_data, labels, text, text_features, img, img_features
    
    # Compute metrics
    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total
    top5_acc = top5_correct_count / total if k in [50, 100, 200] else 0
    return average_loss, accuracy, top5_acc 



def main_train_loop(epochs, model, train_dataloader, test_dataloader, optimizer, 
                    device, text_features_train_all, text_features_test_all, 
                    img_features_train_all, img_features_test_all):
    
    log_file = "log.txt"

    for epoch in range(epochs):
        # Training phase
        train_loss, train_accuracy, features_tensor = train_one_step(
            text_features_train_all, img_features_train_all, train_dataloader, optimizer, device, model
        )

        test_loss, test_accuracy, top5_acc = evaluate_model(
            text_features_test_all, img_features_test_all, test_dataloader, device, model, k=200
        )

        _, v2_acc, _ = evaluate_model(text_features_test_all, img_features_test_all, test_dataloader, device, model, k=2)
        _, v4_acc, _ = evaluate_model(text_features_test_all, img_features_test_all, test_dataloader, device, model, k=4)
        _, v10_acc, _ = evaluate_model(text_features_test_all, img_features_test_all, test_dataloader, device, model, k=10)
        _, v50_acc, v50_top5_acc = evaluate_model(text_features_test_all, img_features_test_all, test_dataloader, device, model, k=50)
        _, v100_acc, v100_top5_acc = evaluate_model(text_features_test_all, img_features_test_all, test_dataloader, device, model, k=100)
        

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
    """Main function to parse arguments and run training"""
    data_path = '/mnt/dataset0/ldy/datasets/THINGS_EEG/Preprocessed_data_250Hz'
    lr = 3e-4
    epochs = 50
    batch_size = 48
    sub = 'sub-01'
    subjects = [f'sub-{i:02d}' for i in range(1, 11)]


    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    train_dataset = EEGDataset(data_path, adap_subject=sub, subjects=[sub], train=True)
    test_dataset = EEGDataset(data_path, adap_subject=sub, subjects=[sub], train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)

    # Get features
    text_features_train_all = train_dataset.text_features
    text_features_test_all = test_dataset.text_features
    img_features_train_all = train_dataset.img_features
    img_features_test_all = test_dataset.img_features

    model = FusedEEGViT()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    epochs = 50
    
    results = main_train_loop(
        epochs, model, train_loader, test_loader, optimizer, 
        device, text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all
    )

if __name__ == "__main__":
    main()