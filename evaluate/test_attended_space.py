import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from data_preparing.blureegdatasets import EEGDataset
from model.EEG_MedformerTS import eeg_encoder
from BlurMed import ImageFeatureAttention      # ⚠️ 如果 ImageFeatureAttention 在其他地方，请改路径

# ===================== 路径和配置 ===================== #
model_path = "/home/wenxiao/workspace/qhy/BMCA/data/joint-space/joint-subject/sub-01/epoch60.pth"
data_path = "/mnt/dataset0/ldy/datasets/THINGS_EEG/Preprocessed_data_250Hz"
subjects = [f"sub-{i:02d}" for i in range(1, 11)]
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")


# =====================================================
#  评估函数（与你给的一致）
# =====================================================
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
#  加载模型+注意力模块权重
# =====================================================
print("Loading weights =", model_path)

# 初始化模型
model = eeg_encoder().to(device)

# ⚠️ 必须先加载训练集中的 img_attention 维度
checkpoint = torch.load(model_path, map_location=device)

# 推断注意力维度
feature_dim = checkpoint["img_attention_state_dict"]["attn.in_proj_weight"].shape[-1]
img_attention = ImageFeatureAttention(embed_dim=feature_dim).to(device)

# Load weights
model.load_state_dict(checkpoint["model_state_dict"])
img_attention.load_state_dict(checkpoint["img_attention_state_dict"])

model.eval()
img_attention.eval()

print("✔ Model + Attention restored!")


# =====================================================
# 评估一个 subject
# =====================================================
def evaluate_subject(sub):
    test_dataset = EEGDataset(data_path, adap_subject=sub, subjects=[sub], train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    text_features = test_dataset.text_features
    img_features = test_dataset.img_features

    _, _, _, acc, top5 = evaluate_model(text_features, img_features, test_loader,
                                     optimizer=None, device=device, model=model, img_attention=img_attention, k=200)

    _, _, _, v2, _ = evaluate_model(text_features, img_features, test_loader, None, device, model, img_attention,k=2)
    _, _, _, v4, _ = evaluate_model(text_features, img_features, test_loader, None, device, model, img_attention,k=4)
    _, _, _, v10, _ = evaluate_model(text_features, img_features, test_loader, None, device, model, img_attention,k=10)
    _, _, _, v50, _ = evaluate_model(text_features, img_features, test_loader, None, device, model, img_attention,k=50)
    _, _, _, v100, _ = evaluate_model(text_features, img_features, test_loader, None, device, model, img_attention,k=100)

    return [acc, top5, v2, v4, v10, v50, v100]


# =====================================================
#  主程序
# =====================================================
results = []

for sub in subjects:
    print(f"Evaluating {sub} ...")
    metrics = evaluate_subject(sub)
    results.append(metrics)
    print(f"{sub}: {metrics}")

results = np.array(results)
mean_metrics = results.mean(axis=0)

print("\n====== Summary (Averages over 10 subjects) ======")
print("Accuracy     :", mean_metrics[0])
print("Top-5        :", mean_metrics[1])
print("v2 Accuracy  :", mean_metrics[2])
print("v4 Accuracy  :", mean_metrics[3])
print("v10 Accuracy :", mean_metrics[4])
print("v50 Accuracy :", mean_metrics[5])
print("v100 Accuracy:", mean_metrics[6])
