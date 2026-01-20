import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import random
from torch.utils.data import DataLoader
# from data_preparing.blureegdatasets import EEGDataset
# from data_preparing.blureegdatasets_selected_avg import EEGDataset
from data_preparing.blureegdatasets_selected_quality_avg import EEGDataset
from model.EEG_MedformerTS import eeg_encoder
from model.uni import Projector, EEGConformer_Encoder, MetaEEG, EEGNet_Encoder, ShallowFBCSPNet_Encoder, NICE, ATCNet_Encoder, EEGITNet_Encoder, EEGProjectLayer
from BlurMed_PL import EEGLightningModule
import numpy as np

model_path = "/home/wenxiao/workspace/qhy/BMCA/data/blur+avgs+med/joint-subject/checkpoints/epoch=136-val_top1_acc=0.4950.ckpt"
data_path = '/mnt/dataset0/ldy/datasets/THINGS_EEG/Preprocessed_data_250Hz'
subjects = [f'sub-{i:02d}' for i in range(1, 11)]
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

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

def evaluate_subject(sub):
    test_dataset = EEGDataset(data_path, subjects=[sub], train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    text_features = test_dataset.text_features
    img_features = test_dataset.img_features

    loss, acc, top5 = evaluate_model(
        text_features,
        img_features,
        test_loader,
        optimizer=None,
        device=device,
        model=model,
        k=200
    )

    _, v2, _ = evaluate_model(text_features, img_features, test_loader, None, device, model, k=2)
    _, v4, _ = evaluate_model(text_features, img_features, test_loader, None, device, model, k=4)
    _, v10, _ = evaluate_model(text_features, img_features, test_loader, None, device, model, k=10)
    _, v50, _ = evaluate_model(text_features, img_features, test_loader, None, device, model, k=50)
    _, v100, _ = evaluate_model(text_features, img_features, test_loader, None, device, model, k=100)

    return [acc, top5, v2, v4, v10, v50, v100]


# ------------- main -------------
ckpt = torch.load(model_path, map_location=device)
state = {k.replace("brain.", ""): v for k, v in ckpt["state_dict"].items()}
model = eeg_encoder().to(device)
# # model = NICE().to(device)
# model.load_state_dict(state, strict=True)
# model = eeg_encoder().to(device)
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.eval()

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
