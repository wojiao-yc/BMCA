import os
os.environ['http_proxy'] = 'http://127.0.0.1:7896'
os.environ['https_proxy'] = 'http://127.0.0.1:7896'
os.environ['all_proxy'] = 'socks5://127.0.0.1:7897'
os.sched_setaffinity(0, {38})
from typing import Tuple
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from PIL import Image

# 你自己的数据和模型
from data_preparing.blureegdatasets_selected_avg import EEGDataset
from model.EEG_MedformerTS import eeg_encoder
from model.uni import Projector, EEGConformer_Encoder, MetaEEG, EEGNet_Encoder, ShallowFBCSPNet_Encoder, NICE, ATCNet_Encoder, EEGITNet_Encoder, EEGProjectLayer

# ==========================
#   PyTorch Lightning 相关
# ==========================
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy


# ===========================================
#   LightningModule：封装你的训练 & 测试逻辑
# ===========================================
class EEGLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        text_features_train_all: torch.Tensor,
        img_features_train_all: torch.Tensor,
        text_features_test_all: torch.Tensor,
        img_features_test_all: torch.Tensor,
        lr: float = 3e-4,
    ):
        """
        model: 你的 eeg_encoder() 实例
        *_all: 从 dataset 中取出的全局特征库（不修改 dataset）
        """
        super().__init__()

        self.brain = model  # 保留原模型
        self.lr = lr

        # 注册全局特征库为 buffer，这样 Lightning 会自动把它们搬到 GPU 上
        self.register_buffer(
            "text_features_train_all",
            text_features_train_all.float(),
            persistent=False,
        )
        self.register_buffer(
            "img_features_train_all",
            img_features_train_all.float(),
            persistent=False,
        )
        self.register_buffer(
            "text_features_test_all",
            text_features_test_all.float(),
            persistent=False,
        )
        self.register_buffer(
            "img_features_test_all",
            img_features_test_all.float(),
            persistent=False,
        )

    def forward(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        前向：只输出 eeg 特征，和你原来 model(eeg_data) 一样
        """
        return self.brain(eeg_data)

    # ------------------------------
    # 训练步：等价于原来的 train_one_step
    # ------------------------------
    def training_step(self, batch: Tuple, batch_idx: int):
        # 保持原来的 batch 解包方式（不改 dataset）
        eeg_data, labels, text, text_features, img, img_features = batch

        # Lightning 会自动搬到 GPU，这里只要转 dtype
        eeg_data = eeg_data.float()
        img_features = img_features.float()
        labels = labels.long()

        # 前向
        eeg_features = self.brain(eeg_data)
        logit_scale = self.brain.logit_scale

        # 使用你原来的损失函数：loss_func(img_features, eeg_features, logit_scale)
        logit_scale = self.brain.softplus(logit_scale)

        eeg_loss, img_loss, logits_per_image= self.brain.loss_func_blur(img_features, eeg_features, logit_scale)
        eeg_alignment_loss = (eeg_loss.mean() + img_loss.mean()) / 2
        loss = eeg_alignment_loss

        # 记录训练 loss
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=eeg_data.size(0),
        )

        # ========= 训练精度（用全局训练集图像特征库）=========
        with torch.no_grad():
            # 原代码里有 img_features_all[::10]，这里为了简单，直接用全部特征库
            img_bank = self.img_features_train_all  # [N_img, D]
            logits_img = logit_scale * (eeg_features @ img_bank.T)  # [B, N_img]

            # Top-1
            top1_pred = torch.argmax(logits_img, dim=1)
            top1_acc = (top1_pred == labels).float().mean()

            # Top-5
            top5 = torch.topk(logits_img, k=min(5, logits_img.size(1)), dim=1).indices
            top5_correct = (top5 == labels.unsqueeze(1)).any(dim=1).float().mean()

            self.log(
                "train_top1_acc",
                top1_acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=eeg_data.size(0),
            )
            self.log(
                "train_top5_acc",
                top5_correct,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=eeg_data.size(0),
            )

        return loss

    # ------------------------------
    # 验证步：相当于你原来的 evaluate_model（k 比较大时的 top1/top5）
    # ------------------------------
    def validation_step(self, batch: Tuple, batch_idx: int):
        eeg_data, labels, text, text_features, img, img_features = batch

        eeg_data = eeg_data.float()
        img_features = img_features.float()
        labels = labels.long()

        eeg_features = self.brain(eeg_data)
        logit_scale = self.brain.logit_scale

        # 验证也用同样的 loss 定义
        eeg_alignment_loss = self.brain.loss_func(img_features, eeg_features, logit_scale)
        loss = eeg_alignment_loss

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=eeg_data.size(0),
        )

        # ========= 验证精度（用 test 全局特征库）=========
        with torch.no_grad():
            img_bank = self.img_features_test_all  # [N_img, D]
            logits_img = logit_scale * (eeg_features @ img_bank.T)  # [B, N_img]

            top1_pred = torch.argmax(logits_img, dim=1)
            top1_acc = (top1_pred == labels).float().mean()

            top5 = torch.topk(logits_img, k=min(5, logits_img.size(1)), dim=1).indices
            top5_correct = (top5 == labels.unsqueeze(1)).any(dim=1).float().mean()

            self.log(
                "val_top1_acc",
                top1_acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=eeg_data.size(0),
            )
            self.log(
                "val_top5_acc",
                top5_correct,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=eeg_data.size(0),
            )

        return loss

    # ------------------------------
    # 测试步：和验证类似，单独记录 test_*
    # ------------------------------
    def test_step(self, batch: Tuple, batch_idx: int):
        eeg_data, labels, text, text_features, img, img_features = batch

        eeg_data = eeg_data.float()
        img_features = img_features.float()
        labels = labels.long()

        eeg_features = self.brain(eeg_data)
        logit_scale = self.brain.logit_scale

        eeg_alignment_loss = self.brain.loss_func(img_features, eeg_features, logit_scale)
        loss = eeg_alignment_loss

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=eeg_data.size(0),
        )

        with torch.no_grad():
            img_bank = self.img_features_test_all
            logits_img = logit_scale * (eeg_features @ img_bank.T)

            top1_pred = torch.argmax(logits_img, dim=1)
            top1_acc = (top1_pred == labels).float().mean()

            top5 = torch.topk(logits_img, k=min(5, logits_img.size(1)), dim=1).indices
            top5_correct = (top5 == labels.unsqueeze(1)).any(dim=1).float().mean()

            self.log(
                "test_top1_acc",
                top1_acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=eeg_data.size(0),
            )
            self.log(
                "test_top5_acc",
                top5_correct,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=eeg_data.size(0),
            )

        return loss

    # ------------------------------
    # 优化器
    # ------------------------------
    def configure_optimizers(self):
        optimizer = AdamW(self.brain.parameters(), lr=self.lr)
        return optimizer


# ===========================================
#   训练主逻辑：类似你原来的 run_all_modes
# ===========================================
def run_all_modes(
    data_path: str,
    save_root: str,
    lr: float = 1e-4,
    epochs: int = 100,
    batch_size: int = 1024,
):
    os.makedirs(save_root, exist_ok=True)

    subjects = [f"sub-{i:02d}" for i in range(1, 11)]
    # subjects = [f"sub-{i:02d}" for i in range(2, 3)]
    # 只用你原来正在用的 inter-subject 模式
    modes = {
        1: "in-subject",
        2: "joint-subject",
        # 3: "inter-subject",
    }

    for mode_id, mode_name in modes.items():
        print(f"\n========== 运行模式: {mode_name} ==========\n")

        if mode_id == 2:  # joint-subject 只训练一次
            test_subjects = ["sub-01"]
        else:
            test_subjects = subjects

        for test_sub in test_subjects:
            if mode_id == 1:  # in-subject
                train_subjects = [test_sub]
                eval_subjects = [test_sub]
            elif mode_id == 2:  # joint-subject
                train_subjects = subjects
                eval_subjects = [test_sub]
            elif mode_id == 3:  # inter-subject
                train_subjects = [s for s in subjects if s != test_sub]
                eval_subjects = [test_sub]

            print(f"[{mode_name}] train={train_subjects}  test={test_sub}")

            exp_name = f"{mode_name}"
            log_dir = os.path.join(save_root, exp_name, str(test_sub))
            results_path = os.path.join(log_dir, "test_results.json")
            if os.path.exists(results_path):
                print(f"[{mode_name}] {test_sub} 已完成，跳过：{results_path}")
                continue

            train_dataset = EEGDataset(
                data_path,
                subjects=train_subjects,
                train=True,
                avg=True,
            )
            test_dataset = EEGDataset(
                data_path,
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

            # base_model = eeg_encoder()
            # base_model = Projector()
            base_model = NICE()
            # base_model = EEGProjectLayer()
            # base_model = MetaEEG()
            # base_model = EEGNet_Encoder()
            # base_model = EEGConformer_Encoder()

            pl_model = EEGLightningModule(
                model=base_model,
                text_features_train_all=train_dataset.text_features,
                img_features_train_all=train_dataset.img_features,
                text_features_test_all=test_dataset.text_features,
                img_features_test_all=test_dataset.img_features,
                lr=lr,
            )

            # ===== Logger，每个被试一个 log 目录 =====
            exp_name = f"{mode_name}"
            logger = TensorBoardLogger(
                save_dir=save_root,
                name=exp_name,
                version=test_sub,  # 每个被试一个 version
            )
            os.makedirs(logger.log_dir, exist_ok=True)
            print(f"TensorBoard log dir: {logger.log_dir}")

            # ===== Checkpoint & EarlyStopping =====
            checkpoint_callback = ModelCheckpoint(
                save_last=True,
                monitor="val_top1_acc",
                mode="max",
                filename="{epoch:02d}-{val_top1_acc:.4f}",
            )

            # 模式是 inter-subject，对标你示例里使用 val_top1_acc 早停
            early_stop_callback = EarlyStopping(
                monitor="val_top1_acc",
                min_delta=0.5,
                patience=100,
                verbose=False,
                mode="max",
            )

            # ===== Trainer（使用你示例中的训练策略）=====
            trainer = Trainer(
                devices=[7],  # 指定 GPU 卡号
                log_every_n_steps=10,
                # strategy=DDPStrategy(find_unused_parameters=True),
                strategy="auto",
                # callbacks=[early_stop_callback, checkpoint_callback],
                callbacks=[checkpoint_callback],
                max_epochs=epochs,
                accelerator="cuda" if torch.cuda.is_available() else "cpu",
                logger=logger,
            )

            # ===== 训练 =====
            trainer.fit(
                pl_model,
                train_dataloaders=train_loader,
                val_dataloaders=test_loader,  # 没有单独 val，就用 test 做验证
                ckpt_path="last",  
            )

            best = checkpoint_callback.best_model_path
            test_results = trainer.test(
                pl_model,
                dataloaders=test_loader,
                ckpt_path=best,
            )

            # 保存测试结果
            with open(os.path.join(logger.log_dir, "test_results.json"), "w") as f:
                json.dump(test_results, f, indent=4, default=lambda o: float(o))
            print(f"==== joint-subject | {test_sub} 测试完成 ====")
            print(f"==== {mode_name} | {test_sub} 测试完成，结果已写入 test_results.json ====")

        print(f"==== {mode_name} 模式已完成 ====\n")

    print("======= 所有模式训练已完成 =======\n")


# ===========================================
#   main：简单 argparse 包一下
# ===========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/dataset0/ldy/datasets/THINGS_EEG/Preprocessed_data_250Hz",
        help="EEG 数据路径",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="/home/wenxiao/workspace/qhy/BMCA/data/contrast/pipline/NICE",
        help="日志和 checkpoint 保存根目录",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="学习率",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="最大训练 epoch 数",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="训练 batch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )

    opt = parser.parse_args()

    seed_everything(opt.seed)

    run_all_modes(
        data_path=opt.data_path,
        save_root=opt.save_root,
        lr=opt.lr,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
    )


if __name__ == "__main__":
    main()
