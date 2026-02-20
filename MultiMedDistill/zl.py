import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import os
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from scipy import stats
import cv2
from PIL import Image


# ==================== 配置管理 ====================
class Config:
    def __init__(self):
        # 基础配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = 42

        # 模型配置
        self.backbone = 'efficientnet_b0'  # 或 'efficientnet_b3'
        self.feat_dim = 1280  # EfficientNet-B0 的特征维度
        self.segmentation_tasks = {
            'busi': {'type': 'binary', 'classes': 2},
            'kvasir': {'type': 'binary', 'classes': 2},
            'covid': {'type': 'binary', 'classes': 2},
            'msd_heart': {'type': 'multi', 'classes': 4}
        }

        # 分类任务配置
        self.classification_tasks = {
            'aptos': {'classes': 5}  # 糖尿病视网膜病变分级
        }

        # 识别任务配置
        self.recognition_tasks = {
            'isic': {'classes': 7}  # 皮肤病变识别
        }

        # 训练配置
        self.batch_size = 32
        self.num_epochs = 100
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.warmup_epochs = 5

        # 损失权重初始化
        self.alpha_task = 1.0
        self.alpha_feat = 0.8
        self.alpha_out = 0.6
        self.alpha_rec = 0.4
        self.alpha_crd = 0.3

        # CRD 配置
        self.crd_temperature = 0.07
        self.crd_momentum = 0.5
        self.memory_bank_size = 16384

        # 数据路径
        self.data_root = './data'
        self.checkpoint_dir = './checkpoints'
        self.log_dir = './logs'
        self.vis_dir = './visualizations'

        # 创建必要目录
        for dir_path in [self.checkpoint_dir, self.log_dir, self.vis_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# ==================== 日志配置 ====================
def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ==================== 数据集定义 ====================
class MedicalDataset(Dataset):
    """单个医疗数据集的基类"""

    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        """子类需要实现此方法"""
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class MultiTaskDataset(Dataset):
    """多任务联合数据集"""

    def __init__(self, datasets_dict: Dict[str, Dataset]):
        self.datasets = list(datasets_dict.values())
        self.task_names = list(datasets_dict.keys())
        self.task_to_id = {name: i for i, name in enumerate(self.task_names)}

        # 计算每个数据集的长度和偏移量
        self.lengths = [len(d) for d in self.datasets]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)
        self.total_length = sum(self.lengths)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # 确定当前索引属于哪个任务
        task_id = np.searchsorted(self.cumulative_lengths[1:], idx, side='right')
        local_idx = idx - self.cumulative_lengths[task_id]

        image, label = self.datasets[task_id][local_idx]

        return {
            'image': image,
            'label': label,
            'task_id': torch.tensor(task_id, dtype=torch.long),
            'task_name': self.task_names[task_id]
        }


# ==================== 学生模型 ====================
class StudentModel(nn.Module):
    def __init__(self, backbone='efficientnet_b0', num_classes_dict=None):
        super().__init__()

        if backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=True)
            # 注册多层特征钩子
            self.feature_layers = [
                self.backbone.features[2],  # Layer 1
                self.backbone.features[4],  # Layer 2
                self.backbone.features[6],  # Layer 3
                self.backbone.features[8]  # Layer 4
            ]
            self.feat_dims = [24, 40, 80, 112]  # EfficientNet-B0 各层维度

        self.backbone.classifier = nn.Identity()

        # 任务头
        self.task_heads = nn.ModuleDict()
        for task_name, num_classes in num_classes_dict.items():
            self.task_heads[task_name] = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1280, num_classes)
            )

    def forward(self, x, task_name=None, return_multi_scale=False):
        """
        Args:
            x: [B, 3, H, W]
            task_name: str
            return_multi_scale: bool - 是否返回多尺度特征
        Returns:
            如果 return_multi_scale=True:
                multi_scale_features: List[[B, C_i, H_i, W_i]]
                final_features: [B, 1280]
                logits: [B, num_classes]
            否则:
                final_features: [B, 1280]
                logits: [B, num_classes]
        """
        multi_scale_features = []

        if return_multi_scale:
            # 提取多尺度特征
            x_temp = x
            for i, layer in enumerate(self.backbone.features):
                x_temp = layer(x_temp)
                if layer in self.feature_layers:
                    multi_scale_features.append(x_temp)

            # 最终特征
            final_features = self.backbone.avgpool(x_temp)
            final_features = torch.flatten(final_features, 1)
        else:
            final_features = self.backbone(x)

        # 分类
        if task_name is not None:
            logits = self.task_heads[task_name](final_features)

            if return_multi_scale:
                return multi_scale_features, final_features, logits
            else:
                return final_features, logits

        return final_features


# ==================== Teacher Gating ====================
class TeacherGate(nn.Module):
    def __init__(self, feat_dim, num_teachers, num_tasks):
        super().__init__()
        self.num_teachers = num_teachers

        # 任务嵌入
        self.task_embedding = nn.Embedding(num_tasks, 128)

        # 学生特征编码器
        self.student_encoder = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # 门控网络（输入维度 = 128 + 128 + num_teachers）
        self.gate_net = nn.Sequential(
            nn.Linear(128 + 128 + num_teachers, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_teachers)
        )

    def forward(self, student_features, teacher_features_dict, task_ids):
        """
        Args:
            student_features: [B, feat_dim]
            teacher_features_dict: {teacher_name: [B, feat_dim]}
            task_ids: [B]
        Returns:
            gate_weights: [B, num_teachers]
        """
        # 1. 学生特征编码（公式2中的 s^(ℓ)）
        student_encoded = self.student_encoder(student_features)  # [B, 128]

        # 2. 任务嵌入（公式2中的 e_t）
        task_emb = self.task_embedding(task_ids)  # [B, 128]

        # 3. 计算学生-教师余弦相似度（公式2中的 r^(ℓ)）
        teacher_features_list = list(teacher_features_dict.values())
        teacher_features_stacked = torch.stack(teacher_features_list, dim=1)  # [B, num_teachers, feat_dim]

        # 归一化
        student_norm = F.normalize(student_features, dim=1)  # [B, feat_dim]
        teacher_norm = F.normalize(teacher_features_stacked, dim=2)  # [B, num_teachers, feat_dim]

        # 余弦相似度
        cosine_sim = torch.bmm(
            teacher_norm,
            student_norm.unsqueeze(2)
        ).squeeze(2)  # [B, num_teachers]

        # 4. 拼接所有特征（公式2）
        combined = torch.cat([student_encoded, task_emb, cosine_sim], dim=1)  # [B, 128+128+num_teachers]

        # 5. 计算门控权重（公式3）
        gate_logits = self.gate_net(combined)
        gate_weights = F.softmax(gate_logits, dim=1)

        return gate_weights


# ==================== Return Decoder ====================
class ReturnDecoder(nn.Module):


    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Forward Projection (公式8)
        self.forward_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        # Inverse Projection (公式9) - 镜像对称
        self.inverse_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, teacher_features):
        """
        Args:
            teacher_features: [B, C_in, H, W]
        Returns:
            aligned_features: [B, C_out, H, W]
            reconstructed_features: [B, C_in, H, W]
        """
        aligned = self.forward_proj(teacher_features)
        reconstructed = self.inverse_proj(aligned)
        return aligned, reconstructed


# ==================== Loss Level Gating ====================
class LossLevelGating(nn.Module):
    def __init__(self, feat_dim, num_losses=5, temperature=2.0):
        super().__init__()
        self.temperature = temperature  # 论文中的 τ

        self.gate_net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_losses)
        )

    def forward(self, student_features):
        """
        Args:
            student_features: [B, feat_dim]
        Returns:
            loss_weights: [B, num_losses] - 温度缩放的权重
        """
        logits = self.gate_net(student_features)

        # 应用温度缩放（公式14）
        scaled_logits = logits / self.temperature
        weights = F.softmax(scaled_logits, dim=1)

        return weights


# ==================== CRD Loss (对比表示蒸馏) ====================
class CRDLoss(nn.Module):
    """Contrastive Representation Distillation Loss"""

    def __init__(self, feat_dim, num_samples, temperature=0.07, momentum=0.5):
        super().__init__()
        self.temperature = temperature
        self.momentum = momentum

        # Memory Bank (队列)
        self.register_buffer('memory_bank', torch.randn(num_samples, feat_dim))
        self.memory_bank = F.normalize(self.memory_bank, dim=1)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        self.num_samples = num_samples

    @torch.no_grad()
    def _update_memory_bank(self, features):
        """更新 Memory Bank"""
        batch_size = features.shape[0]
        ptr = int(self.queue_ptr)

        # 循环队列
        if ptr + batch_size <= self.num_samples:
            self.memory_bank[ptr:ptr + batch_size] = features
            ptr = (ptr + batch_size) % self.num_samples
        else:
            remaining = self.num_samples - ptr
            self.memory_bank[ptr:] = features[:remaining]
            self.memory_bank[:batch_size - remaining] = features[remaining:]
            ptr = batch_size - remaining

        self.queue_ptr[0] = ptr

    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: [B, feat_dim]
            teacher_features: [B, feat_dim]
        """
        # 归一化
        student_features = F.normalize(student_features, dim=1)
        teacher_features = F.normalize(teacher_features, dim=1)

        # 正样本：学生-教师对
        pos_logits = torch.sum(student_features * teacher_features, dim=1, keepdim=True) / self.temperature

        # 负样本：学生 vs Memory Bank
        neg_logits = torch.mm(student_features, self.memory_bank.t()) / self.temperature

        # InfoNCE Loss
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)

        # 更新 Memory Bank
        self._update_memory_bank(teacher_features.detach())

        return loss


# ==================== 重构损失 ====================
class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 论文中的权重配置
        self.lambda_mse = 0.5
        self.lambda_l1 = 0.3
        self.lambda_cos = 0.2

    def forward(self, reconstructed, original):
        mse_loss = F.mse_loss(reconstructed, original)
        l1_loss = F.l1_loss(reconstructed, original)

        # 余弦相似度（需要展平特征）
        recon_flat = reconstructed.view(reconstructed.size(0), -1)
        orig_flat = original.view(original.size(0), -1)
        cos_sim = F.cosine_similarity(recon_flat, orig_flat, dim=1).mean()
        cos_loss = 1 - cos_sim


        total_loss = (
                self.lambda_mse * mse_loss +
                self.lambda_l1 * l1_loss +
                self.lambda_cos * cos_loss
        )

        return total_loss


# ==================== 特征蒸馏损失 ====================
class FeatureDistillationLoss(nn.Module):
    """特征级蒸馏损失"""

    def __init__(self):
        super().__init__()

    def forward(self, student_features, teacher_features_dict, gate_weights):
        """
        Args:
            student_features: [B, feat_dim]
            teacher_features_dict: {teacher_name: [B, feat_dim]}
            gate_weights: [B, num_teachers]
        """
        # 融合教师特征
        teacher_features_list = list(teacher_features_dict.values())
        teacher_features_stacked = torch.stack(teacher_features_list, dim=1)  # [B, num_teachers, feat_dim]

        # 加权融合
        fused_teacher_features = torch.sum(
            teacher_features_stacked * gate_weights.unsqueeze(-1),
            dim=1
        )  # [B, feat_dim]

        # MSE 损失
        loss = F.mse_loss(student_features, fused_teacher_features)

        return loss


# ==================== 输出蒸馏损失 ====================
class OutputDistillationLoss(nn.Module):
    def __init__(self, initial_temperature=4.0, final_temperature=1.0):
        super().__init__()
        self.initial_temp = initial_temperature
        self.final_temp = final_temperature
        self.current_temp = initial_temperature

    def update_temperature(self, epoch, total_epochs):
        """
        线性衰减温度（论文第6页）
        Args:
            epoch: 当前epoch
            total_epochs: 总epoch数
        """
        progress = epoch / total_epochs
        self.current_temp = (
                self.initial_temp -
                (self.initial_temp - self.final_temp) * progress
        )

    def forward(self, student_logits, teacher_logits_dict, gate_weights):
        """使用当前温度进行蒸馏"""
        # 融合教师输出
        teacher_logits_list = list(teacher_logits_dict.values())
        teacher_logits_stacked = torch.stack(teacher_logits_list, dim=1)

        fused_teacher_logits = torch.sum(
            teacher_logits_stacked * gate_weights.unsqueeze(-1),
            dim=1
        )

        # KL 散度（使用自适应温度）
        student_soft = F.log_softmax(student_logits / self.current_temp, dim=1)
        teacher_soft = F.softmax(fused_teacher_logits / self.current_temp, dim=1)

        loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (self.current_temp ** 2)

        return loss



class MultiMedDistillSystem(nn.Module):
    """多医疗任务蒸馏系统"""

    def __init__(self, config, teachers_dict):
        super().__init__()
        self.config = config


        self.student = StudentModel(
            backbone=config.backbone,
            num_classes_dict=config.num_classes_dict
        )


        self.teachers = nn.ModuleDict(teachers_dict)
        for teacher in self.teachers.values():
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False

        # ✅ Teacher-Level Gate
        self.teacher_gate = TeacherGate(
            feat_dim=config.feat_dim,
            num_teachers=len(teachers_dict),
            num_tasks=len(config.num_classes_dict)
        )

        # Return Decoder
        self.return_decoder = ReturnDecoder(
            in_channels=config.feat_dim,
            out_channels=config.feat_dim
        )

        # ✅ Loss-Level Gate
        self.loss_gate = LossLevelGating(
            feat_dim=config.feat_dim,
            num_losses=5,
            temperature=2.0
        )


        self.crd_loss = CRDLoss(
            feat_dim=config.feat_dim,
            num_samples=config.memory_bank_size,
            temperature=config.crd_temperature,
            momentum=config.crd_momentum
        )
        self.rec_loss = ReconstructionLoss()
        self.feat_loss = FeatureDistillationLoss()
        self.out_loss = OutputDistillationLoss()

    def forward(self, images, task_ids, task_name, labels=None):

        batch_size = images.size(0)


        stu_feats, stu_logits = self.student(images, task_name)


        tea_outputs = {}
        with torch.no_grad():
            for name, teacher in self.teachers.items():
                t_feat, t_logit = teacher(images, task_name)
                tea_outputs[name] = {'feat': t_feat, 'logit': t_logit}


        tea_feats_dict = {k: v['feat'] for k, v in tea_outputs.items()}
        tea_logits_dict = {k: v['logit'] for k, v in tea_outputs.items()}


        teacher_weights = self.teacher_gate(
            student_features=stu_feats,
            teacher_features_dict=tea_feats_dict,
            task_ids=task_ids
        )

        teacher_features_stacked = torch.stack(
            list(tea_feats_dict.values()),
            dim=1
        )  # [B, num_teachers, feat_dim]

        fused_teacher_features = torch.sum(
            teacher_features_stacked * teacher_weights.unsqueeze(-1),
            dim=1
        )  # [B, feat_dim]


        teacher_logits_stacked = torch.stack(
            list(tea_logits_dict.values()),
            dim=1
        )  # [B, num_teachers, num_classes]

        fused_teacher_logits = torch.sum(
            teacher_logits_stacked * teacher_weights.unsqueeze(-1),
            dim=1
        )  # [B, num_classes]


        stu_feats_2d = stu_feats.view(batch_size, -1, 1, 1)  # [B, C, 1, 1]
        fused_tea_feats_2d = fused_teacher_features.view(batch_size, -1, 1, 1)

        aligned_feats, reconstructed_feats = self.return_decoder(fused_tea_feats_2d)


        aligned_feats = aligned_feats.view(batch_size, -1)
        reconstructed_feats = reconstructed_feats.view(batch_size, -1)


        losses = {}


        if labels is not None:
            losses['task'] = F.cross_entropy(stu_logits, labels)
        else:
            losses['task'] = torch.tensor(0.0, device=images.device)


        losses['feat'] = F.mse_loss(aligned_feats, fused_teacher_features)


        student_soft = F.log_softmax(
            stu_logits / self.out_loss.current_temp,
            dim=1
        )
        teacher_soft = F.softmax(
            fused_teacher_logits / self.out_loss.current_temp,
            dim=1
        )
        losses['out'] = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (self.out_loss.current_temp ** 2)


        losses['rec'] = self.rec_loss(
            reconstructed_feats,
            fused_teacher_features.detach()
        )


        losses['crd'] = self.crd_loss(aligned_feats, fused_teacher_features)


        loss_weights = self.loss_gate(stu_feats.detach())  # [B, 5]


        loss_components = torch.stack([
            losses['task'].expand(batch_size),
            losses['feat'].expand(batch_size),
            losses['out'].expand(batch_size),
            losses['rec'].expand(batch_size),
            losses['crd'].expand(batch_size)
        ], dim=1)  # [B, 5]


        weighted_losses = loss_components * loss_weights  # [B, 5]
        total_loss = weighted_losses.sum(dim=1).mean()  # 标量


        return (
            total_loss,
            losses,
            teacher_weights,
            loss_weights,
            stu_logits
        )



def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, logger):

    model.train()


    metrics = {
        'total_loss': [],
        'task_loss': [],
        'feat_loss': [],
        'out_loss': [],
        'rec_loss': [],
        'crd_loss': [],
        'loss_weights': []
    }

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):

        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        task_ids = batch['task_id'].to(device)
        task_name = batch['task_name'][0]  # 假设同一batch来自同一任务


        optimizer.zero_grad()
        total_loss, loss_dict, loss_weights, predictions = model(
            images, task_ids, task_name, labels
        )


        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


        metrics['total_loss'].append(total_loss.item())
        for k in ['task', 'feat', 'out', 'rec', 'crd']:
            metrics[f'{k}_loss'].append(loss_dict[k].item())
        metrics['loss_weights'].append(loss_weights.mean(0).detach().cpu().numpy())


        pbar.set_postfix({
            'Loss': f"{total_loss.item():.4f}",
            'LW': loss_weights.mean(0).detach().cpu().numpy().round(3)
        })


    if scheduler is not None:
        scheduler.step()


    avg_metrics = {k: np.mean(v) for k, v in metrics.items() if k != 'loss_weights'}
    avg_metrics['loss_weights'] = np.mean(metrics['loss_weights'], axis=0)

    return avg_metrics



@torch.no_grad()
def evaluate(model, dataloader, device, logger):
    """评估模型性能"""
    model.eval()

    all_predictions = []
    all_labels = []
    all_features = []
    task_metrics = {}

    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        task_ids = batch['task_id'].to(device)
        task_name = batch['task_name'][0]

        # 前向传播（仅评估模式）
        _, _, _, predictions = model(images, task_ids, task_name, labels)

        # 提取特征用于可视化
        features = model.student(images, task_name)[0]

        all_predictions.append(predictions.cpu())
        all_labels.append(labels.cpu())
        all_features.append(features.cpu())

    # 合并所有批次
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_features = torch.cat(all_features, dim=0)

    # 计算准确率
    pred_classes = torch.argmax(all_predictions, dim=1)
    accuracy = (pred_classes == all_labels).float().mean().item()

    logger.info(f"Evaluation Accuracy: {accuracy:.4f}")

    return {
        'accuracy': accuracy,
        'predictions': all_predictions.numpy(),
        'labels': all_labels.numpy(),
        'features': all_features.numpy()
    }


# ==================== 可视化工具 ====================
class Visualizer:
    """可视化工具类"""

    @staticmethod
    def plot_tsne(features, labels, save_path, title="t-SNE Visualization"):
        """绘制t-SNE特征分布图"""
        from sklearn.manifold import TSNE

        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)

        # 绘图
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            c=labels,
            cmap='tab10',
            alpha=0.6,
            s=20
        )
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    @staticmethod
    def plot_loss_curves(train_losses, val_losses, save_path):
        """绘制训练和验证损失曲线"""
        plt.figure(figsize=(12, 6))

        # 子图1：总损失
        plt.subplot(1, 2, 1)
        plt.plot(train_losses['total_loss'], label='Train')
        plt.plot(val_losses['total_loss'], label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Total Loss')
        plt.legend()
        plt.grid(True)

        # 子图2：各分量损失
        plt.subplot(1, 2, 2)
        for key in ['task_loss', 'feat_loss', 'out_loss', 'rec_loss', 'crd_loss']:
            plt.plot(train_losses[key], label=key, alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Component Losses')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    @staticmethod
    def plot_loss_weights_evolution(loss_weights_history, save_path):
        """绘制损失权重演化图"""
        plt.figure(figsize=(10, 6))

        loss_names = ['Task', 'Feature', 'Output', 'Reconstruction', 'CRD']
        loss_weights_array = np.array(loss_weights_history)

        for i, name in enumerate(loss_names):
            plt.plot(loss_weights_array[:, i], label=name, linewidth=2)

        plt.xlabel('Epoch')
        plt.ylabel('Loss Weight')
        plt.title('Dynamic Loss Weight Evolution')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    @staticmethod
    def apply_gradcam(model, image, target_layer, task_name, save_path):
        """应用Grad-CAM生成热力图"""
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image

            # 创建GradCAM对象
            cam = GradCAM(model=model.student.backbone, target_layers=[target_layer])

            # 生成热力图
            grayscale_cam = cam(input_tensor=image.unsqueeze(0))
            grayscale_cam = grayscale_cam[0, :]

            # 转换图像格式
            img_np = image.cpu().permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

            # 叠加热力图
            visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

            # 保存
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img_np)
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(visualization)
            plt.title('Grad-CAM')
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()

        except ImportError:
            print("Warning: pytorch-grad-cam not installed. Skipping Grad-CAM visualization.")

    @staticmethod
    def statistical_significance_test(baseline_scores, proposed_scores):
        """统计显著性检验（Wilcoxon符号秩检验）"""
        from scipy import stats

        statistic, p_value = stats.wilcoxon(baseline_scores, proposed_scores)

        print(f"Wilcoxon Test Results:")
        print(f"  Statistic: {statistic:.4f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Significant (p < 0.05): {p_value < 0.05}")

        return {'statistic': statistic, 'p_value': p_value}


# ==================== 具体数据集实现示例 ====================
class BUSIDataset(MedicalDataset):
    """BUSI乳腺超声数据集"""

    def _load_samples(self):
        samples = []
        data_dir = self.root_dir / self.split

        for class_name in ['benign', 'malignant', 'normal']:
            class_dir = data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.png'):
                    label = ['benign', 'malignant', 'normal'].index(class_name)
                    samples.append((str(img_path), label))

        return samples


class NIHChestXrayDataset(MedicalDataset):
    """NIH胸部X光数据集"""

    def _load_samples(self):
        samples = []
        # 简化实现：读取CSV文件获取标签
        csv_path = self.root_dir / f'{self.split}_list.csv'

        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)

            for _, row in df.iterrows():
                img_path = self.root_dir / 'images' / row['Image Index']
                # NIH是多标签分类，这里简化处理
                labels = row['Finding Labels'].split('|')
                # 将第一个标签作为主标签
                label = hash(labels[0]) % 14  # 简化的标签编码
                samples.append((str(img_path), label))

        return samples


# ==================== 数据加载器创建 ====================
def create_dataloaders(config):
    """创建训练和验证数据加载器"""

    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 创建各任务数据集
    train_datasets = {
        'busi': BUSIDataset(
            root_dir=Path(config.data_root) / 'BUSI',
            split='train',
            transform=train_transform
        ),
        'nih': NIHChestXrayDataset(
            root_dir=Path(config.data_root) / 'NIH',
            split='train',
            transform=train_transform
        ),
        # 可以继续添加 ISIC、Kvasir 等数据集
    }

    val_datasets = {
        'busi': BUSIDataset(
            root_dir=Path(config.data_root) / 'BUSI',
            split='val',
            transform=val_transform
        ),
        'nih': NIHChestXrayDataset(
            root_dir=Path(config.data_root) / 'NIH',
            split='val',
            transform=val_transform
        ),
    }

    # 创建多任务数据集
    train_dataset = MultiTaskDataset(train_datasets)
    val_dataset = MultiTaskDataset(val_datasets)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader


# ==================== 教师模型创建（示例） ====================
def create_teacher_models(config):
    """创建教师模型字典"""

    # 这里使用预训练的ResNet作为示例教师
    # 实际应用中应该加载真实的MedSAM、CLIP等模型
    class TeacherWrapper(nn.Module):
        def __init__(self, backbone, num_classes_dict):
            super().__init__()
            self.backbone = backbone
            self.backbone.fc = nn.Identity()

            self.task_heads = nn.ModuleDict()
            for task_name, num_classes in num_classes_dict.items():
                self.task_heads[task_name] = nn.Linear(2048, num_classes)

        def forward(self, x, task_name):
            feat = self.backbone(x)
            logit = self.task_heads[task_name](feat)
            return feat, logit

    teachers = {}

    # 教师1：MedSAM风格（使用ResNet50）
    teachers['medsam'] = TeacherWrapper(
        models.resnet50(pretrained=True),
        config.num_classes_dict
    )

    # 教师2：CLIP风格（使用ResNet50）
    teachers['clip'] = TeacherWrapper(
        models.resnet50(pretrained=True),
        config.num_classes_dict
    )

    return teachers


# ==================== 主训练函数 ====================
def main():
    # 1. 初始化配置
    config = Config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # 2. 设置日志
    logger = setup_logger(Path(config.log_dir) / 'training.log')
    logger.info("=" * 50)
    logger.info("Starting MultiMedDistill Training")
    logger.info("=" * 50)

    # 3. 创建数据加载器
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")

    # 4. 创建教师模型
    logger.info("Creating teacher models...")
    teachers = create_teacher_models(config)
    for name in teachers.keys():
        teachers[name] = teachers[name].to(config.device)

    # 5. 创建蒸馏系统
    logger.info("Creating distillation system...")
    model = MultiMedDistillSystem(config, teachers).to(config.device)

    # 6. 优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=1e-6
    )

    # 7. 训练循环
    best_accuracy = 0.0
    train_history = {k: [] for k in ['total_loss', 'task_loss', 'feat_loss',
                                     'out_loss', 'rec_loss', 'crd_loss']}
    val_history = {'total_loss': [], 'accuracy': []}
    loss_weights_history = []

    logger.info("Starting training...")
    for epoch in range(1, config.num_epochs + 1):
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            config.device, epoch, logger
        )

        # 记录训练指标
        for k in train_history.keys():
            train_history[k].append(train_metrics[k])
        loss_weights_history.append(train_metrics['loss_weights'])

        # 评估
        if epoch % 5 == 0 or epoch == config.num_epochs:
            val_results = evaluate(model, val_loader, config.device, logger)
            val_history['accuracy'].append(val_results['accuracy'])

            # 保存最佳模型
            if val_results['accuracy'] > best_accuracy:
                best_accuracy = val_results['accuracy']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_accuracy,
                }, Path(config.checkpoint_dir) / 'best_model.pth')
                logger.info(f"✓ New best model saved! Accuracy: {best_accuracy:.4f}")

            # 可视化（每10个epoch）
            if epoch % 10 == 0:
                vis = Visualizer()

                # t-SNE可视化
                vis.plot_tsne(
                    val_results['features'],
                    val_results['labels'],
                    Path(config.vis_dir) / f'tsne_epoch_{epoch}.png'
                )

                # 损失权重演化
                vis.plot_loss_weights_evolution(
                    loss_weights_history,
                    Path(config.vis_dir) / f'loss_weights_epoch_{epoch}.png'
                )

        # 日志输出
        logger.info(f"Epoch {epoch}/{config.num_epochs} - "
                    f"Train Loss: {train_metrics['total_loss']:.4f}, "
                    f"Val Acc: {val_history['accuracy'][-1] if val_history['accuracy'] else 'N/A'}")

    # 8. 保存最终模型
    torch.save(model.state_dict(), Path(config.checkpoint_dir) / 'final_model.pth')

    # 9. 绘制最终结果
    vis = Visualizer()
    vis.plot_loss_curves(
        train_history,
        val_history,
        Path(config.vis_dir) / 'training_curves.png'
    )

    # 10. 推理性能测试
    logger.info("\n" + "=" * 50)
    logger.info("Performance Benchmarking")
    logger.info("=" * 50)

    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(config.device)

    # 延迟测试
    import time
    warmup = 10
    test_runs = 100

    for _ in range(warmup):
        _ = model.student(dummy_input, 'busi')

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    for _ in range(test_runs):
        _ = model.student(dummy_input, 'busi')

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    avg_time = (time.time() - start_time) / test_runs * 1000

    logger.info(f"Average Inference Time: {avg_time:.2f} ms")

    # FLOPs计算
    try:
        from thop import profile
        flops, params = profile(model.student, inputs=(dummy_input, 'busi'))
        logger.info(f"FLOPs: {flops / 1e9:.2f} G")
        logger.info(f"Parameters: {params / 1e6:.2f} M")
    except ImportError:
        logger.warning("thop not installed. Skipping FLOPs calculation.")

    logger.info("=" * 50)
    logger.info("Training completed!")
    logger.info(f"Best Validation Accuracy: {best_accuracy:.4f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

