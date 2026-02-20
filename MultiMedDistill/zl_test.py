import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
import numpy as np
import os
import json
import time
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import pandas as pd
import cv2
from PIL import Image
from load_teacher_model import load_models
from datasets import DataIntegrationManager
import gc
import math
from typing import Dict, List, Tuple, Optional, Any

# ============================================
# 1. 教师模型包装器
# ============================================

class TeacherWrapper(ABC):
    """教师模型包装器基类，用于统一不同教师模型的接口"""

    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取多层特征"""
        pass

    @abstractmethod
    def get_task_output(self, x: torch.Tensor) -> torch.Tensor:
        """获取任务特定的输出"""
        pass

    @abstractmethod
    def get_feature_dims(self) -> Dict[str, int]:
        """获取各层特征的维度"""
        pass


# ============================================
# 2. 跨任务知识蒸馏组件
# ============================================

class CrossTeacherAttention(nn.Module):
    """
    跨教师空间注意力模块 (Spatial Cross-Attention)

    职责：
    1. 接收由 Gating 加权后的聚合教师特征。
    2. 通过 Query(学生) 与 Key(教师) 的匹配，计算空间对齐权重。
    3. 将教师的知识（Value）精确转移到学生对应的空间位置。
    """

    def __init__(self, student_dim, teacher_dim, key_dim=64):
        super().__init__()
        self.query_conv = nn.Conv2d(student_dim, key_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(teacher_dim, key_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(teacher_dim, student_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))  # 残差系数，初始为0确保训练初期稳定
        self.scale = key_dim ** -0.5

    def forward(self, s_feat, fused_t_feat):
        """
        Args:
            s_feat: 学生原始特征 [B, C_s, H, W]
            fused_t_feat: 已经根据 Gating 权重加权求和后的教师特征 [B, C_t, H, W]
        """
        B, C_s, H, W = s_feat.shape

        # 1. 生成 Query (来自学生)
        proj_query = self.query_conv(s_feat).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, K]

        # 2. 生成 Key 和 Value (来自融合后的教师)
        proj_key = self.key_conv(fused_t_feat).view(B, -1, H * W)  # [B, K, HW]
        proj_value = self.value_conv(fused_t_feat).view(B, -1, H * W)  # [B, C_s, HW]

        # 3. 计算空间注意力图 (Spatial Attention Map)
        # 能量矩阵表示学生位置 i 与教师位置 j 的相关性
        energy = torch.bmm(proj_query, proj_key) * self.scale  # [B, HW, HW]
        attention = F.softmax(energy, dim=-1)

        # 4. 知识传递
        # 将教师的 Value 按照注意力权重重新分布到学生空间
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # [B, C_s, HW]
        out = out.view(B, C_s, H, W)

        # 5. 残差连接：只学习“差额”知识
        return s_feat + self.gamma * out


class BidirectionalProjector(nn.Module):
    """双向投影器：学生->教师 和 教师->学生"""

    def __init__(self, student_dim, teacher_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (student_dim + teacher_dim) // 2

        # 学生特征 -> 教师特征空间
        self.student_to_teacher = nn.Sequential(
            nn.Linear(student_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, teacher_dim)
        )

        # 教师特征 -> 学生特征空间（用于重构）
        self.teacher_to_student = nn.Sequential(
            nn.Linear(teacher_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, student_dim)
        )

    def forward(self, student_feat, teacher_feat):
        """
        Args:
            student_feat: [B, student_dim] 或 [B, C, H, W]
            teacher_feat: [B, teacher_dim] 或 [B, C, H, W]
        Returns:
            aligned_feat: 对齐到教师空间的学生特征
            reconstructed_feat: 重构的学生特征
        """
        # 处理空间特征
        if len(student_feat.shape) == 4:
            B, C, H, W = student_feat.shape
            student_feat = F.adaptive_avg_pool2d(student_feat, 1).view(B, C)
        if len(teacher_feat.shape) == 4:
            B, C, H, W = teacher_feat.shape
            teacher_feat = F.adaptive_avg_pool2d(teacher_feat, 1).view(B, C)

        # 前向投影：学生 -> 教师空间
        aligned_feat = self.student_to_teacher(student_feat)

        # 反向投影：对齐特征 -> 学生空间（重构）
        reconstructed_feat = self.teacher_to_student(aligned_feat)

        return aligned_feat, reconstructed_feat


class AdaptiveMultiTeacherProjector(nn.Module):
    """为多个教师创建独立的投影器"""

    def __init__(self, student_dim, teacher_dims_dict):
        """
        Args:
            student_dim: 学生特征维度
            teacher_dims_dict: {'teacher_name': teacher_dim, ...}
        """
        super().__init__()
        self.teacher_names = list(teacher_dims_dict.keys())

        # 为每个教师创建独立的双向投影器
        self.projectors = nn.ModuleDict({
            name: BidirectionalProjector(student_dim, dim)
            for name, dim in teacher_dims_dict.items()
        })

    def forward(self, student_features, teacher_features_dict):
        """
        Args:
            student_features: dict of {layer_name: feature_tensor}
            teacher_features_dict: dict of {teacher_name: {layer_name: feature}}
        Returns:
            aligned_features: {teacher_name: {layer_name: aligned_feat}}
            reconstructed_features: {teacher_name: {layer_name: reconstructed_feat}}
        """
        aligned_features = {}
        reconstructed_features = {}

        for teacher_name in self.teacher_names:
            if teacher_name not in teacher_features_dict:
                continue

            aligned_features[teacher_name] = {}
            reconstructed_features[teacher_name] = {}

            teacher_feats = teacher_features_dict[teacher_name]

            for layer_name, student_feat in student_features.items():
                if layer_name not in teacher_feats:
                    continue

                teacher_feat = teacher_feats[layer_name]

                # 使用对应的投影器
                aligned, reconstructed = self.projectors[teacher_name](
                    student_feat, teacher_feat
                )

                aligned_features[teacher_name][layer_name] = aligned
                reconstructed_features[teacher_name][layer_name] = reconstructed

        return aligned_features, reconstructed_features


class ReconstructionLoss(nn.Module):
    """
    三组分重构损失：MSE + L1 + Cosine Similarity
    确保投影后的特征能够还原回教师特征的语义
    """

    def __init__(self, lambda_mse=0.5, lambda_l1=0.3, lambda_cos=0.2):  # ← 改这里
        super(ReconstructionLoss, self).__init__()
        self.lambda_mse = lambda_mse
        self.lambda_l1 = lambda_l1
        self.lambda_cos = lambda_cos

    def forward(self, reconstructed_feat, teacher_feat):
        """
        Args:
            reconstructed_feat: 逆向重构的特征 [B, C, H, W]
            teacher_feat: 原始教师特征 [B, C, H, W]
        Returns:
            total_loss: 加权后的总重构损失
        """
        # 1. MSE Loss：全局能量对齐
        loss_mse = F.mse_loss(reconstructed_feat, teacher_feat)

        # 2. L1 Loss：稀疏性与细节保持
        loss_l1 = F.l1_loss(reconstructed_feat, teacher_feat)

        # 3. Cosine Similarity Loss：语义方向对齐
        # 将空间维度展平 [B, C, H, W] -> [B, C*H*W]
        recon_flat = reconstructed_feat.flatten(1)
        teacher_flat = teacher_feat.flatten(1)

        # 计算余弦相似度 (范围 [-1, 1])
        cos_sim = F.cosine_similarity(recon_flat, teacher_flat, dim=1)
        # 转换为损失 (1 - similarity)，范围 [0, 2]
        loss_cos = (1 - cos_sim).mean()

        # 加权组合
        total_loss = (self.lambda_mse * loss_mse +
                      self.lambda_l1 * loss_l1 +
                      self.lambda_cos * loss_cos)

        return total_loss, {
            'mse': loss_mse.item(),
            'l1': loss_l1.item(),
            'cos': loss_cos.item()
        }


class TaskAdapter(nn.Module):
    """任务特定适配器"""

    def __init__(self, in_dim, out_dim, reduction=4):
        super().__init__()
        hidden_dim = in_dim // reduction
        self.adapter = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, 1)
        )

    def forward(self, x):
        return x + self.adapter(x)  # 残差连接


class CRDLoss(nn.Module):
    """
    ✅ 修复版 Contrastive Representation Distillation Loss
    使用 Memory Bank 存储负样本
    """

    def __init__(self, feat_dim=128, num_samples=16384, temperature=0.07, momentum=0.5):
        super().__init__()

        self.feat_dim = feat_dim
        self.num_samples = num_samples
        self.temperature = temperature
        self.momentum = momentum

        # ✅ Memory Bank（队列）
        self.register_buffer(
            'memory_bank',
            torch.randn(num_samples, feat_dim)
        )
        self.memory_bank = F.normalize(self.memory_bank, dim=1)

        # 队列指针
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        # 特征投影层
        self.student_proj = None
        self.teacher_proj = None

    def _init_projectors(self, student_dim, teacher_dim):
        """初始化投影层"""
        if student_dim != self.feat_dim:
            self.student_proj = nn.Sequential(
                nn.Linear(student_dim, self.feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_dim, self.feat_dim)
            ).to(self.memory_bank.device)

        if teacher_dim != self.feat_dim:
            self.teacher_proj = nn.Sequential(
                nn.Linear(teacher_dim, self.feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_dim, self.feat_dim)
            ).to(self.memory_bank.device)

    @torch.no_grad()
    def _update_memory_bank(self, features):
        """✅ 更新 Memory Bank（循环队列）"""
        batch_size = features.shape[0]
        ptr = int(self.queue_ptr)

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
        """✅ 修复版前向传播"""
        # 特征展平
        if student_features.dim() == 4:
            student_features = F.adaptive_avg_pool2d(student_features, (1, 1)).flatten(1)
        if teacher_features.dim() == 4:
            teacher_features = F.adaptive_avg_pool2d(teacher_features, (1, 1)).flatten(1)

        B = student_features.shape[0]

        # 初始化投影层
        if self.student_proj is None:
            self._init_projectors(
                student_features.shape[1],
                teacher_features.shape[1]
            )

        # 投影
        if self.student_proj is not None:
            student_features = self.student_proj(student_features)
        if self.teacher_proj is not None:
            teacher_features = self.teacher_proj(teacher_features)

        # L2 归一化
        student_features = F.normalize(student_features, dim=1)
        teacher_features = F.normalize(teacher_features, dim=1)

        # ✅ 正样本相似度
        pos_logits = torch.sum(
            student_features * teacher_features,
            dim=1,
            keepdim=True
        ) / self.temperature  # [B, 1]

        # ✅ 负样本相似度（学生 vs Memory Bank）
        neg_logits = torch.mm(
            student_features,
            self.memory_bank.t()
        ) / self.temperature  # [B, num_samples]

        # ✅ InfoNCE Loss
        logits = torch.cat([pos_logits, neg_logits], dim=1)  # [B, 1 + num_samples]
        labels = torch.zeros(B, dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)

        # 更新 Memory Bank
        self._update_memory_bank(teacher_features.detach())

        return loss


class TeacherLevelGating(nn.Module):
    """
    教师级门控网络 (Teacher-Level Gating Network)

    职责：
    1. 比较学生投影特征与教师原始特征的语义一致性（余弦相似度）。
    2. 结合当前任务上下文（Task Embedding）。
    3. 输出每个教师的重要性权重 [B, num_teachers]，和为 1。
    """

    def __init__(self, student_dim, num_teachers, num_tasks=4, ema_momentum=0.9):
        super().__init__()
        self.num_teachers = num_teachers
        self.ema_momentum = ema_momentum

        # 1. 任务嵌入：为不同数据集提供上下文信息
        self.task_embeddings = nn.Embedding(num_tasks, 64)

        # 2. 门控 MLP
        # 输入：学生全局特征 (student_dim) + 任务嵌入 (64) + 相似度向量 (num_teachers)
        gating_input_dim = student_dim + 64 + num_teachers

        self.gating_net = nn.Sequential(
            nn.Linear(gating_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_teachers)
        )

        # 3. EMA 权重缓冲区：平滑训练波动，不参与反向传播
        self.register_buffer(
            'ema_weights',
            torch.ones(num_teachers) / num_teachers
        )

    def compute_similarity(self, student_feat, teacher_feats_dict):
        """
        """
        B = student_feat.shape[0]

        # ✅ 全局平均池化
        student_global = F.adaptive_avg_pool2d(student_feat, (1, 1)).flatten(1)  # [B, C_s]
        student_norm = F.normalize(student_global, dim=1)

        # 收集教师特征
        teacher_globals = []
        for teacher_name in sorted(teacher_feats_dict.keys()):
            t_feat = teacher_feats_dict[teacher_name]
            t_global = F.adaptive_avg_pool2d(t_feat, (1, 1)).flatten(1)  # [B, C_t]
            t_norm = F.normalize(t_global, dim=1)
            teacher_globals.append(t_norm)

        # ✅ 计算余弦相似度
        cosine_sim = []
        for t_norm in teacher_globals:
            sim = F.cosine_similarity(student_norm, t_norm, dim=1)  # [B]
            cosine_sim.append(sim)

        cosine_sim = torch.stack(cosine_sim, dim=1)  # [B, num_teachers]

        return cosine_sim

    def forward(self, student_feat, teacher_feats_projected, teacher_feats_original, task_id, use_ema=True):

        B, C, H, W = student_feat.shape

        # 1. 提取学生全局上下文 [B, C]
        student_pooled = F.adaptive_avg_pool2d(student_feat, 1).view(B, C)

        # 2. 获取任务嵌入 [B, 64]
        task_id_tensor = torch.tensor([task_id], device=student_feat.device).expand(B)
        task_embed = self.task_embeddings(task_id_tensor)

        # 3. 计算学生与各个教师的特征相似度向量 [B, num_teachers]
        similarity_vector = self.compute_similarity(
            teacher_feats_projected,
            teacher_feats_original
        )

        # 4. 融合输入并计算原始权重
        gating_input = torch.cat([student_pooled, task_embed, similarity_vector], dim=1)
        raw_logits = self.gating_net(gating_input)
        raw_weights = F.softmax(raw_logits, dim=-1)  # 每个样本的实时权重 [B, num_teachers]

        # 5. EMA 平滑逻辑
        if self.training and use_ema:
            # 计算当前 Batch 的平均权重
            batch_avg_weights = raw_weights.mean(dim=0)

            # 更新缓冲区
            with torch.no_grad():
                self.ema_weights = (
                        self.ema_momentum * self.ema_weights +
                        (1 - self.ema_momentum) * batch_avg_weights
                )

            # 训练时返回平滑后的权重（广播回全 Batch）
            return self.ema_weights.unsqueeze(0).expand(B, -1)
        else:
            # 验证或关闭 EMA 时，返回该样本的真实权重
            return raw_weights


class LossLevelGating(nn.Module):
    """
    损失级门控网络（修复版）

    功能：
    1. 根据学生特征和任务上下文，动态调整三个损失项的权重
    2. 使用 Gumbel-Softmax 确保权重和为 1
    3. 支持 EMA 平滑，避免训练不稳定

    输出：
    - w_task: 任务损失权重
    - w_crd: CRD 对比损失权重
    - w_output: 输出蒸馏损失权重
    """

    def __init__(
            self,
            student_dim: int,
            num_loss_terms: int = 3,
            num_tasks: int = 4,
            ema_momentum: float = 0.9,
            temperature: float = 1.0
    ):
        super().__init__()
        """
        Args:
            student_dim: 学生特征维度
            num_loss_terms: 损失项数量（3个：task, CRD, output）
            num_tasks: 任务数量
            ema_momentum: EMA 平滑系数
            temperature: Gumbel-Softmax 温度参数
        """
        self.num_loss_terms = num_loss_terms
        self.num_tasks = num_tasks
        self.ema_momentum = ema_momentum
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # ========== 1. 全局特征提取 ==========
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ========== 2. 任务嵌入 ==========
        self.task_embedding = nn.Embedding(num_tasks, 64)

        # ========== 3. 门控网络（输入：学生特征 + 任务嵌入）==========
        gating_input_dim = student_dim + 64

        self.gate_network = nn.Sequential(
            nn.Linear(gating_input_dim, 256),
            nn.LayerNorm(256),  # ✅ 添加 LayerNorm 稳定训练
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(128, num_loss_terms)  # 输出 3 个 logits
        )

        # ========== 4. EMA 权重缓冲区 ==========
        self.register_buffer(
            'ema_weights',
            torch.ones(num_loss_terms) / num_loss_terms  # 初始化为均匀分布
        )

        # ========== 5. 初始化权重 ==========
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
            self,
            student_features: torch.Tensor,
            task_id: int,
            use_ema: bool = True
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            student_features: 学生特征 [B, C, H, W]
            task_id: 任务ID (0~3)
            use_ema: 是否使用 EMA 平滑（训练时建议开启）

        Returns:
            weights: 损失权重 [B, 3] (w_task, w_crd, w_output)
        """
        batch_size = student_features.size(0)
        device = student_features.device

        # ========== 1. 全局池化提取特征 ==========
        global_feat = self.global_pool(student_features).flatten(1)  # [B, C]

        # ========== 2. 获取任务嵌入 ==========
        task_id_tensor = torch.tensor([task_id], device=device).expand(batch_size)
        task_embed = self.task_embedding(task_id_tensor)  # [B, 64]

        # ========== 3. 拼接特征 ==========
        combined = torch.cat([global_feat, task_embed], dim=1)  # [B, C + 64]

        # ========== 4. 通过门控网络 ==========
        logits = self.gate_network(combined)  # [B, 3]

        # ========== 5. Gumbel-Softmax（确保权重和为 1）==========
        if self.training:
            # 训练时使用 Gumbel-Softmax（可微分）
            weights = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=-1)
        else:
            # 验证时使用标准 Softmax
            weights = F.softmax(logits / self.temperature, dim=-1)

        # ========== 6. EMA 平滑（可选）==========
        if self.training and use_ema:
            # 计算当前 Batch 的平均权重
            batch_avg_weights = weights.mean(dim=0)  # [3]

            # 更新 EMA 缓冲区
            with torch.no_grad():
                self.ema_weights = (
                        self.ema_momentum * self.ema_weights +
                        (1 - self.ema_momentum) * batch_avg_weights
                )

            # 返回平滑后的权重（广播到全 Batch）
            return self.ema_weights.unsqueeze(0).expand(batch_size, -1)
        else:
            # 验证时或关闭 EMA 时，返回原始权重
            return weights

    def get_ema_weights(self) -> torch.Tensor:
        """获取当前的 EMA 权重（用于日志记录）"""
        return self.ema_weights.clone()


class CrossTaskKnowledgeDistiller(nn.Module):
    """跨任务知识蒸馏器 - 完整修复版"""

    def __init__(
            self,
            student_dims: Dict[str, int],
            teacher_dims: Dict[str, Dict[str, int]],
            verbose: bool = False
    ):
        super().__init__()
        self.verbose = verbose
        self.student_dims = student_dims
        self.teacher_dims = teacher_dims
        self.num_teachers = len(teacher_dims)
        self.temperature = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        # ========== ✅ 修复1：添加 Return Decoder（Forward + Inverse）==========
        self.return_decoder_forward = nn.ModuleDict()
        self.return_decoder_inverse = nn.ModuleDict()

        # 计算每层的平均教师维度
        avg_teacher_dims = {}
        for layer_name in student_dims.keys():
            dims = []
            for t_dims in teacher_dims.values():
                if layer_name in t_dims:
                    dims.append(t_dims[layer_name])
            avg_teacher_dims[layer_name] = int(np.mean(dims)) if dims else student_dims[layer_name]

        # 为每层创建 Forward 和 Inverse Decoder
        for layer_name, s_dim in student_dims.items():
            t_dim = avg_teacher_dims[layer_name]

            # Forward Projection: 学生维度 -> 教师维度
            self.return_decoder_forward[layer_name] = nn.Sequential(
                nn.Conv2d(s_dim, t_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(t_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(t_dim, t_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(t_dim)
            )

            # ✅ Inverse Projection: 教师维度 -> 学生维度（重构）
            self.return_decoder_inverse[layer_name] = nn.Sequential(
                nn.Conv2d(t_dim, t_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(t_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(t_dim, s_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(s_dim)
            )

        # 初始化其他模块
        self.channel_aligners = nn.ModuleDict()
        self.projectors = nn.ModuleDict()
        self.teacher_gates = nn.ModuleDict()
        self.attention_modules = nn.ModuleDict()
        self.adapters = nn.ModuleDict()

        self.layer_mapping = {
            'stem': 'stem', 'block2': 'block2', 'block3': 'block3',
            'block5': 'block5', 'final': 'final', 'decoder_input': 'decoder_input',
            'decoder_stage_0': 'decoder_stage_0', 'decoder_stage_1': 'decoder_stage_1',
            'decoder_stage_2': 'decoder_stage_2', 'decoder_stage_3': 'decoder_stage_3',
            'decoder_output': 'decoder_output'
        }

        # 初始化通道对齐器和投影器
        for teacher_name, t_dims in teacher_dims.items():
            self.channel_aligners[teacher_name] = nn.ModuleDict()
            self.projectors[teacher_name] = nn.ModuleDict()

            for layer_name in student_dims.keys():
                s_dim = self._get_student_dim(layer_name, student_dims)
                t_dim = self._get_teacher_dim(layer_name, t_dims)

                if t_dim is None:
                    t_dim = self._find_nearest_teacher_dim(layer_name, t_dims)

                if t_dim is not None:
                    if t_dim != s_dim:
                        self.channel_aligners[teacher_name][layer_name] = nn.Sequential(
                            nn.Conv2d(t_dim, s_dim, kernel_size=1, bias=False),
                            nn.BatchNorm2d(s_dim),
                            nn.ReLU(inplace=True)
                        )

                    self.projectors[teacher_name][layer_name] = BidirectionalProjector(
                        student_dim=s_dim,
                        teacher_dim=t_dim
                    )

        # 初始化门控和注意力模块
        for layer_name, s_dim in student_dims.items():
            s_dim = self._get_student_dim(layer_name, student_dims)
            self.teacher_gates[layer_name] = TeacherLevelGating(
                student_dim=s_dim,
                num_teachers=self.num_teachers,
                num_tasks=4
            )
            self.attention_modules[layer_name] = CrossTeacherAttention(
                student_dim=s_dim,
                teacher_dim=s_dim
            )

        self._initialize_weights()

    def forward(
            self,
            student_features: Dict[str, torch.Tensor],
            all_teacher_features: Dict[str, Dict[str, torch.Tensor]],
            task_id: Optional[str] = None
    ) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
        """
        ✅ 修复版前向传播

        Args:
            student_features: {layer_name: [B, C_s, H, W]}
            all_teacher_features: {teacher_name: {layer_name: [B, C_t, H, W]}}
            task_id: 任务ID

        Returns:
            aligned_features: 对齐后的特征
            reconstructed_features: 重构的特征
            teacher_weights_log: 教师权重日志
            original_teacher_features: ✅ 原始教师特征（用于重构损失）
        """
        device = next(iter(student_features.values())).device
        aligned_features = {}
        reconstructed_features = {}
        teacher_weights_log = {}
        original_teacher_features = {}  # ✅ 新增

        active_teachers = self._get_active_teachers(task_id)

        for layer_name, s_feat in student_features.items():
            # ✅ 初始化该层的教师特征字典
            original_teacher_features[layer_name] = {}

            # 收集教师特征
            teacher_feats_list = []
            teacher_weights_for_layer = []

            for teacher_name in active_teachers:
                if teacher_name not in all_teacher_features:
                    continue

                teacher_features = all_teacher_features[teacher_name]
                if layer_name not in teacher_features:
                    continue

                t_feat = teacher_features[layer_name]

                # ✅ 空间对齐（保留原始通道数）
                if t_feat.shape[-2:] != s_feat.shape[-2:]:
                    t_feat_spatial_aligned = F.interpolate(
                        t_feat,
                        size=s_feat.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                else:
                    t_feat_spatial_aligned = t_feat

                # ✅ 保存原始教师特征（空间对齐后，通道对齐前）
                original_teacher_features[layer_name][teacher_name] = t_feat_spatial_aligned.detach().clone()

                # 通道对齐
                adapter_key = f"{teacher_name}_{layer_name}"
                if adapter_key in self.adapters:
                    t_feat_aligned = self.adapters[adapter_key](t_feat_spatial_aligned)
                elif teacher_name in self.channel_aligners and layer_name in self.channel_aligners[teacher_name]:
                    t_feat_aligned = self.channel_aligners[teacher_name][layer_name](t_feat_spatial_aligned)
                else:
                    t_feat_aligned = t_feat_spatial_aligned

                teacher_feats_list.append(t_feat_aligned)
                weight = self._get_teacher_weight(teacher_name, task_id)
                teacher_weights_for_layer.append(weight)

            # 加权融合
            if teacher_feats_list:
                total_weight = sum(teacher_weights_for_layer)
                normalized_weights = [w / total_weight for w in teacher_weights_for_layer] if total_weight > 0 else [
                                                                                                                        1.0 / len(
                                                                                                                            teacher_weights_for_layer)] * len(
                    teacher_weights_for_layer)

                weighted_teacher_feat = sum(
                    w * t_feat for w, t_feat in zip(normalized_weights, teacher_feats_list)
                )
                aligned_features[layer_name] = weighted_teacher_feat
                teacher_weights_log[layer_name] = dict(zip(active_teachers, normalized_weights))
            else:
                aligned_features[layer_name] = s_feat.clone()
                teacher_weights_log[layer_name] = {}

            # ✅ Return Decoder（Forward + Inverse）
            if layer_name in self.return_decoder_forward:
                try:
                    # Forward Projection
                    forward_feat = self.return_decoder_forward[layer_name](s_feat)

                    # ✅ Inverse Projection（重构）
                    reconstructed_feat = self.return_decoder_inverse[layer_name](forward_feat)

                    reconstructed_features[layer_name] = reconstructed_feat
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠️  Return decoder failed for {layer_name}: {e}")
                    reconstructed_features[layer_name] = s_feat.clone()
            else:
                reconstructed_features[layer_name] = s_feat.clone()

        # ✅ 返回四个值
        return aligned_features, reconstructed_features, teacher_weights_log, original_teacher_features

    def _get_student_dim(self, layer_name, student_dims):
        """获取学生维度"""
        if layer_name.startswith('decoder'):
            decoder_dims = {
                'decoder_input': 192, 'decoder_stage_0': 256,
                'decoder_stage_1': 128, 'decoder_stage_2': 64,
                'decoder_stage_3': 32, 'decoder_output': 1
            }
            return decoder_dims.get(layer_name, student_dims.get(layer_name, 1280))
        return student_dims.get(layer_name, 1280)

    def _get_teacher_dim(self, layer_name, t_dims):
        """获取教师维度"""
        if layer_name in t_dims:
            return t_dims[layer_name]
        return t_dims.get('final')

    def _find_nearest_teacher_dim(self, layer_name, teacher_layer_dims):
        """查找最近的教师维度"""
        if layer_name in teacher_layer_dims:
            return teacher_layer_dims[layer_name]
        if 'final' in teacher_layer_dims:
            return teacher_layer_dims['final']
        return next(iter(teacher_layer_dims.values())) if teacher_layer_dims else None

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_active_teachers(self, task_id: Optional[str] = None) -> List[str]:
        """获取活跃教师"""
        task_teacher_mapping = {
            0: ['MedSAM', 'USFM'],
            1: ['RETFound_MAE'],
            2: ['BioMedPrase']
        }
        if task_id is None:
            return list(self.teacher_dims.keys())
        return task_teacher_mapping.get(task_id, list(self.teacher_dims.keys()))

    def _get_teacher_weight(self, teacher_name: str, task_id: Optional[int] = None) -> float:
        """获取教师权重"""
        default_weights = {
            'MedSAM': 0.3,
            'USFM': 0.3,
            'RETFound_MAE': 0.2,
            'BioMedPrase': 0.2
        }
        task_specific_weights = {
            0: {'MedSAM': 0.5, 'USFM': 0.5},
            1: {'RETFound_MAE': 1.0},
            2: {'BioMedPrase': 1.0}
        }
        if task_id is not None and task_id in task_specific_weights:
            return task_specific_weights[task_id].get(teacher_name, 0.0)
        return default_weights.get(teacher_name, 0.25)


def create_teacher_dims_config(
        teacher_models: Dict[str, nn.Module],
        sample_input: torch.Tensor
) -> Dict[str, Dict[str, int]]:
    """
    自动提取教师模型的特征维度配置

    Args:
        teacher_models: 教师模型字典
        sample_input: 样本输入 [1, 3, 224, 224]

    Returns:
        teacher_dims: 特征维度配置
    """
    teacher_dims = {}

    for name, model in teacher_models.items():
        model.eval()
        with torch.no_grad():
            try:
                # 前向传播获取特征
                if hasattr(model, 'forward_features'):
                    features = model.forward_features(sample_input)
                else:
                    features = model(sample_input)

                # 提取维度
                if isinstance(features, dict):
                    teacher_dims[name] = {
                        k: v.size(1) if v.dim() == 4 else v.size(-1)
                        for k, v in features.items()
                    }
                elif isinstance(features, torch.Tensor):
                    # 单一特征输出
                    if features.dim() == 3:  # ViT: [B, N, C]
                        teacher_dims[name] = {'final': features.size(-1)}
                    elif features.dim() == 4:  # CNN: [B, C, H, W]
                        teacher_dims[name] = {'final': features.size(1)}
                else:
                    print(f"Warning: Unknown feature format for {name}")
                    teacher_dims[name] = {'final': 768}  # 默认值

            except Exception as e:
                print(f"Error extracting dims for {name}: {e}")
                teacher_dims[name] = {'final': 768}

    return teacher_dims


def validate_distiller_initialization(
        distiller: CrossTaskKnowledgeDistiller,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, Dict[str, torch.Tensor]]
):
    """验证蒸馏器初始化是否正确"""
    print("\n=== Distiller Validation ===")

    for layer_name, s_feat in student_features.items():
        print(f"\n[{layer_name}]")
        print(f"  Student: {s_feat.shape}")

        for teacher_name, t_feats in teacher_features.items():
            if layer_name in t_feats:
                t_feat = t_feats[layer_name]
                print(f"  Teacher {teacher_name}: {t_feat.shape}")

                # 检查是否有对齐器
                if (teacher_name in distiller.channel_aligners and
                        layer_name in distiller.channel_aligners[teacher_name]):
                    print(f"    ✅ Channel aligner exists")
                else:
                    if t_feat.size(1) != s_feat.size(1):
                        print(f"    ⚠️  No channel aligner (dim mismatch)")


# ============================================
# 3. 改进的多任务学生模型
# ============================================

class ImprovedMultiTaskStudent(nn.Module):
    """改进的多任务学生模型（支持统一训练）"""

    def __init__(self, num_classes_dict: Dict[str, int], efficientnet_version='b0'):
        super().__init__()

        self.efficientnet_version = efficientnet_version
        # 使用EfficientNet作为backbone
        if efficientnet_version == 'b0':
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.feature_dims = {
                'stem': 32, 'block1': 16, 'block2': 24,
                'block3': 40, 'block4': 80, 'block5': 112,
                'block6': 192, 'block7': 320, 'final': 1280,
                'default': 1280
            }
        elif efficientnet_version == 'b3':
            self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
            self.feature_dims = {
                'stem': 40, 'block1': 24, 'block2': 32,
                'block3': 48, 'block4': 96, 'block5': 136,
                'block6': 232, 'block7': 384, 'final': 1536,
                'default': 1536
            }
        feature_dim = self.feature_dims['final']
        self.task_adapters = nn.ModuleDict({
            'segmentation': TaskAdapter(feature_dim, feature_dim),
            'classification': TaskAdapter(feature_dim, feature_dim),
            'recognition': TaskAdapter(feature_dim, feature_dim)
        })

        # 移除原始的分类头
        self.backbone.classifier = nn.Identity()

        # 添加特征提取钩子
        self.feature_hooks = []
        self.features = {}

        # 注册钩子
        self._register_hooks()

        # 任务特定的解码器
        self.task_decoders = nn.ModuleDict({
            'segmentation': self._create_segmentation_decoder(),
            'classification': self._create_classification_head(num_classes_dict['classification']),
            'recognition': self._create_classification_head(num_classes_dict['recognition'])
        })

        # ========== 新增：数据集到任务类型的映射 ==========
        self.dataset_to_task = {
            'BUSI': 'segmentation',
            'kvasir_seg': 'segmentation',
            'APTOS2019': 'classification',
            'ISIC2017': 'recognition'
        }

        # ========== 新增：数据集到任务ID的映射 ==========
        self.dataset_to_task_id = {
            'BUSI': 0,
            'kvasir_seg': 0,
            'APTOS2019': 1,
            'ISIC2017': 2
        }

        # 任务选择器
        feature_dim = self.feature_dims['final']
        self.task_selector = nn.Embedding(3, feature_dim)  # 3个任务类型
        # 4个损失项：task_loss, feature_distill, output_distill


    def _get_task_type(self, dataset_name: str) -> str:
        """
        根据数据集名称返回任务类型

        Args:
            dataset_name: 数据集名称 (BUSI, kvasir_seg, APTOS2019, ISIC2017)

        Returns:
            task_type: 任务类型 (segmentation, classification, recognition)
        """
        task_type = self.dataset_to_task.get(dataset_name, 'segmentation')

        # 验证任务类型是否有效
        if task_type not in self.task_adapters:
            raise ValueError(
                f"Unknown task type '{task_type}' for dataset '{dataset_name}'. "
                f"Available types: {list(self.task_adapters.keys())}"
            )

        return task_type

    def _register_hooks(self):
        """注册前向钩子以提取中间特征"""

        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output

            return hook

        # 注册到关键层
        self.feature_hooks.append(
            self.backbone.features[0].register_forward_hook(get_activation('stem'))
        )
        self.feature_hooks.append(
            self.backbone.features[2].register_forward_hook(get_activation('block2'))
        )
        self.feature_hooks.append(
            self.backbone.features[3].register_forward_hook(get_activation('block3'))
        )
        self.feature_hooks.append(
            self.backbone.features[5].register_forward_hook(get_activation('block5'))
        )

    def _create_segmentation_decoder(self):
        """创建通用的分割解码器"""

        if self.efficientnet_version == 'b0':
            return nn.ModuleList([
                # 上采样路径
                nn.Sequential(
                    nn.Conv2d(192, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                ),
                nn.Sequential(
                    nn.Conv2d(256 + 112, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                ),
                nn.Sequential(
                    nn.Conv2d(128 + 40, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                ),
                nn.Sequential(
                    nn.Conv2d(64 + 24, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                ),
                nn.Sequential(
                    nn.Conv2d(32, 16, 3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 1, 1)  # 单通道输出用于二值分割
                )
            ])
        else:  # 对于b3版本
            return nn.ModuleList([
                nn.Sequential(
                    nn.ConvTranspose2d(232, 256, 3, 2, 1, 1),  # b3的block6输出232
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                ),
                # ... [其他层适配b3] ...
            ])

    def _create_classification_head(self, num_classes):
        """创建分类头（适用于分类和识别任务）"""
        feature_dim = self.feature_dims['final']
        return nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(256, num_classes)  # 输出分类结果
        )

    def decode(self, features: Dict[str, torch.Tensor], dataset_name: str = None, task_id: int = None) -> torch.Tensor:
        """
        从融合特征生成最终预测

        Args:
            features: 融合后的特征字典
            dataset_name: 数据集名称
            task_id: 任务ID

        Returns:
            predictions: 最终预测结果
        """
        # ========== 1. 确定任务类型 ==========
        if dataset_name is not None:
            task_type = self.dataset_to_task[dataset_name]
            task_id = self.dataset_to_task_id[dataset_name]
        elif task_id is not None:
            task_type_map = {0: 'segmentation', 1: 'classification', 2: 'recognition'}
            task_type = task_type_map[task_id]
        else:
            raise ValueError("Either dataset_name or task_id must be provided")

        # ========== 2. 获取最终特征 ==========
        if 'final' in features:
            final_feat = features['final']
        else:
            # 使用最后一个特征
            final_feat = list(features.values())[-1]

        # ========== 3. 应用任务适配器 ==========
        adapted_feat = self.task_adapters[task_type](final_feat)

        # ========== 4. 根据任务类型解码 ==========
        if task_type == 'segmentation':
            # 分割任务需要完整的特征金字塔
            return self._decode_segmentation(features)
        else:
            # 分类/识别任务只需要最终特征
            return self.task_decoders[task_type](adapted_feat)

    def _decode_segmentation(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """分割任务的解码（使用融合特征）"""
        # 获取编码器特征（按顺序）
        enc_features = []
        for layer_name in ['block2', 'block3', 'block5', 'final']:
            if layer_name in features:
                enc_features.append(features[layer_name])

        if not enc_features:
            raise ValueError("No valid encoder features for segmentation")

        # 解码器前向传播
        x_dec = enc_features[-1]  # 从最深层开始

        for i, decoder_block in enumerate(self.task_decoders['segmentation'][:-1]):
            x_dec = decoder_block(x_dec)

            # 跳跃连接
            if i < len(enc_features) - 1:
                skip = enc_features[-(i + 2)]

                # 调整尺寸
                if x_dec.shape[-2:] != skip.shape[-2:]:
                    x_dec = F.interpolate(
                        x_dec,
                        size=skip.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )

                # 拼接
                x_dec = torch.cat([x_dec, skip], dim=1)

        # 最终输出层
        output = self.task_decoders['segmentation'][-1](x_dec)

        return output

    # ========== 核心修改：支持 task_id 参数 ==========
    def forward(self, x, dataset_name: str = None, task_id: int = None):
        """
        前向传播（支持统一训练）

        Args:
            x: 输入图像 [B, C, H, W]
            dataset_name: 数据集名称（优先使用）
            task_id: 任务ID（当 dataset_name 为 None 时使用）

        Returns:
            output: 任务输出
            features: 提取的特征字典
        """
        # 清空特征缓存
        self.features = {}

        # ========== 新增：灵活的任务ID获取 ==========
        if dataset_name is not None:
            # 优先使用 dataset_name
            task_id = self.dataset_to_task_id[dataset_name]
            task_type = self.dataset_to_task[dataset_name]
        elif task_id is not None:
            # 使用传入的 task_id
            task_type_map = {0: 'segmentation', 1: 'classification', 2: 'recognition'}
            task_type = task_type_map[task_id]
        else:
            raise ValueError("Either dataset_name or task_id must be provided")

        # 通过backbone
        features = self.backbone.features(x)

        # ✅ 应用任务适配器
        adapted_features = self.task_adapters[task_type](features)

        # 获取任务嵌入并添加到特征中
        task_embed = self.task_selector(torch.tensor(task_id, device=x.device))
        task_embed = task_embed.view(1, -1, 1, 1).expand_as(features)
        task_features = features + task_embed

        # ========== 根据任务选择解码器并返回特征 ==========
        if task_type == 'segmentation':  # 分割任务
            seg_output, decoder_features = self._forward_segmentation(x)
            # 合并Encoder和Decoder特征
            all_features = {**self.features, **decoder_features}
            return seg_output, all_features
        elif task_type == 'classification':  # 分类任务
            cls_output = self.task_decoders['classification'](task_features)
            return cls_output, self.features
        else:  # 识别任务
            rec_output = self.task_decoders['recognition'](task_features)
            return rec_output, self.features

    def _forward_segmentation(self, x):
        """分割任务的前向传播（带跳跃连接）+ 返回Decoder特征"""
        # 编码器特征
        enc_features = []
        x_in = x

        for i, block in enumerate(self.backbone.features):
            x_in = block(x_in)
            if i in [2, 3, 5, 6]:  # 保存中间特征：block2, block3, block5, block6
                enc_features.append(x_in)

        # ========== 记录Decoder特征 ==========
        decoder_features = {}

        # 解码器
        x_dec = enc_features[-1]  # 使用block6的输出
        decoder_features['decoder_input'] = x_dec  # 记录Decoder输入

        for i, decoder_block in enumerate(self.task_decoders['segmentation'][:-1]):
            x_dec = decoder_block(x_dec)

            # ✅ 在跳跃连接前记录特征
            decoder_features[f'decoder_stage_{i}_before_skip'] = x_dec

            if i < len(enc_features) - 1:
                # 跳跃连接
                skip = enc_features[-(i + 2)]
                # 调整尺寸
                if x_dec.shape[-2:] != skip.shape[-2:]:
                    x_dec = F.interpolate(
                        x_dec,
                        size=skip.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                x_dec = torch.cat([x_dec, skip], dim=1)

                # ✅ 记录融合后的特征
                decoder_features[f'decoder_stage_{i}_after_skip'] = x_dec

        # 最终输出层
        output = self.task_decoders['segmentation'][-1](x_dec)
        decoder_features['decoder_output'] = output  # 记录最终输出

        # 确保输出形状与输入一致
        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(
                output,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        return output, decoder_features


# ============================================
# 4. 异构蒸馏损失函数
# ============================================


class HeterogeneousDistillationLoss(nn.Module):
    """
    异构蒸馏损失模块
    整合任务损失、输出蒸馏、特征蒸馏、CRD损失和重构损失
    """

    def __init__(
            self,
            num_tasks=4,
            task_weight=1.0,
            output_distill_weight=1.0,
            feature_distill_weight=1.0,
            crd_weight=0.5,
            recon_weight=0.1,
            temperature=4.0,
            crd_temperature=0.07,
            loss_components=None,
            use_loss_gating=True,  # ✅ 新增：是否启用损失级门控
            student_dim=1280  # ✅ 新增：学生特征维度
    ):
        super().__init__()

        # 损失权重（固定权重，作为后备）
        self.task_weight = task_weight
        self.output_distill_weight = output_distill_weight
        self.feature_distill_weight = feature_distill_weight
        self.crd_weight = crd_weight
        self.recon_weight = recon_weight
        self.temperature = temperature

        # 损失组件选择
        if loss_components is None:
            self.loss_components = ['task', 'output', 'feature', 'crd', 'recon']
        else:
            self.loss_components = loss_components

        # ✅ 初始化 CRD 损失
        if 'crd' in self.loss_components:
            self.crd_loss = CRDLoss(temperature=crd_temperature)

        # 任务特定损失函数
        self.task_losses = nn.ModuleDict({
            'segmentation': nn.CrossEntropyLoss(ignore_index=255),
            'depth': nn.L1Loss(),
            'detection': nn.CrossEntropyLoss(),
            'classification': nn.CrossEntropyLoss()
        })

        # 特征蒸馏损失
        self.feature_criterion = nn.MSELoss()

        # 重构损失
        self.reconstruction_criterion = nn.MSELoss()

        # ✅ 初始化损失级门控网络
        self.use_loss_gating = use_loss_gating
        if self.use_loss_gating:
            self.loss_gate = LossLevelGating(
                student_dim=student_dim,
                num_loss_terms=3,  # task, crd, output
                num_tasks=num_tasks,
                ema_momentum=0.9,
                temperature=1.0
            )
            print(f"✅ Loss-Level Gating Network initialized (student_dim={student_dim})")
        else:
            self.loss_gate = None
            print(f"⚠️  Loss-Level Gating disabled, using fixed weights")

    def _align_features_for_loss(self, source_feat, target_feat):
        """对齐特征用于损失计算"""
        # 空间对齐
        if source_feat.shape[-2:] != target_feat.shape[-2:]:
            source_feat = F.interpolate(
                source_feat,
                size=target_feat.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        # 通道对齐
        if source_feat.shape[1] != target_feat.shape[1]:
            key = f"recon_align_{source_feat.shape[1]}_{target_feat.shape[1]}"

            if not hasattr(self, key):
                conv = nn.Conv2d(
                    source_feat.shape[1],
                    target_feat.shape[1],
                    kernel_size=1,
                    bias=False
                ).to(source_feat.device)
                setattr(self, key, conv)

            conv = getattr(self, key)
            source_feat = conv(source_feat)

        return source_feat

    def compute_task_loss(self, outputs, targets, dataset_name):
        """
        计算任务特定损失
        """
        if dataset_name not in self.task_losses:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        task_output = outputs['task_output']
        loss_fn = self.task_losses[dataset_name]

        # 根据任务类型调整输出和目标的形状
        if dataset_name == 'segmentation':
            # 语义分割：outputs [B, C, H, W], targets [B, H, W]
            return loss_fn(task_output, targets)

        elif dataset_name == 'depth':
            # 深度估计：outputs [B, 1, H, W], targets [B, 1, H, W]
            return loss_fn(task_output, targets)

        elif dataset_name == 'detection':
            # 目标检测：简化处理
            # 实际应该使用更复杂的检测损失（如Focal Loss + IoU Loss）
            return loss_fn(task_output, targets)

        elif dataset_name == 'classification':
            # 图像分类：outputs [B, num_classes], targets [B]
            return loss_fn(task_output, targets)

        else:
            return loss_fn(task_output, targets)

    def compute_output_distillation_loss(self, student_output, teacher_outputs):
        """
        计算输出层蒸馏损失（KL散度）
        """
        if not teacher_outputs:
            return torch.tensor(0.0, device=student_output.device)

        total_loss = 0
        num_teachers = len(teacher_outputs)

        # 对学生输出应用温度缩放
        student_soft = F.log_softmax(student_output / self.temperature, dim=1)

        for teacher_name, teacher_output in teacher_outputs.items():
            # 对教师输出应用温度缩放
            teacher_soft = F.softmax(teacher_output / self.temperature, dim=1)

            # KL散度损失
            kl_loss = F.kl_div(
                student_soft,
                teacher_soft,
                reduction='batchmean'
            ) * (self.temperature ** 2)

            total_loss += kl_loss

        return total_loss / num_teachers

    def compute_feature_distillation_loss(self, student_features, teacher_features_dict):
        """
        计算特征层蒸馏损失
        """
        if not teacher_features_dict:
            return torch.tensor(0.0, device=list(student_features.values())[0].device)

        total_loss = 0
        num_pairs = 0

        for teacher_name, teacher_features in teacher_features_dict.items():
            for layer_name, student_feat in student_features.items():
                if layer_name in teacher_features:
                    teacher_feat = teacher_features[layer_name]

                    # 确保特征维度匹配
                    if student_feat.shape != teacher_feat.shape:
                        # 使用自适应池化调整空间维度
                        teacher_feat = F.adaptive_avg_pool2d(
                            teacher_feat,
                            student_feat.shape[2:]
                        )

                        # 如果通道数不同，使用1x1卷积调整
                        if student_feat.shape[1] != teacher_feat.shape[1]:
                            continue  # 跳过通道数不匹配的层

                    # 计算MSE损失
                    loss = self.feature_criterion(student_feat, teacher_feat)
                    total_loss += loss
                    num_pairs += 1

        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)

    def compute_crd_loss(self, student_features, teacher_features_dict):
        """
        ✅ 新增：计算CRD对比损失
        """
        if not teacher_features_dict or 'crd' not in self.loss_components:
            return torch.tensor(0.0, device=list(student_features.values())[0].device)

        total_loss = 0
        num_pairs = 0

        for teacher_name, teacher_features in teacher_features_dict.items():
            for layer_name, student_feat in student_features.items():
                if layer_name in teacher_features:
                    teacher_feat = teacher_features[layer_name]

                    # 确保空间维度匹配
                    if student_feat.shape[2:] != teacher_feat.shape[2:]:
                        teacher_feat = F.adaptive_avg_pool2d(
                            teacher_feat,
                            student_feat.shape[2:]
                        )

                    # 确保通道数匹配
                    if student_feat.shape[1] != teacher_feat.shape[1]:
                        continue  # 跳过通道数不匹配的层

                    # 计算CRD损失
                    crd_loss = self.crd_loss(student_feat, teacher_feat)
                    total_loss += crd_loss
                    num_pairs += 1

        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)

    def compute_reconstruction_loss(self, outputs):
        """
        计算重构损失
        """
        if 'reconstruction_loss' in outputs:
            return outputs['reconstruction_loss']
        return torch.tensor(0.0, device=list(outputs['student_features'].values())[0].device)

    def forward(self, outputs, targets, dataset_name=None, task_id=None, verbose=False):
        """
        计算异构蒸馏损失（启用损失级门控）

        Args:
            outputs: 模型输出字典
            targets: 真实标签
            dataset_name: 数据集名称
            task_id: 任务ID（✅ 新增，用于损失级门控）
            verbose: 是否打印详细信息

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        device = next(self.parameters()).device
        losses = {}

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"[Loss Computation] Dataset: {dataset_name}, Task ID: {task_id}")
            print(f"{'=' * 80}")

        # ========== 1. 任务损失 ==========
        if 'task' in self.loss_components:
            student_output = outputs.get('student_output')

            if student_output is not None and targets is not None:
                if dataset_name in ['coco', 'voc']:
                    # 检测任务
                    if isinstance(targets, (list, tuple)):
                        task_loss = 0
                        for pred, target in zip(student_output, targets):
                            if isinstance(pred, dict):
                                task_loss += sum(loss for loss in pred.values())
                            else:
                                task_loss += F.cross_entropy(pred, target)
                        losses['task'] = task_loss / len(targets)
                    else:
                        losses['task'] = F.cross_entropy(student_output, targets)
                else:
                    # 分类任务
                    losses['task'] = F.cross_entropy(student_output, targets)
            else:
                losses['task'] = torch.tensor(0.0, device=device)

            if verbose:
                print(f"  Task Loss: {losses['task'].item():.4f}")

        # ========== 2. CRD 对比损失 ==========
        if 'crd' in self.loss_components:
            crd_loss = 0
            num_layers = 0

            student_features = outputs.get('student_features', {})
            aligned_features = outputs.get('aligned_features', {})

            for layer_name in student_features.keys():
                if layer_name not in aligned_features:
                    continue

                s_feat = student_features[layer_name]
                t_feat = aligned_features[layer_name]

                # 空间对齐
                if s_feat.shape[-2:] != t_feat.shape[-2:]:
                    t_feat = F.interpolate(
                        t_feat,
                        size=s_feat.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )

                # 通道对齐
                if s_feat.shape[1] != t_feat.shape[1]:
                    proj_key = f'_crd_proj_{layer_name}'
                    if not hasattr(self, proj_key):
                        proj = nn.Conv2d(
                            t_feat.shape[1],
                            s_feat.shape[1],
                            kernel_size=1,
                            bias=False
                        ).to(device)
                        setattr(self, proj_key, proj)

                    proj = getattr(self, proj_key)
                    t_feat = proj(t_feat)

                # 计算 CRD 损失
                crd_loss += self.crd_loss(s_feat, t_feat)
                num_layers += 1

            losses['crd'] = crd_loss / num_layers if num_layers > 0 else torch.tensor(0.0, device=device)

            if verbose:
                print(f"  CRD Loss: {losses['crd'].item():.4f} (from {num_layers} layers)")

        # ========== 3. 输出蒸馏损失 ==========
        if 'output' in self.loss_components:
            student_output = outputs.get('student_output')
            teacher_output = outputs.get('teacher_output')

            if student_output is not None and teacher_output is not None:
                if dataset_name in ['coco', 'voc']:
                    losses['output'] = torch.tensor(0.0, device=device)
                else:
                    student_log_probs = F.log_softmax(student_output / self.temperature, dim=1)
                    teacher_probs = F.softmax(teacher_output / self.temperature, dim=1)
                    losses['output'] = F.kl_div(
                        student_log_probs,
                        teacher_probs,
                        reduction='batchmean'
                    ) * (self.temperature ** 2)
            else:
                losses['output'] = torch.tensor(0.0, device=device)

            if verbose:
                print(f"  Output Loss: {losses['output'].item():.4f}")

        # ========== 4. 特征蒸馏损失 ==========
        if 'feature' in self.loss_components:
            if verbose:
                print(f"\n[Loss Component 4] Feature Distillation Loss:")

            feature_loss = torch.tensor(0.0, device=device)

            # 检查必需的键是否存在
            if 'student_features' in outputs and 'aligned_features' in outputs:
                student_features = outputs['student_features']
                aligned_features = outputs['aligned_features']

                # 遍历每一层计算特征蒸馏损失
                for layer_name in student_features.keys():
                    if layer_name in aligned_features:
                        s_feat = student_features[layer_name]
                        t_feat = aligned_features[layer_name]

                        # 确保特征维度匹配
                        if s_feat.shape == t_feat.shape:
                            # 计算 L2 损失（MSE）
                            layer_loss = F.mse_loss(s_feat, t_feat)
                            feature_loss += layer_loss

                            if verbose:
                                print(f"  Layer '{layer_name}': {layer_loss.item():.4f}")
                        else:
                            if verbose:
                                print(f"  Layer '{layer_name}': Shape mismatch - "
                                      f"Student {s_feat.shape} vs Teacher {t_feat.shape}")

                # 归一化（按层数平均）
                num_layers = len([k for k in student_features.keys() if k in aligned_features])
                if num_layers > 0:
                    feature_loss = feature_loss / num_layers

                losses['feature'] = feature_loss

                if verbose:
                    print(f"  Total Feature Loss: {feature_loss.item():.4f}")
            else:
                # 如果缺少必需的特征，损失为 0
                losses['feature'] = torch.tensor(0.0, device=device)
                if verbose:
                    print(f"  Feature Loss: 0.0 (missing required features)")

        # ========== 5. 重构损失 ==========
        if 'recon' in self.loss_components:
            if verbose:
                print(f"\n[Loss Component 5] Reconstruction Loss:")


            recon_loss, rec_loss_dict = self._compute_reconstruction_loss(
                reconstructed_features=outputs.get('reconstructed_features', {}),
                original_teacher_features=outputs.get('original_teacher_features', {}),
                device=device
            )

            losses['recon'] = recon_loss

            if verbose:
                print(f"  Total Reconstruction Loss: {recon_loss.item():.4f}")
                for layer_name, loss_value in rec_loss_dict.items():
                    print(f"    {layer_name}: {loss_value:.4f}")

            # 检查必需的键是否存在
            if 'reconstructed_features' in outputs and 'original_teacher_features' in outputs:
                reconstructed_features = outputs['reconstructed_features']
                original_teacher_features = outputs['original_teacher_features']

                # 遍历每一层计算重构损失
                for layer_name in reconstructed_features.keys():
                    if layer_name in original_teacher_features:
                        recon_feat = reconstructed_features[layer_name]

                        # 获取该层所有教师的原始特征
                        teacher_feats_list = original_teacher_features[layer_name]

                        if len(teacher_feats_list) > 0:
                            # 计算重构特征与每个教师特征的损失
                            layer_recon_loss = torch.tensor(0.0, device=device)

                            for teacher_name, orig_feat in teacher_feats_list:
                                # 确保空间维度匹配
                                if orig_feat.shape[-2:] != recon_feat.shape[-2:]:
                                    orig_feat = F.interpolate(
                                        orig_feat,
                                        size=recon_feat.shape[-2:],
                                        mode='bilinear',
                                        align_corners=False
                                    )

                                # 确保通道维度匹配
                                if orig_feat.shape[1] != recon_feat.shape[1]:
                                    # 使用自适应平均池化在通道维度上对齐
                                    if orig_feat.shape[1] > recon_feat.shape[1]:
                                        # 教师通道数更多，需要降维
                                        orig_feat = F.adaptive_avg_pool3d(
                                            orig_feat.unsqueeze(2),
                                            (1, orig_feat.shape[2], orig_feat.shape[3])
                                        ).squeeze(2)
                                        # 使用 1x1 卷积对齐通道
                                        if orig_feat.shape[1] != recon_feat.shape[1]:
                                            orig_feat = F.interpolate(
                                                orig_feat,
                                                size=(recon_feat.shape[1],
                                                      recon_feat.shape[2],
                                                      recon_feat.shape[3]),
                                                mode='trilinear',
                                                align_corners=False
                                            )
                                    else:
                                        # 重构特征通道数更多，需要降维
                                        recon_feat_aligned = F.adaptive_avg_pool3d(
                                            recon_feat.unsqueeze(2),
                                            (1, recon_feat.shape[2], recon_feat.shape[3])
                                        ).squeeze(2)

                                        if recon_feat_aligned.shape[1] != orig_feat.shape[1]:
                                            recon_feat_aligned = F.interpolate(
                                                recon_feat_aligned,
                                                size=(orig_feat.shape[1],
                                                      orig_feat.shape[2],
                                                      orig_feat.shape[3]),
                                                mode='trilinear',
                                                align_corners=False
                                            )
                                        recon_feat = recon_feat_aligned

                                # 计算 L2 损失
                                teacher_loss = F.mse_loss(recon_feat, orig_feat)
                                layer_recon_loss += teacher_loss

                                if verbose:
                                    print(f"  Layer '{layer_name}' - Teacher '{teacher_name}': "
                                          f"{teacher_loss.item():.4f}")

                            # 平均所有教师的重构损失
                            layer_recon_loss = layer_recon_loss / len(teacher_feats_list)
                            recon_loss += layer_recon_loss

                # 归一化（按层数平均）
                num_layers = len([k for k in reconstructed_features.keys()
                                  if k in original_teacher_features])
                if num_layers > 0:
                    recon_loss = recon_loss / num_layers

                losses['recon'] = recon_loss

                if verbose:
                    print(f"  Total Reconstruction Loss: {recon_loss.item():.4f}")
            else:
                # 如果缺少必需的特征，损失为 0
                losses['recon'] = torch.tensor(0.0, device=device)
                if verbose:
                    print(f"  Reconstruction Loss: 0.0 (missing required features)")

        # ========== 6. ✅ 使用损失级门控计算总损失 ==========
        if self.use_loss_gating and task_id is not None:
            # 获取学生特征（用于门控网络）
            student_features = outputs.get('student_features', {})

            if 'final' in student_features:
                student_feat = student_features['final']
            elif len(student_features) > 0:
                student_feat = list(student_features.values())[-1]
            else:
                # 如果没有特征，使用固定权重
                if verbose:
                    print("  ⚠️  No student features, using fixed weights")
                return self._compute_fixed_weighted_loss(losses), losses

            # 通过损失级门控网络
            loss_weights = self.loss_gate(
                student_features=student_feat,
                task_id=task_id,
                use_ema=self.training  # 训练时使用 EMA
            )  # [B, 3]

            # 取 Batch 平均权重
            loss_weights = loss_weights.mean(dim=0)  # [3]

            # 计算加权总损失
            total_loss = (
                    loss_weights[0] * losses.get('task', 0) +
                    loss_weights[1] * losses.get('crd', 0) +
                    loss_weights[2] * losses.get('output', 0)
            )

            # 添加其他损失（使用固定权重）
            if 'feature' in losses:
                total_loss += self.feature_distill_weight * losses['feature']
            if 'recon' in losses:
                total_loss += self.recon_weight * losses['recon']

            if verbose:
                print(f"\n{'─' * 80}")
                print(f"  Dynamic Loss Weights (from Gating):")
                print(f"    w_task:   {loss_weights[0].item():.4f}")
                print(f"    w_crd:    {loss_weights[1].item():.4f}")
                print(f"    w_output: {loss_weights[2].item():.4f}")
                print(f"  Total Loss: {total_loss.item():.4f}")
                print(f"{'=' * 80}\n")
        else:
            # 使用固定权重
            total_loss = self._compute_fixed_weighted_loss(losses)

            if verbose:
                print(f"\n{'─' * 80}")
                print(f"  Using Fixed Weights")
                print(f"  Total Loss: {total_loss.item():.4f}")
                print(f"{'=' * 80}\n")

        return total_loss, losses

    def _compute_reconstruction_loss(
            self,
            reconstructed_features: Dict[str, torch.Tensor],
            original_teacher_features: Dict[str, Dict[str, torch.Tensor]],
            device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        ✅ 修复版重构损失计算

        Args:
            reconstructed_features: {layer_name: [B, C_s, H, W]}
            original_teacher_features: {layer_name: {teacher_name: [B, C_t, H, W]}}
            device: 设备

        Returns:
            total_loss: 总重构损失
            loss_dict: 每层的损失字典
        """
        if not reconstructed_features or not original_teacher_features:
            return torch.tensor(0.0, device=device), {}

        total_rec_loss = torch.tensor(0.0, device=device)
        rec_loss_dict = {}
        num_layers = 0

        for layer_name, recon_feat in reconstructed_features.items():
            if layer_name not in original_teacher_features:
                continue

            # ✅ 获取该层的教师特征字典
            teacher_feats_dict = original_teacher_features[layer_name]

            if not teacher_feats_dict:
                continue

            layer_loss = torch.tensor(0.0, device=device)
            num_teachers = 0

            # ✅ 遍历每个教师的特征
            for teacher_name, orig_feat in teacher_feats_dict.items():
                # 空间对齐
                if recon_feat.shape[-2:] != orig_feat.shape[-2:]:
                    orig_feat = F.interpolate(
                        orig_feat,
                        size=recon_feat.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )

                # ✅ 通道对齐
                if recon_feat.shape[1] != orig_feat.shape[1]:
                    proj_key = f'_recon_proj_{layer_name}_{teacher_name}'
                    if not hasattr(self, proj_key):
                        proj = nn.Conv2d(
                            recon_feat.shape[1],
                            orig_feat.shape[1],
                            kernel_size=1,
                            bias=False
                        ).to(device)
                        setattr(self, proj_key, proj)

                    proj = getattr(self, proj_key)
                    recon_feat_aligned = proj(recon_feat)
                else:
                    recon_feat_aligned = recon_feat

                # ✅ 计算 MSE 损失
                teacher_loss = F.mse_loss(recon_feat_aligned, orig_feat.detach())
                layer_loss += teacher_loss
                num_teachers += 1

            if num_teachers > 0:
                layer_loss = layer_loss / num_teachers
                total_rec_loss += layer_loss
                rec_loss_dict[layer_name] = layer_loss.item()
                num_layers += 1

        if num_layers > 0:
            total_rec_loss = total_rec_loss / num_layers

        return total_rec_loss, rec_loss_dict

    def _compute_fixed_weighted_loss(self, losses):
        """使用固定权重计算总损失"""
        return (
                self.task_weight * losses.get('task', 0) +
                self.crd_weight * losses.get('crd', 0) +
                self.output_distill_weight * losses.get('output', 0) +
                self.feature_distill_weight * losses.get('feature', 0) +
                self.recon_weight * losses.get('recon', 0)
        )


# ============================================
# 5. 数据集类
# ============================================

class SegmentationDataset(Dataset):
    """分割任务数据集"""

    def __init__(self, image_paths, mask_paths, transform=None, img_size=224):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.transform = transform

        if not self.transform:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图像
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        # 调整大小
        image = image.resize((self.img_size, self.img_size))
        mask = mask.resize((self.img_size, self.img_size))

        # 应用变换
        image = self.transform(image)
        mask = torch.from_numpy(np.array(mask) / 255.0).float().unsqueeze(0)

        return image, mask


class ClassificationDataset(Dataset):
    """分类/识别任务数据集"""

    def __init__(self, image_paths, labels, transform=None, img_size=224):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
        self.transform = transform

        if not self.transform:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图像
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        # 应用变换
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# ============================================
# 6. 早停机制
# ============================================

class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_improvement(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


# ============================================
# 7. 完整的训练器类
# ============================================

class MultiTeacherDistillationTrainer:
    """完整的多教师知识蒸馏训练器"""

    def __init__(
            self,
            student_model: nn.Module,
            teacher_models: Dict[str, nn.Module],
            train_loaders: Dict[str, DataLoader],
            val_loaders: Dict[str, DataLoader],
            config: Dict,
            device: torch.device,
            save_dir: str = "./experiments",
            verbose: bool = True
    ):
        """
        多教师蒸馏训练器初始化（完整修复版 - 支持损失级门控）

        Args:
            student_model: 学生模型
            teacher_models: 教师模型字典 {teacher_name: model}
            train_loaders: 训练数据加载器 {dataset_name: loader}
            val_loaders: 验证数据加载器 {dataset_name: loader}
            config: 配置字典
            device: 训练设备
            save_dir: 保存目录
            verbose: 是否打印详细信息
        """
        # ========== 1. 基础设置 ==========
        self.student_model = student_model.to(device)
        self.teacher_models = {k: v.to(device).eval() for k, v in teacher_models.items()}
        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.config = config
        self.device = device
        self.verbose = verbose
        self.train_history = {
            'train_losses': [],
            'val_metrics': [],
            'learning_rates': [],
            'teacher_weights': []
        }

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Initializing Multi-Teacher Distillation Trainer")
            print(f"{'=' * 80}")
            print(f"Student Model: {student_model.__class__.__name__}")
            print(f"Teachers: {list(teacher_models.keys())}")
            print(f"Datasets: {list(train_loaders.keys())}")

        # ========== 2. 创建保存目录 ==========
        self.experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = Path(save_dir) / self.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"\nExperiment Directory: {self.save_dir}")

        # 保存配置
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        # ========== 3. 获取学生特征维度 ==========
        if hasattr(student_model, 'feature_dims'):
            student_dim = student_model.feature_dims.get('final', 1280)
            if verbose:
                print(f"\n{'─' * 80}")
                print(f"Student Feature Dimensions:")
                print(f"{'─' * 80}")
                for layer, dim in student_model.feature_dims.items():
                    print(f"  {layer:15s}: {dim}")
        else:
            student_dim = 1280
            if verbose:
                print(f"\n⚠️  Using default student dimension: {student_dim}")

        # ========== 4. 任务映射（提前定义）==========
        self.task_mapping = {
            'BUSI': ('segmentation_busi', 'USFM'),
            'kvasir_seg': ('segmentation_kvasir', 'MedSAM'),
            'APTOS2019': ('classification', 'RETFound'),
            'ISIC2017': ('recognition', 'BioMedPhrase')
        }

        if verbose:
            print(f"\n{'─' * 80}")
            print(f"Task Mapping:")
            print(f"{'─' * 80}")
            for dataset, (task_type, teacher) in self.task_mapping.items():
                print(f"  {dataset:15s} -> {task_type:25s} (Teacher: {teacher})")

        # ========== 5. 初始化跨任务知识蒸馏器（包含 Return Decoder）==========
        if verbose:
            print(f"\n{'─' * 80}")
            print(f"Initializing Cross-Task Knowledge Distiller (with Return Decoder)...")
            print(f"{'─' * 80}")

        self.cross_task_distiller = self._create_cross_task_distiller()

        if self.cross_task_distiller is not None and verbose:
            # 统计 Return Decoder 的参数
            total_projector_params = 0
            for teacher_name, projectors in self.cross_task_distiller.projectors.items():
                teacher_params = sum(p.numel() for proj in projectors.values()
                                     for p in proj.parameters())
                total_projector_params += teacher_params
                print(f"  {teacher_name}: {teacher_params:,} parameters")
            print(f"  Total Projector Parameters: {total_projector_params:,}")

        # ========== 6. 初始化教师级门控网络 ==========
        if verbose:
            print(f"\n{'─' * 80}")
            print(f"Initializing Teacher-Level Gating...")
            print(f"{'─' * 80}")

        self.teacher_gating = TeacherLevelGating(
            student_dim=student_dim,
            num_teachers=len(teacher_models),
            num_tasks=len(train_loaders),
            ema_momentum=config.get('ema_momentum', 0.9)
        ).to(device)

        if verbose:
            gating_params = sum(p.numel() for p in self.teacher_gating.parameters())
            print(f"  Gating Parameters: {gating_params:,}")

        # ========== 7. 初始化重构特征融合权重 ==========
        if verbose:
            print(f"\n{'─' * 80}")
            print(f"Initializing Reconstruction Fusion Weight...")
            print(f"{'─' * 80}")

        self.reconstruction_weight = nn.Parameter(
            torch.tensor(config.get('reconstruction_fusion_weight', 0.3), device=device),
            requires_grad=True
        )

        if verbose:
            print(f"  Initial Fusion Weight: {self.reconstruction_weight.item():.4f}")
            print(f"  (Controls balance between aligned and reconstructed features)")

        # ========== 8. 初始化特征融合模块 ==========
        if verbose:
            print(f"\n{'─' * 80}")
            print(f"Initializing Feature Fusion Module...")
            print(f"{'─' * 80}")

        self.feature_fusion = nn.ModuleDict()
        fusion_layers = ['stem', 'block2', 'block3', 'block5', 'final']

        for layer_name in fusion_layers:
            if hasattr(student_model, 'feature_dims') and layer_name in student_model.feature_dims:
                layer_dim = student_model.feature_dims[layer_name]

                # 简单的融合模块：1x1卷积 + BN + ReLU
                self.feature_fusion[layer_name] = nn.Sequential(
                    nn.Conv2d(layer_dim, layer_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(layer_dim),
                    nn.ReLU(inplace=True)
                ).to(device)

                if verbose:
                    print(f"  {layer_name:15s}: fusion module created (dim={layer_dim})")

        if verbose:
            fusion_params = sum(p.numel() for p in self.feature_fusion.parameters())
            print(f"  Total Fusion Parameters: {fusion_params:,}")

        # ========== 9. 初始化损失函数（✅ 包含损失级门控）==========
        if verbose:
            print(f"\n{'─' * 80}")
            print(f"Initializing Loss Functions (with Loss-Level Gating)...")
            print(f"{'─' * 80}")

        self.distillation_loss = HeterogeneousDistillationLoss(
            num_tasks=len(train_loaders),
            task_weight=config.get('task_weight', 1.0),
            output_distill_weight=config.get('output_weight', 0.5),
            feature_distill_weight=config.get('feature_weight', 0.3),
            crd_weight=config.get('crd_weight', 0.5),
            recon_weight=config.get('recon_weight', 0.2),
            temperature=config.get('temperature', 1.5),
            crd_temperature=config.get('crd_temperature', 0.07),
            use_loss_gating=config.get('use_loss_gating', True),  # ✅ 启用损失级门控
            student_dim=student_dim  # ✅ 传递学生维度
        ).to(device)

        if verbose:
            print(f"  Temperature: {config.get('temperature', 1.5)}")
            print(f"  Task Weight: {config.get('task_weight', 1.0)}")
            print(f"  Output Distill Weight: {config.get('output_weight', 0.5)}")
            print(f"  Feature Distill Weight: {config.get('feature_weight', 0.3)}")
            print(f"  CRD Weight: {config.get('crd_weight', 0.5)}")
            print(f"  Reconstruction Weight: {config.get('recon_weight', 0.2)}")
            print(f"  Loss-Level Gating: {'✅ Enabled' if config.get('use_loss_gating', True) else '❌ Disabled'}")

        # ========== 10. 初始化优化器 ==========
        if verbose:
            print(f"\n{'─' * 80}")
            print(f"Initializing Optimizer...")
            print(f"{'─' * 80}")

        self.optimizer = self._create_optimizer()

        # ✅ 将重构融合权重添加到优化器
        self.optimizer.add_param_group({
            'params': [self.reconstruction_weight],
            'lr': config.get('learning_rate', 1e-4) * 0.1,
            'weight_decay': 0.0
        })

        if verbose:
            print(f"  Base Learning Rate: {config.get('learning_rate', 1e-4)}")
            print(f"  Fusion Weight LR: {config.get('learning_rate', 1e-4) * 0.1}")
            print(f"  Optimizer: {self.optimizer.__class__.__name__}")

            # 统计总参数量
            total_params = sum(p.numel() for p in self.student_model.parameters())
            trainable_params = sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)

            # ✅ 添加门控网络和融合模块的参数
            gating_params = sum(p.numel() for p in self.teacher_gating.parameters())
            fusion_params = sum(p.numel() for p in self.feature_fusion.parameters())
            loss_gating_params = 0
            if hasattr(self.distillation_loss, 'loss_gate') and self.distillation_loss.loss_gate is not None:
                loss_gating_params = sum(p.numel() for p in self.distillation_loss.loss_gate.parameters())

            print(f"\n  Parameter Statistics:")
            print(f"    Student Model: {trainable_params:,}")
            print(f"    Teacher Gating: {gating_params:,}")
            print(f"    Loss Gating: {loss_gating_params:,}")
            print(f"    Feature Fusion: {fusion_params:,}")
            print(f"    Reconstruction Weight: 1")
            print(f"    Total Trainable: {trainable_params + gating_params + loss_gating_params + fusion_params + 1:,}")

        # ========== 11. 初始化学习率调度器 ==========
        self.scheduler = self._create_scheduler()

        if verbose and self.scheduler is not None:
            print(f"\n  Scheduler: {self.scheduler.__class__.__name__}")

        # ========== 12. TensorBoard 初始化 ==========
        self.writer = SummaryWriter(self.save_dir / "tensorboard")

        if verbose:
            print(f"\n{'─' * 80}")
            print(f"TensorBoard:")
            print(f"{'─' * 80}")
            print(f"  Log Directory: {self.save_dir / 'tensorboard'}")

        # ========== 13. 早停机制 ==========
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 15),
            min_delta=config.get('early_stopping_delta', 0.001)
        )

        if verbose:
            print(f"\n{'─' * 80}")
            print(f"Early Stopping:")
            print(f"{'─' * 80}")
            print(f"  Patience: {config.get('early_stopping_patience', 15)} epochs")
            print(f"  Min Delta: {config.get('early_stopping_delta', 0.001)}")

        # ========== 14. 最佳模型跟踪 ==========
        self.best_metrics = {}
        self.best_epoch = 0

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Initialization Complete!")
            print(f"{'=' * 80}\n")

            # 打印总结
            print(f"Summary:")
            print(f"  - Student Model: {student_model.__class__.__name__}")
            print(f"  - Number of Teachers: {len(teacher_models)}")
            print(f"  - Number of Tasks: {len(train_loaders)}")
            print(f"  - Return Decoder: ✅ Enabled")
            print(f"  - Teacher-Level Gating: ✅ Enabled")
            print(f"  - Loss-Level Gating: {'✅ Enabled' if config.get('use_loss_gating', True) else '❌ Disabled'}")
            print(f"  - Reconstruction Fusion Weight: {self.reconstruction_weight.item():.4f}")
            print(
                f"  - Total Trainable Parameters: {trainable_params + gating_params + loss_gating_params + fusion_params + 1:,}")
            print(f"  - Experiment Directory: {self.save_dir}")
            print(f"\n{'=' * 80}\n")

    def train_unified(self, dataset_names):
        """
        统一模型训练：在一个 Epoch 内交替训练所有任务
        """
        print(f"\n🚀 Starting unified training for {len(dataset_names)} tasks")

        for epoch in range(self.config['num_epochs']):
            print(f"\n{'=' * 80}")
            print(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
            print(f"{'=' * 80}")

            self.student_model.train()
            epoch_losses = {name: [] for name in dataset_names}

            # 任务交替训练策略
            for task_id, dataset_name in enumerate(dataset_names):
                print(f"\n📊 Training on task: {dataset_name} (Task ID: {task_id})")

                train_loader = self.train_loaders[dataset_name]
                pbar = tqdm(train_loader, desc=f"  {dataset_name}")

                for batch_idx, batch in enumerate(pbar):
                    # 解包数据（根据你的 DataLoader 返回格式调整）
                    if len(batch) == 2:
                        images, targets = batch
                    else:
                        images, targets, _ = batch

                    images = images.to(self.device)
                    targets = targets.to(self.device)

                    # 前向传播（传入 task_id 以触发任务特定的门控）
                    loss = self._compute_loss(
                        images,
                        targets,
                        dataset_name=dataset_name,
                        task_id=task_id
                    )

                    # 反向传播
                    self.optimizer.zero_grad()
                    loss.backward()

                    # 梯度裁剪（防止梯度爆炸）
                    torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)

                    self.optimizer.step()

                    # 记录损失
                    epoch_losses[dataset_name].append(loss.item())
                    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            # Epoch 结束后的验证和保存
            avg_losses = {name: np.mean(losses) for name, losses in epoch_losses.items()}
            print(f"\n📈 Epoch {epoch + 1} Average Losses:")
            for name, loss in avg_losses.items():
                print(f"   {name}: {loss:.4f}")

            # 验证所有任务
            val_results = self._validate_all_tasks(dataset_names)

            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()

            # 保存检查点
            self._save_checkpoint(epoch, avg_losses, val_results)

            # 早停检查
            avg_val_loss = np.mean(list(val_results.values()))
            if self._check_early_stopping(avg_val_loss):
                print(f"\n⚠️ Early stopping triggered at epoch {epoch + 1}")
                break

    def _validate_all_tasks(self, dataset_names):
        """验证所有任务"""
        self.student_model.eval()
        val_losses = {}

        with torch.no_grad():
            for task_id, dataset_name in enumerate(dataset_names):
                val_loader = self.val_loaders[dataset_name]
                losses = []

                for batch in val_loader:
                    if len(batch) == 2:
                        images, targets = batch
                    else:
                        images, targets, _ = batch

                    images = images.to(self.device)
                    targets = targets.to(self.device)

                    loss = self._compute_loss(
                        images,
                        targets,
                        dataset_name=dataset_name,
                        task_id=task_id
                    )
                    losses.append(loss.item())

                val_losses[dataset_name] = np.mean(losses)
                print(f"   Val {dataset_name}: {val_losses[dataset_name]:.4f}")

        return val_losses

    def _compute_loss(self, images, targets, dataset_name, task_id):
        """
        计算完整的损失（任务损失 + 蒸馏损失）
        """
        # 1. 完整的前向传播
        outputs = self._forward_pass(
            images=images,
            targets=targets,
            dataset_name=dataset_name,
            task_id=task_id,
            verbose=False
        )

        # 2. 计算所有损失
        total_loss, loss_dict = self.distillation_loss(
            outputs=outputs,
            targets=targets,
            dataset_name=dataset_name,
            task_id=task_id,
            verbose=False
        )

        return total_loss

    def _get_task_type(self, dataset_name):
        """根据数据集名称返回任务类型"""
        task_mapping = {
            'BUSI': 'segmentation',
            'kvasir_seg': 'segmentation',
            'ISIC2017': 'recognition',
            'APTOS2019': 'classification'
        }
        return task_mapping.get(dataset_name, 'classification')

    def _create_optimizer(self):
        """创建优化器（使用差异化学习率）"""
        param_groups = [
            {'params': self.student_model.backbone.parameters(),
             'lr': self.config['learning_rate'] * 0.1},
            {'params': self.student_model.task_decoders.parameters(),
             'lr': self.config['learning_rate']},
            {'params': self.student_model.task_selector.parameters(),
             'lr': self.config['learning_rate']},
            # ========== 🆕 添加门控网络参数 ==========
            {'params': self.teacher_gating.parameters(),
             'lr': self.config['learning_rate'] * 0.5}  # 门控网络使用较低学习率
        ]

        if self.config.get('optimizer', 'adam') == 'adam':
            return torch.optim.Adam(param_groups, weight_decay=1e-5)
        elif self.config['optimizer'] == 'adamw':
            return torch.optim.AdamW(param_groups, weight_decay=0.01)
        else:
            return torch.optim.SGD(param_groups, momentum=0.9, weight_decay=1e-4)

    def _create_scheduler(self):
        """创建学习率调度器"""
        scheduler_type = self.config.get('scheduler', 'cosine')

        if scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config['num_epochs'], eta_min=1e-6
            )
        elif scheduler_type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5, factor=0.5
            )
        else:
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )

    def _create_cross_task_distiller(self):
        """创建跨任务蒸馏器（修复版）"""

        # ============ 核心修复：自动提取教师维度 ============
        print("\n" + "=" * 80)
        print("Extracting teacher model dimensions...")
        print("=" * 80)

        # 创建样本输入
        sample_input = torch.randn(1, 3, 224, 224).to(self.device)

        # 使用辅助函数自动提取维度
        teacher_dims = {}
        for teacher_name, teacher_model in self.teacher_models.items():
            print(f"\nProcessing teacher: {teacher_name}")
            teacher_model.eval()

            with torch.no_grad():
                try:
                    # 尝试提取特征
                    if teacher_name == 'MedSAM':
                        # MedSAM 特殊处理
                        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
                        denorm_img = sample_input * std + mean
                        denorm_img = denorm_img * 255.0
                        numpy_img = denorm_img.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

                        medsam_input = [{"image": numpy_img[0]}]
                        features = teacher_model.extract_features(medsam_input)

                        if features is not None:
                            if features.dim() == 4:  # [B, C, H, W]
                                dim = features.size(1)
                            elif features.dim() == 3:  # [B, N, C]
                                dim = features.size(-1)
                            else:
                                dim = 256  # 默认值

                            teacher_dims[teacher_name] = {
                                'stem': dim, 'block2': dim, 'block3': dim,
                                'block5': dim, 'final': dim
                            }
                            print(f"  ✅ Extracted dimension: {dim}")
                        else:
                            raise ValueError("MedSAM features is None")

                    elif teacher_name == 'RETFound_MAE':
                        # RETFound_MAE 处理
                        img_224 = F.interpolate(sample_input, size=(224, 224), mode='bilinear')
                        result = teacher_model(img_224)

                        # 处理返回值
                        if isinstance(result, tuple):
                            features = result[1] if len(result) > 1 else result[0]
                        else:
                            features = result

                        if isinstance(features, torch.Tensor):
                            if features.dim() == 3:  # [B, N, C]
                                dim = features.size(-1)
                            elif features.dim() == 4:  # [B, C, H, W]
                                dim = features.size(1)
                            else:
                                dim = 768

                            teacher_dims[teacher_name] = {
                                'stem': dim, 'block2': dim, 'block3': dim,
                                'block5': dim, 'final': dim
                            }
                            print(f"  ✅ Extracted dimension: {dim}")
                        else:
                            raise ValueError("RETFound_MAE features is not a tensor")

                    elif teacher_name == 'USFM':
                        # USFM 处理
                        output = teacher_model(sample_input)

                        if hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'encoder'):
                            features = teacher_model.model.encoder(sample_input)
                            if features.dim() == 4:
                                dim = features.size(1)
                            else:
                                dim = 256
                        else:
                            dim = 256

                        teacher_dims[teacher_name] = {
                            'stem': dim, 'block2': dim * 2, 'block3': dim * 2,
                            'block5': dim * 4, 'final': dim * 8
                        }
                        print(f"  ✅ Extracted dimensions: {teacher_dims[teacher_name]}")

                    elif teacher_name == 'BioMedPrase':
                        # BioMedPrase 处理
                        try:
                            output = teacher_model(sample_input)
                            if isinstance(output, torch.Tensor):
                                if output.dim() == 4:
                                    dim = output.size(1)
                                elif output.dim() == 3:
                                    dim = output.size(-1)
                                else:
                                    dim = 512
                            else:
                                dim = 512
                        except:
                            dim = 512

                        teacher_dims[teacher_name] = {
                            'stem': dim, 'block2': dim, 'block3': dim,
                            'block5': dim, 'final': dim
                        }
                        print(f"  ✅ Extracted dimension: {dim}")

                    else:
                        # 默认处理
                        output = teacher_model(sample_input)
                        if isinstance(output, torch.Tensor):
                            if output.dim() == 4:
                                dim = output.size(1)
                            elif output.dim() == 3:
                                dim = output.size(-1)
                            else:
                                dim = 512
                        else:
                            dim = 512

                        teacher_dims[teacher_name] = {
                            'stem': dim, 'block2': dim, 'block3': dim,
                            'block5': dim, 'final': dim
                        }
                        print(f"  ✅ Extracted dimension: {dim}")

                except Exception as e:
                    print(f"  ⚠️  Error extracting {teacher_name}: {e}")
                    # 使用默认维度
                    teacher_dims[teacher_name] = {
                        'stem': 256, 'block2': 512, 'block3': 512,
                        'block5': 1024, 'final': 2048
                    }
                    print(f"  ℹ️  Using default dimensions")

        # 使用学生模型的实际特征维度
        student_dims = self.student_model.feature_dims if hasattr(self.student_model, 'feature_dims') else {
            'default': 1280, 'stem': 32, 'block1': 16,
            'block2': 24, 'block3': 40, 'block4': 80,
            'block5': 112, 'block6': 192, 'block7': 320,
            'final': 1280
        }

        print("\n" + "=" * 80)
        print("Creating CrossTaskKnowledgeDistiller:")
        print(f"  Student dims: {student_dims}")
        print(f"  Teacher dims: {teacher_dims}")
        print("=" * 80 + "\n")

        # 创建蒸馏器
        distiller = CrossTaskKnowledgeDistiller(
            student_dims,
            teacher_dims,
            verbose=self.verbose  # ✅ 传递 verbose
        ).to(self.device)

        # ============ 验证初始化 ============
        print("\nValidating distiller initialization...")

        # 创建测试特征
        test_student_features = {
            'stem': torch.randn(2, student_dims['stem'], 112, 112).to(self.device),
            'block2': torch.randn(2, student_dims['block2'], 56, 56).to(self.device),
            'block3': torch.randn(2, student_dims['block3'], 28, 28).to(self.device),
            'block5': torch.randn(2, student_dims['block5'], 14, 14).to(self.device),
            'final': torch.randn(2, student_dims['final'], 7, 7).to(self.device)
        }

        test_teacher_features = {}
        for t_name, t_dims in teacher_dims.items():
            test_teacher_features[t_name] = {
                layer: torch.randn(2, dim, 14, 14).to(self.device)
                for layer, dim in t_dims.items()
                if layer in test_student_features
            }

        # 运行验证
        validate_distiller_initialization(
            distiller,
            test_student_features,
            test_teacher_features
        )

        print("✅ Distiller initialization validated!\n")

        return distiller

    def train(self, dataset_name: str):
        """针对单个数据集的训练流程"""
        print(f"\n{'=' * 80}")
        print(f"Starting training for dataset: {dataset_name}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Training for {self.config['num_epochs']} epochs")
        print("-" * 80)

        # 打印加载的教师模型信息
        print(f"\nLoaded {len(self.teacher_models)} teacher models:")
        for teacher_name in self.teacher_models.keys():
            print(f"  - {teacher_name}")
        print()

        for epoch in range(self.config['num_epochs']):
            # 训练阶段（只训练当前数据集）
            train_losses = self._train_epoch(epoch, dataset_name)

            # 验证阶段（只验证当前数据集）
            val_metrics = self._validate_epoch(epoch, dataset_name)

            # 更新学习率
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                avg_loss = train_losses['total']
                self.scheduler.step(avg_loss)
            else:
                self.scheduler.step()

            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.train_history['learning_rates'].append(current_lr)

            # 打印epoch总结
            self._print_epoch_summary(epoch, train_losses, val_metrics, current_lr, dataset_name)

            # 保存检查点
            self._save_checkpoint(epoch, val_metrics, dataset_name)

            # 早停检查
            if dataset_name in ['BUSI', 'kvasir_seg']:
                metric = val_metrics['IoU']
            else:
                metric = val_metrics['Accuracy']

            self.early_stopping(metric)
            if self.early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        # 训练结束后的操作
        self._on_training_end(dataset_name)

    def _train_epoch(self, epoch: int, dataset_name: str) -> Dict[str, float]:

        self.student_model.train()

        # ========== 设置门控网络为训练模式 ==========
        self.teacher_gating.train()
        if hasattr(self.distillation_loss, 'out_loss'):
            self.distillation_loss.out_loss.update_temperature(
                epoch,
                self.config['num_epochs']
            )


            current_temp = self.distillation_loss.out_loss.current_temp
            print(f"[Epoch {epoch}] Output Distillation Temperature: {current_temp:.3f}")

        loader = self.train_loaders[dataset_name]
        print(f"\n{'=' * 80}")
        print(f"Training on dataset: {dataset_name} | Epoch: {epoch + 1}")
        print(f"{'=' * 80}")

        # 🔧 修改1：初始化损失统计（添加 recon 损失）
        losses = {
            'total': 0,
            'task': 0,
            'output_distill': 0,
            'feature_distill': 0,
            'recon': 0  # ← 添加重构损失
        }
        num_batches = 0

        # ========== 获取任务ID映射 ==========
        dataset_to_task_id = {
            'BUSI': 0,
            'kvasir_seg': 0,
            'APTOS2019': 1,
            'ISIC2017': 2
        }
        task_id = dataset_to_task_id.get(dataset_name, 0)

        # 创建进度条
        pbar = tqdm(loader, desc=f'Training {dataset_name}')

        # ========== 用于统计教师权重 ==========
        teacher_weights_accumulator = []

        for batch_idx, (images, targets_list) in enumerate(pbar):
            try:
                # ========== 1. 数据准备 ==========
                images = images.to(self.device)

                # 处理不同格式的targets
                if isinstance(targets_list, (list, tuple)):
                    if len(targets_list) == 1:
                        targets = targets_list[0].to(self.device)
                    else:
                        targets = [t.to(self.device) if isinstance(t, torch.Tensor) else t
                                   for t in targets_list]
                else:
                    targets = targets_list.to(self.device)

                # 检查数据有效性
                if images.size(0) == 0:
                    print(f"⚠️  Warning: Empty batch at index {batch_idx}")
                    continue

                if torch.isnan(images).any() or torch.isinf(images).any():
                    print(f"⚠️  Warning: Invalid values in images at batch {batch_idx}")
                    continue

                # 🔧 修改2：前向传播时使用 verbose 属性
                verbose_flag = (batch_idx % 100 == 0) and getattr(self, 'verbose', False)

                outputs = self._forward_pass(
                    images=images,
                    targets=targets,
                    dataset_name=dataset_name,
                    task_id=task_id,
                    verbose=verbose_flag  # ← 使用 verbose_flag
                )

                # 检查前向传播输出
                if outputs is None or outputs.get('student_output') is None:
                    print(f"⚠️  Warning: Invalid forward pass output at batch {batch_idx}")
                    self.optimizer.zero_grad()
                    continue

                # 🔧 修改3：使用 self.distillation_loss 而不是 self.criterion
                loss_dict = self.distillation_loss(
                    outputs=outputs,
                    targets=targets,
                    dataset_name=dataset_name,
                    task_id=task_id,  # ← 添加这一行
                    verbose=verbose_flag  # ← 传递 verbose 标志
                )

                # 检查损失有效性
                if loss_dict is None or torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"⚠️  Warning: Invalid loss at batch {batch_idx}")
                    print(f"   Loss dict: {loss_dict}")
                    self.optimizer.zero_grad()
                    continue

                # ✅ 修复后
                total_loss, loss_dict = self.distillation_loss(
                    outputs=outputs,
                    targets=targets,
                    dataset_name=dataset_name,
                    task_id=task_id,
                    verbose=verbose_flag
                )

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"⚠️  Warning: NaN/Inf loss at batch {batch_idx}")
                    print(f"   Loss dict: {loss_dict}")
                    self.optimizer.zero_grad()
                    continue

                # ========== 4. 反向传播和优化 ==========
                self.optimizer.zero_grad()
                total_loss.backward()

                # 🔧 修改4：梯度裁剪（添加损失级门控网络）
                torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(),
                    max_norm=1.0
                )
                torch.nn.utils.clip_grad_norm_(
                    self.teacher_gating.parameters(),
                    max_norm=1.0
                )
                # 裁剪损失级门控网络的梯度
                if hasattr(self.distillation_loss, 'loss_gate'):
                    torch.nn.utils.clip_grad_norm_(
                        self.distillation_loss.loss_gate.parameters(),
                        max_norm=1.0
                    )

                self.optimizer.step()

                # 🔧 修改5：安全地累积损失统计
                for key in losses.keys():
                    if key in loss_dict:
                        loss_value = loss_dict[key]
                        if isinstance(loss_value, torch.Tensor):
                            losses[key] += loss_value.item()
                        else:
                            losses[key] += loss_value
                num_batches += 1

                # ========== 6. 收集教师权重（用于统计） ==========
                if outputs.get('teacher_weights') is not None:
                    teacher_weights_accumulator.append(
                        outputs['teacher_weights'].detach().cpu()
                    )

                # 🔧 修改6：更新进度条（添加 recon 损失，只显示非零项）
                if num_batches > 0:
                    avg_loss = losses['total'] / num_batches
                    postfix_dict = {
                        'loss': f'{avg_loss:.4f}',
                        'task': f'{losses["task"] / num_batches:.4f}',
                    }

                    # 只显示非零的损失项
                    if losses["output_distill"] > 0:
                        postfix_dict['out_dist'] = f'{losses["output_distill"] / num_batches:.4f}'
                    if losses["feature_distill"] > 0:
                        postfix_dict['feat_dist'] = f'{losses["feature_distill"] / num_batches:.4f}'
                    if losses["recon"] > 0:
                        postfix_dict['recon'] = f'{losses["recon"] / num_batches:.4f}'

                    pbar.set_postfix(postfix_dict)

                # ========== 8. 定期打印教师权重统计 ==========
                if batch_idx > 0 and batch_idx % 100 == 0:
                    stats = self.get_teacher_weights_stats()
                    if stats and 'mean' in stats:
                        print(f"\n{'─' * 60}")
                        print(f"[Batch {batch_idx}] Teacher Weights Statistics:")
                        teacher_names = list(self.teacher_models.keys())
                        for i, name in enumerate(teacher_names):
                            if i < len(stats['mean']):
                                print(f"  {name:20s}: {stats['mean'][i]:.3f} ± {stats['std'][i]:.3f}")
                        print(f"{'─' * 60}\n")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠️  GPU OOM at batch {batch_idx}! Clearing cache...")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                    continue
                else:
                    print(f"\n❌ Runtime error at batch {batch_idx}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    self.optimizer.zero_grad()
                    continue

            except Exception as e:
                print(f"\n❌ Unexpected error at batch {batch_idx}: {str(e)}")
                import traceback
                traceback.print_exc()
                self.optimizer.zero_grad()
                continue

        # ========== 9. 计算epoch平均损失 ==========
        if num_batches == 0:
            print(f"⚠️  Warning: No valid batches in epoch {epoch + 1}")
            return {k: 0.0 for k in losses.keys()}

        epoch_losses = {k: v / num_batches for k, v in losses.items()}

        # 🔧 修改7：打印epoch总结（添加 recon 损失）
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1} Training Summary for {dataset_name}:")
        print(f"{'─' * 80}")
        print(f"  Total Loss:           {epoch_losses['total']:.4f}")
        print(f"  Task Loss:            {epoch_losses['task']:.4f}")
        print(f"  Output Distill Loss:  {epoch_losses['output_distill']:.4f}")
        print(f"  Feature Distill Loss: {epoch_losses['feature_distill']:.4f}")
        print(f"  Recon Loss:           {epoch_losses['recon']:.4f}")  # ← 添加
        print(f"  Valid Batches:        {num_batches}/{len(loader)}")

        # ========== 11. 打印整个epoch的教师权重统计 ==========
        if len(teacher_weights_accumulator) > 0:
            all_weights = torch.cat(teacher_weights_accumulator, dim=0)
            mean_weights = all_weights.mean(dim=0)
            std_weights = all_weights.std(dim=0)

            print(f"\n{'─' * 80}")
            print(f"Epoch {epoch + 1} Teacher Weights Summary:")
            print(f"{'─' * 80}")
            teacher_names = list(self.teacher_models.keys())
            for i, name in enumerate(teacher_names):
                if i < len(mean_weights):
                    print(f"  {name:20s}: {mean_weights[i]:.3f} ± {std_weights[i]:.3f}")
            print(f"{'─' * 80}")

        print(f"{'=' * 80}\n")

        return epoch_losses

    def _forward_pass(self, images, targets, dataset_name, task_id, verbose=False):
        """
        执行一次完整的前向传播

        Args:
            images: 输入图像
            targets: 目标标签
            dataset_name: 数据集名称
            task_id: 任务ID
            verbose: 是否打印详细信息

        Returns:
            outputs: 包含所有中间结果的字典
        """
        device = next(self.student_model.parameters()).device
        images = images.to(device)

        if verbose:
            print(f"\n{'=' * 100}")
            print(f"[Forward Pass] Dataset: {dataset_name}, Task: {task_id}")
            print(f"{'=' * 100}")

        # ========== 1. 学生模型前向传播 ==========
        if verbose:
            print(f"\n[Step 1] Student Model Forward:")

        student_features = {}

        def student_hook(module, input, output, name):
            student_features[name] = output

        # 注册钩子
        hooks = []
        for name, module in self.student_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                if any(keyword in name for keyword in ['layer1', 'layer2', 'layer3', 'layer4', 'backbone']):
                    hook = module.register_forward_hook(
                        lambda m, i, o, n=name: student_hook(m, i, o, n)
                    )
                    hooks.append(hook)

        # 前向传播
        # ✅ 正确调用学生模型（传入 task_id）
        if dataset_name in ['coco', 'voc']:
            # 检测任务
            if self.training:
                predictions, student_features = self.student_model(
                    images,
                    dataset_name=dataset_name,
                    task_id=task_id
                )
            else:
                predictions, student_features = self.student_model(
                    images,
                    dataset_name=dataset_name,
                    task_id=task_id
                )
        else:
            # 分类/分割任务
            predictions, student_features = self.student_model(
                images,
                dataset_name=dataset_name,
                task_id=task_id
            )

        # ✅ 移除钩子逻辑（因为 student_model.forward 已经返回特征）
        # 删除第 3040-3060 行的钩子代码

        # 移除钩子
        for hook in hooks:
            hook.remove()

        if verbose:
            print(f"  Student features: {len(student_features)} layers")
            print(f"  Feature shapes: {[(k, v.shape) for k, v in list(student_features.items())[:3]]}")

        # ========== 2. 教师模型前向传播 ==========
        if verbose:
            print(f"\n[Step 2] Teacher Models Forward:")

        all_teacher_features = {}
        teacher_names_list = []

        for teacher_name, teacher_model in self.teacher_models.items():
            if teacher_model is None:
                continue

            teacher_features = {}

            def teacher_hook(module, input, output, name):
                teacher_features[name] = output

            # 注册钩子
            hooks = []
            for name, module in teacher_model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                    if any(keyword in name for keyword in ['layer1', 'layer2', 'layer3', 'layer4', 'backbone']):
                        hook = module.register_forward_hook(
                            lambda m, i, o, n=name: teacher_hook(m, i, o, n)
                        )
                        hooks.append(hook)

            # 前向传播
            with torch.no_grad():
                if dataset_name in ['coco', 'voc']:
                    _ = teacher_model(images)
                else:
                    _ = teacher_model(images)

            # 移除钩子
            for hook in hooks:
                hook.remove()

            all_teacher_features[teacher_name] = teacher_features
            teacher_names_list.append(teacher_name)

            if verbose:
                print(f"  {teacher_name}: {len(teacher_features)} layers")

        # ========== 3. 跨任务蒸馏（✅ 修复：接收原始特征）==========
        if verbose:
            print(f"\n[Step 3] Cross-Task Distillation:")

        aligned_features = {}
        reconstructed_features = {}
        teacher_weights_log = {}
        original_teacher_features = {}  # ✅ 新增

        if self.cross_task_distiller is not None and teacher_names_list:
            try:
                # ✅ 接收四个返回值
                aligned_features, reconstructed_features, teacher_weights_log, original_teacher_features = \
                    self.cross_task_distiller(
                        student_features=student_features,
                        teacher_features=all_teacher_features,
                        active_teachers=teacher_names_list,
                        task_id=task_id
                    )

                if verbose:
                    print(f"  Aligned layers: {len(aligned_features)}")
                    print(f"  Reconstructed layers: {len(reconstructed_features)}")
                    print(f"  Original teacher features: {len(original_teacher_features)}")  # ✅ 新增
                    print(f"  Teacher weights: {teacher_weights_log}")

            except Exception as e:
                if verbose:
                    print(f"⚠️  Cross-task distillation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    teacher_weights_log = {}  # ✅ 确保有默认值

                # 使用学生特征作为后备
                aligned_features = {k: v.clone() for k, v in student_features.items()}
                reconstructed_features = {k: v.clone() for k, v in student_features.items()}
                original_teacher_features = {}  # ✅ 空字典
        else:
            if verbose:
                print(f"  ⚠️  No cross-task distiller or active teachers")
            aligned_features = {k: v.clone() for k, v in student_features.items()}
            reconstructed_features = {k: v.clone() for k, v in student_features.items()}
            original_teacher_features = {}  # ✅ 空字典
            teacher_weights_log = {}  # ✅ 确保有默认值

        # ========== 4. 特征融合 ==========
        if verbose:
            print(f"\n[Step 4] Feature Fusion:")

        fused_features = {}

        if self.feature_fusion is not None:
            try:
                fused_features = self._fuse_features(
                    student_features=student_features,
                    aligned_features=aligned_features,
                    reconstructed_features=reconstructed_features,
                    teacher_features=all_teacher_features,
                    teacher_names=teacher_names_list,
                    task_id=task_id,
                    verbose=verbose
                )

                if verbose:
                    print(f"  Fused layers: {len(fused_features)}")

            except Exception as e:
                if verbose:
                    print(f"⚠️  Feature fusion failed: {e}")
                fused_features = {k: v.clone() for k, v in student_features.items()}
        else:
            fused_features = {k: v.clone() for k, v in student_features.items()}

        # ========== 5. 融合教师输出 ==========
        if verbose:
            print(f"\n[Step 5] Fused Teacher Output:")

        fused_teacher_output = None

        if teacher_names_list and dataset_name not in ['coco', 'voc']:
            # 只对分类任务融合输出
            teacher_outputs = []

            for teacher_name in teacher_names_list:
                teacher_model = self.teacher_models[teacher_name]
                with torch.no_grad():
                    teacher_output = teacher_model(images)
                    teacher_outputs.append(teacher_output)

            if teacher_outputs:
                # 简单平均
                fused_teacher_output = sum(teacher_outputs) / len(teacher_outputs)

                if verbose:
                    print(f"  Fused from {len(teacher_outputs)} teachers")

        # ========== 6. 验证重构目标 ==========
        if verbose and original_teacher_features:
            print(f"\n[Step 6] Reconstruction Target Validation:")
            print(f"  Original teacher features saved: {len(original_teacher_features)}")

            # 显示前3个特征的信息
            for i, (key, feat) in enumerate(list(original_teacher_features.items())[:3]):
                print(f"    {key}: shape={feat.shape}, mean={feat.mean().item():.4f}, std={feat.std().item():.4f}")
                if i >= 2:
                    break

        # ========== 7. 返回所有中间结果（✅ 添加原始特征）==========
        # 在 _forward_pass 的最后（第 3200 行左右）
        outputs = {
            'student_output': predictions,  # ✅ 必需
            'task_output': predictions,  # ✅ 添加这一行（与 student_output 相同）
            'teacher_output': fused_teacher_output,
            'student_features': student_features,  # ✅ 必需
            'aligned_features': aligned_features,
            'reconstructed_features': reconstructed_features,
            'fused_features': fused_features,
            'teacher_weights': teacher_weights_log,
            'teacher_features': all_teacher_features,
            'original_teacher_features': original_teacher_features  # ✅ 必需
        }

        if verbose:
            print(f"\n{'=' * 100}")
            print(f"[Forward Pass Complete]")
            print(f"  Output keys: {list(outputs.keys())}")
            print(f"{'=' * 100}\n")

        return outputs

    def _fuse_features(
            self,
            student_features: Dict[str, torch.Tensor],
            aligned_features: Dict[str, torch.Tensor],
            reconstructed_features: Dict[str, torch.Tensor],
            teacher_features: Dict[str, Dict[str, torch.Tensor]],
            teacher_names: List[str],
            task_id: int,
            verbose: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        融合学生、对齐、重构和教师特征（完整修复版）

        融合策略：
        1. 基础融合：aligned + λ * (reconstructed - student)
        2. 教师知识注入：通过1x1卷积融合多个教师特征
        3. 最终精炼：通过融合模块输出

        Args:
            student_features: 学生原始特征
            aligned_features: 经过跨任务蒸馏对齐的特征
            reconstructed_features: Return Decoder 重构的特征
            teacher_features: 教师特征字典
            teacher_names: 活跃的教师名称列表
            task_id: 当前任务ID
            verbose: 是否打印详细信息

        Returns:
            fused_features: 融合后的特征字典
        """
        fused = {}

        for layer_name in student_features.keys():
            # ========== 1. 获取三种特征 ==========
            s_feat = student_features.get(layer_name)
            a_feat = aligned_features.get(layer_name, s_feat)
            r_feat = reconstructed_features.get(layer_name, s_feat)

            if s_feat is None:
                continue

            # ========== 2. 基础融合：对齐特征 + 重构残差 ==========
            # 融合公式：fused = aligned + λ * (reconstructed - student)
            reconstruction_residual = r_feat - s_feat
            fused_feat = a_feat + self.reconstruction_weight * reconstruction_residual

            # ========== 3. 教师知识注入（如果有融合模块）==========
            if layer_name in self.feature_fusion and teacher_names:
                # 收集所有有效的教师特征
                teacher_feats_list = []

                for t_name in teacher_names:
                    if t_name not in teacher_features:
                        continue

                    t_feat = teacher_features[t_name].get(layer_name)
                    if t_feat is None:
                        continue

                    # 空间对齐
                    if t_feat.shape[-2:] != s_feat.shape[-2:]:
                        t_feat = F.interpolate(
                            t_feat,
                            size=s_feat.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )

                    # 通道对齐
                    if t_feat.shape[1] != s_feat.shape[1]:
                        if (hasattr(self.cross_task_distiller, 'channel_aligners') and
                                t_name in self.cross_task_distiller.channel_aligners and
                                layer_name in self.cross_task_distiller.channel_aligners[t_name]):
                            t_feat = self.cross_task_distiller.channel_aligners[t_name][layer_name](t_feat)
                        else:
                            continue  # 跳过无法对齐的特征

                    teacher_feats_list.append(t_feat)

                # 如果有有效的教师特征，进行融合
                if teacher_feats_list:
                    try:
                        # ✅ 修复：正确的融合方式
                        # 方案1：简单平均后通过融合模块
                        all_feats = torch.stack(
                            [fused_feat] + teacher_feats_list + [r_feat],
                            dim=0
                        )  # [N, B, C, H, W]

                        # 加权平均
                        avg_feat = all_feats.mean(dim=0)  # [B, C, H, W]

                        # 通过融合模块精炼
                        fused_feat = self.feature_fusion[layer_name](avg_feat)

                    except Exception as e:
                        if verbose:
                            print(f"  ⚠️  Fusion failed for {layer_name}: {e}")
                            import traceback
                            traceback.print_exc()

            fused[layer_name] = fused_feat

            # ========== 4. 打印融合信息（调试用）==========
            if verbose and layer_name in ['block2', 'block3', 'final']:
                recon_diff = F.mse_loss(r_feat, s_feat).item()
                align_diff = F.mse_loss(a_feat, s_feat).item()
                fused_diff = F.mse_loss(fused_feat, s_feat).item()

                print(f"  {layer_name}:")
                print(f"    Reconstruction MSE: {recon_diff:.6f}")
                print(f"    Alignment MSE: {align_diff:.6f}")
                print(f"    Fused MSE: {fused_diff:.6f}")
                print(f"    Fusion weight: {self.reconstruction_weight.item():.4f}")


        return fused

    def _get_teacher_output(self, teacher_model, teacher_name, images, dataset_name=None):
        """
        获取单个教师模型的输出和特征（完全修复版）

        Returns:
            teacher_output: 教师模型的最终输出（用于输出蒸馏）
            teacher_features: 教师模型的中间特征字典（用于特征蒸馏）
        """
        teacher_output = None
        teacher_features = None

        try:
            if teacher_name == 'MedSAM':
                # ========== MedSAM 处理 ==========
                mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
                denorm_images = images * std + mean
                denorm_images = denorm_images * 255.0
                numpy_images = denorm_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

                medsam_input = [{"image": img} for img in numpy_images]

                # 获取输出
                teacher_output = teacher_model(medsam_input, multimask_output=False)
                if teacher_output is not None:
                    teacher_output = teacher_output.float()

                # ✅ 关键修复：正确提取 MedSAM 的中间特征
                if hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'image_encoder'):
                    encoder = teacher_model.model.image_encoder

                    # 将输入转换为 MedSAM 格式
                    medsam_tensor = torch.from_numpy(numpy_images).permute(0, 3, 1, 2).float().to(images.device)

                    # 提取编码器特征
                    with torch.no_grad():
                        # MedSAM 使用 ViT 编码器
                        x = encoder.patch_embed(medsam_tensor)

                        # 收集不同层的特征
                        features_list = []
                        for i, blk in enumerate(encoder.blocks):
                            x = blk(x)
                            # 每隔几层保存一次特征
                            if i in [2, 5, 8, 11]:  # ViT-B 有 12 层
                                features_list.append(x)

                        # 转换为字典格式
                        teacher_features = {
                            'stem': features_list[0] if len(features_list) > 0 else x,
                            'block2': features_list[1] if len(features_list) > 1 else x,
                            'block3': features_list[2] if len(features_list) > 2 else x,
                            'block5': features_list[3] if len(features_list) > 3 else x,
                            'final': x
                        }
                else:
                    # 如果无法提取中间特征，使用输出作为所有层的特征
                    teacher_features = {
                        'stem': teacher_output,
                        'block2': teacher_output,
                        'block3': teacher_output,
                        'block5': teacher_output,
                        'final': teacher_output
                    }

            elif teacher_name == 'RETFound_MAE':
                # ========== RETFound_MAE 处理 ==========
                images_224 = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

                # ✅ 关键修复：提取 MAE 的中间特征
                if hasattr(teacher_model, 'forward_features'):
                    # 使用 forward_features 方法
                    features = teacher_model.forward_features(images_224)
                    teacher_output = features

                    # 如果返回的是字典，直接使用
                    if isinstance(features, dict):
                        teacher_features = features
                    else:
                        # 如果是单个张量，复制到所有层
                        teacher_features = {
                            'stem': features,
                            'block2': features,
                            'block3': features,
                            'block5': features,
                            'final': features
                        }
                else:
                    # 使用标准前向传播
                    result = teacher_model(images_224)

                    if isinstance(result, tuple):
                        teacher_output = result[1] if len(result) > 1 else result[0]
                    else:
                        teacher_output = result

                    # 转换为特征字典
                    teacher_features = {
                        'stem': teacher_output,
                        'block2': teacher_output,
                        'block3': teacher_output,
                        'block5': teacher_output,
                        'final': teacher_output
                    }

            elif teacher_name == 'USFM':
                # ========== USFM 处理 ==========
                teacher_output = teacher_model(images)
                if teacher_output is not None:
                    teacher_output = teacher_output.float()

                # ✅ 关键修复：提取 USFM 的编码器特征
                if hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'encoder'):
                    encoder = teacher_model.model.encoder

                    # 提取编码器的中间特征
                    features_list = []
                    x = images

                    for i, layer in enumerate(encoder):
                        x = layer(x)
                        # 保存关键层的特征
                        if i in [0, 2, 4, 6]:  # 根据 USFM 的实际结构调整
                            features_list.append(x)

                    teacher_features = {
                        'stem': features_list[0] if len(features_list) > 0 else x,
                        'block2': features_list[1] if len(features_list) > 1 else x,
                        'block3': features_list[2] if len(features_list) > 2 else x,
                        'block5': features_list[3] if len(features_list) > 3 else x,
                        'final': x
                    }
                else:
                    # 使用输出作为特征
                    teacher_features = {
                        'stem': teacher_output,
                        'block2': teacher_output,
                        'block3': teacher_output,
                        'block5': teacher_output,
                        'final': teacher_output
                    }

            elif teacher_name == 'BioMedPrase':
                # ========== BioMedPrase 处理 ==========
                try:
                    teacher_output = teacher_model(images)
                except:
                    try:
                        teacher_output = teacher_model({'image': images})
                    except:
                        teacher_output = None

                if teacher_output is not None:
                    teacher_output = teacher_output.float()

                    # ✅ 尝试提取中间特征
                    if hasattr(teacher_model, 'extract_features'):
                        teacher_features = teacher_model.extract_features(images)
                    else:
                        # 使用输出作为特征
                        teacher_features = {
                            'stem': teacher_output,
                            'block2': teacher_output,
                            'block3': teacher_output,
                            'block5': teacher_output,
                            'final': teacher_output
                        }

            else:
                # ========== 默认处理 ==========
                teacher_output = teacher_model(images)
                if teacher_output is not None:
                    teacher_output = teacher_output.float()
                    teacher_features = {
                        'stem': teacher_output,
                        'block2': teacher_output,
                        'block3': teacher_output,
                        'block5': teacher_output,
                        'final': teacher_output
                    }

        except Exception as e:
            print(f"⚠️  Error in {teacher_name}: {e}")
            import traceback
            traceback.print_exc()
            teacher_output = None
            teacher_features = None

        return teacher_output, teacher_features

    def _fuse_teacher_outputs(self, teacher_outputs, dataset_name, target_shape):
        """融合所有教师模型的输出"""
        fused_output = None

        # 根据任务类型选择融合策略
        if dataset_name in ['BUSI', 'kvasir_seg']:
            # 分割任务：使用加权平均融合
            weighted_outputs = []
            total_weight = 0

            for teacher_name, output in teacher_outputs.items():
                if output is None:
                    continue

                # 为不同教师分配权重（可以根据教师的专业领域调整）
                if teacher_name == 'MedSAM':
                    weight = 0.4  # MedSAM在分割任务上表现较好
                elif teacher_name == 'USFM':
                    weight = 0.3  # USFM也是分割模型
                elif teacher_name == 'BioMedPrase':
                    weight = 0.2
                else:  # RETFound_MAE
                    weight = 0.1

                # 调整输出尺寸以匹配目标
                if output.shape != target_shape:
                    if len(output.shape) == 4:
                        output = F.interpolate(
                            output,
                            size=target_shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    elif len(output.shape) == 3:
                        output = F.interpolate(
                            output.unsqueeze(1),
                            size=target_shape[2:],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(1)

                # 对分割输出应用sigmoid
                if len(output.shape) == 4 and output.shape[1] == 1:
                    output = torch.sigmoid(output)
                elif len(output.shape) == 3:
                    output = torch.sigmoid(output.unsqueeze(1))

                weighted_outputs.append(output * weight)
                total_weight += weight

            if weighted_outputs:
                fused_output = sum(weighted_outputs) / total_weight

        else:
            # 分类任务：使用平均概率融合
            all_probs = []
            for teacher_name, output in teacher_outputs.items():
                if output is None:
                    continue

                # 确保输出是logits
                if output.dim() == 4:
                    # 如果是4D输出，先全局平均池化
                    output = F.adaptive_avg_pool2d(output, 1).view(output.size(0), -1)

                # 应用softmax获取概率
                probs = F.softmax(output, dim=1)
                all_probs.append(probs)

            if all_probs:
                # 平均所有教师的概率
                fused_output = torch.mean(torch.stack(all_probs), dim=0)
                # 转换回logits（用于KL散度计算）
                fused_output = torch.log(fused_output + 1e-8)

        return fused_output

    def _fuse_teacher_outputs_with_gating(
            self,
            teacher_outputs,
            student_feat,
            teacher_feats,
            teacher_names,
            task_id,
            target_shape
    ):
        """
        使用动态门控网络融合多个教师模型的输出

        Args:
            teacher_outputs: 字典 {teacher_name: output}
            student_feat: 学生全局特征 [B, D_s]
            teacher_feats: 教师全局特征列表 [T个 [B, D_t]]
            teacher_names: 教师模型名称列表
            task_id: 任务ID
            target_shape: 目标输出形状 (H, W)

        Returns:
            fused_output: 融合后的输出 [B, C, H, W]
        """
        device = student_feat.device
        batch_size = student_feat.size(0)

        # ========== 1. 对齐教师特征维度 ==========
        teacher_feats_projected = []
        teacher_feats_original = []

        for i, (teacher_name, t_feat) in enumerate(zip(teacher_names, teacher_feats)):
            # 保存原始特征
            teacher_feats_original.append(t_feat)

            # 投影到统一维度
            projector_key = f"teacher_projector_{teacher_name}"
            if not hasattr(self, projector_key):
                # 动态创建投影器
                t_dim = t_feat.size(1)
                s_dim = student_feat.size(1)
                projector = nn.Linear(t_dim, s_dim).to(device)
                setattr(self, projector_key, projector)

            projector = getattr(self, projector_key)
            t_feat_proj = projector(t_feat)  # [B, D_s]
            teacher_feats_projected.append(t_feat_proj)

        # ========== 2. 使用门控网络计算权重 ==========
        teacher_weights = self.teacher_gating(
            student_feat=student_feat,
            teacher_feats_projected=teacher_feats_projected,
            teacher_feats_original=teacher_feats_original,
            task_id=task_id
        )  # [B, T]

        # 保存权重用于可视化（可选）
        self._last_teacher_weights = teacher_weights.detach().cpu()

        # ========== 3. 标准化教师输出 ==========
        normalized_outputs = []

        for teacher_name in teacher_names:
            output = teacher_outputs[teacher_name]

            # 处理不同格式的输出
            if isinstance(output, dict):
                if 'masks' in output:
                    output = output['masks']
                elif 'pred' in output:
                    output = output['pred']
                else:
                    raise ValueError(f"未知的输出格式: {output.keys()}")

            # 确保是4D张量 [B, C, H, W]
            if output.dim() == 3:
                output = output.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

            # 调整到目标尺寸
            if output.shape[-2:] != target_shape:
                output = F.interpolate(
                    output,
                    size=target_shape,
                    mode='bilinear',
                    align_corners=False
                )

            normalized_outputs.append(output)

        # ========== 4. 加权融合 ==========
        # teacher_weights: [B, T]
        # normalized_outputs: T个 [B, C, H, W]

        fused_output = torch.zeros_like(normalized_outputs[0])  # [B, C, H, W]

        for i, output in enumerate(normalized_outputs):
            # 扩展权重维度: [B, T] -> [B, 1, 1, 1]
            weight = teacher_weights[:, i:i + 1, None, None]  # [B, 1, 1, 1]
            fused_output += weight * output

        return fused_output

    def get_teacher_weights_stats(self):
        """
        获取最近一次门控权重的统计信息（用于日志记录）

        Returns:
            dict: 包含权重统计的字典
        """
        if not hasattr(self, '_last_teacher_weights'):
            return {}

        weights = self._last_teacher_weights  # [B, T]

        stats = {
            'mean': weights.mean(dim=0).tolist(),  # 每个教师的平均权重
            'std': weights.std(dim=0).tolist(),  # 标准差
            'max': weights.max(dim=0)[0].tolist(),  # 最大值
            'min': weights.min(dim=0)[0].tolist()  # 最小值
        }

        return stats

    def _compute_feature_distillation_loss(self, student_feat, teacher_feats):
        """
        计算特征蒸馏损失（学生特征与教师特征的对齐）

        Args:
            student_feat: [B, D_s]
            teacher_feats: List of [B, D_t]

        Returns:
            loss: 标量
        """
        if teacher_feats is None or len(teacher_feats) == 0:
            return torch.tensor(0.0, device=student_feat.device)

        total_loss = 0.0
        valid_count = 0

        for i, t_feat in enumerate(teacher_feats):
            if t_feat is None:
                continue

            # 投影到相同维度
            if t_feat.size(1) != student_feat.size(1):
                projector_key = f'_feat_projector_{i}'
                if not hasattr(self, projector_key):
                    projector = nn.Linear(
                        t_feat.size(1), student_feat.size(1)
                    ).to(student_feat.device)
                    setattr(self, projector_key, projector)

                projector = getattr(self, projector_key)
                t_feat = projector(t_feat)

            # 余弦相似度损失
            loss = 1 - F.cosine_similarity(student_feat, t_feat, dim=1).mean()
            total_loss += loss
            valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=student_feat.device)

        return total_loss / valid_count

    def _validate_epoch(self, epoch: int, dataset_name: str) -> Dict[str, float]:
        """验证一个epoch（针对单个数据集）"""
        print(f"Starting validation for epoch {epoch + 1}")
        self.student_model.eval()
        # ========== 🆕 设置门控网络为评估模式 ==========
        self.teacher_gating.eval()

        loader = self.val_loaders[dataset_name]

        try:
            print(f"Validating dataset: {dataset_name}")
            if dataset_name in ['BUSI', 'kvasir_seg']:
                # 分割任务评估
                metrics = self._evaluate_segmentation(loader, dataset_name)
            else:
                # 分类任务评估
                metrics = self._evaluate_classification(loader, dataset_name)

            print(f"Validation metrics for {dataset_name}: {metrics}")

        except Exception as e:
            print(f"Error evaluating dataset {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            # 记录错误但继续验证其他数据集
            metrics = {"error": str(e)}

        # 记录到TensorBoard
        self._log_val_metrics(epoch, metrics, dataset_name)

        return metrics

    def _evaluate_segmentation(self, loader, dataset_name):
        """评估分割任务"""
        self.student_model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (images, targets_list) in enumerate(loader):
                images = images.to(self.device)

                # 从目标列表中提取掩码并转换为张量
                mask_list = []
                for target in targets_list:
                    # 确保目标字典中有'mask'键
                    if 'mask' in target and target['mask'] is not None:
                        mask = target['mask']
                        # 确保mask是2D张量 [H, W]
                        if mask.dim() == 3:
                            # 检查哪个维度是1
                            if mask.shape[0] == 1:
                                mask = mask.squeeze(0)  # [1, H, W] -> [H, W]
                            elif mask.shape[2] == 1:
                                mask = mask.squeeze(2)  # [H, W, 1] -> [H, W]
                            else:
                                # 其他情况，假设是 [H, W, C]，取第一个通道
                                mask = mask[:, :, 0]
                        elif mask.dim() == 1:
                            # 如果是1D，需要reshape
                            _, _, H, W = images.shape
                            mask = mask.view(H, W)
                        mask_list.append(mask)
                    else:
                        # 如果没有掩码，创建一个空掩码
                        _, _, H, W = images.shape
                        mask_list.append(torch.zeros((H, W), dtype=torch.long))

                # 确保所有mask的形状一致
                if mask_list:
                    target_shape = mask_list[0].shape
                    for i in range(len(mask_list)):
                        if mask_list[i].shape != target_shape:
                            mask_list[i] = F.interpolate(
                                mask_list[i].unsqueeze(0).unsqueeze(0).float(),
                                size=target_shape,
                                mode='nearest'
                            ).squeeze(0).squeeze(0).long()

                # 堆叠掩码并移动到设备
                masks = torch.stack(mask_list).to(self.device)

                # 通过学生模型得到输出
                outputs, _ = self.student_model(images, dataset_name)

                # 修复维度问题：移除通道维度 [B, 1, H, W] -> [B, H, W]
                if outputs.dim() == 4 and outputs.shape[1] == 1:
                    # 二值分割：使用sigmoid
                    probs = torch.sigmoid(outputs).squeeze(1)  # [B, H, W]
                    preds = (probs > 0.5).float()
                else:
                    # 多类别分割：使用softmax
                    probs = F.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1).float()  # [B, H, W]

                all_preds.append(preds.cpu().numpy())
                all_targets.append(masks.cpu().numpy())

        # 合并所有预测
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # 计算指标
        metrics = {
            'IoU': self._calculate_iou(all_preds, all_targets),
            'Dice': self._calculate_dice(all_preds, all_targets),
            'Precision': self._calculate_precision(all_preds, all_targets),
            'Recall': self._calculate_recall(all_preds, all_targets),
            'F1': self._calculate_f1(all_preds, all_targets)
        }

        return metrics

    def _evaluate_classification(self, loader, dataset_name):
        """评估分类任务"""
        self.student_model.eval()
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for images, targets_list in loader:
                images = images.to(self.device)

                # 从目标列表中提取标签
                labels = []
                for target in targets_list:
                    # 根据任务类型提取标签
                    if 'class_label' in target:
                        label = target['class_label']
                        # 确保label是标量
                        if isinstance(label, torch.Tensor):
                            if label.numel() == 1:
                                label = label.item()
                            else:
                                # 如果是多维张量，取第一个元素
                                label = label.flatten()[0].item()
                        elif isinstance(label, (list, tuple)):
                            # 如果是列表，取第一个元素
                            label = label[0] if len(label) > 0 else 0
                        labels.append(int(label))
                    elif 'labels' in target and len(target['labels']) > 0:
                        # 对于检测任务，使用第一个标签
                        label = target['labels'][0]
                        if isinstance(label, torch.Tensor):
                            if label.numel() == 1:
                                label = label.item()
                            else:
                                label = label.flatten()[0].item()
                        elif isinstance(label, (list, tuple)):
                            label = label[0] if len(label) > 0 else 0
                        labels.append(int(label))
                    else:
                        labels.append(0)  # 默认值

                # 转换为张量
                targets = torch.tensor(labels, dtype=torch.long).to(self.device)

                outputs, _ = self.student_model(images, dataset_name)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)

        # 计算指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        metrics = {
            'Accuracy': accuracy_score(all_targets, all_preds),
            'Precision': precision_score(all_targets, all_preds, average='weighted', zero_division=0),
            'Recall': recall_score(all_targets, all_preds, average='weighted', zero_division=0),
            'F1': f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        }

        # 如果是二分类，计算AUC
        if all_probs.shape[1] == 2:
            try:
                metrics['AUC'] = roc_auc_score(all_targets, all_probs[:, 1])
            except:
                metrics['AUC'] = 0.5

        return metrics

    def _calculate_iou(self, preds, targets):
        """计算IoU"""
        intersection = np.logical_and(preds, targets).sum()
        union = np.logical_or(preds, targets).sum()
        return intersection / (union + 1e-6)

    def _calculate_dice(self, preds, targets):
        """计算Dice系数"""
        intersection = np.logical_and(preds, targets).sum()
        return 2 * intersection / (preds.sum() + targets.sum() + 1e-6)

    def _calculate_precision(self, preds, targets):
        """计算精确率"""
        true_positive = np.logical_and(preds, targets).sum()
        predicted_positive = preds.sum()
        return true_positive / (predicted_positive + 1e-6)

    def _calculate_recall(self, preds, targets):
        """计算召回率"""
        true_positive = np.logical_and(preds, targets).sum()
        actual_positive = targets.sum()
        return true_positive / (actual_positive + 1e-6)

    def _calculate_f1(self, preds, targets):
        """计算F1分数"""
        precision = self._calculate_precision(preds, targets)
        recall = self._calculate_recall(preds, targets)
        return 2 * (precision * recall) / (precision + recall + 1e-6)

    def _save_checkpoint(self, epoch, val_metrics, dataset_name):
        """保存检查点（针对单个数据集）"""
        # 检查是否是最佳模型
        if dataset_name in ['BUSI', 'kvasir_seg']:
            current_score = val_metrics['IoU']
        else:
            current_score = val_metrics['Accuracy']

        is_best = False
        if not self.best_metrics or current_score > self.best_metrics:
            self.best_metrics = current_score
            self.best_epoch = epoch
            is_best = True

        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            # ========== 🆕 添加门控网络状态 ==========
            'teacher_gating_state_dict': self.teacher_gating.state_dict(),
            'metrics': val_metrics,
            'train_history': self.train_history,
            'config': self.config,
            'dataset_name': dataset_name
        }

        # 保存最新检查点
        torch.save(checkpoint, self.save_dir / f'{dataset_name}_last_checkpoint.pth')

        # 如果是最佳模型，额外保存
        if is_best:
            torch.save(checkpoint, self.save_dir / f'{dataset_name}_best_checkpoint.pth')
            # 单独保存最佳模型权重（便于部署）
            torch.save(self.student_model.state_dict(), self.save_dir / f'{dataset_name}_best_model.pth')
            print(f"New best model saved for {dataset_name}! Score: {current_score:.4f}")

    def _log_train_metrics(self, epoch, losses):
        """记录训练指标到TensorBoard"""
        for dataset_name, loss_dict in losses.items():
            for loss_name, loss_value in loss_dict.items():
                self.writer.add_scalar(f'Train/{dataset_name}/{loss_name}', loss_value, epoch)

    def _log_val_metrics(self, epoch, metrics, dataset_name):
        """记录验证指标到TensorBoard（针对单个数据集）"""
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f'Val/{dataset_name}/{metric_name}', metric_value, epoch)

    def _print_epoch_summary(self, epoch, train_losses, val_metrics, lr, dataset_name):
        """打印epoch总结（针对单个数据集）"""
        print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']} - {dataset_name} - LR: {lr:.6f}")
        print("-" * 80)

        # 打印训练损失
        print("Training Losses:")
        loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_losses.items()])
        print(f"  {dataset_name}: {loss_str}")

        # 打印验证指标
        print("\nValidation Metrics:")
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
        print(f"  {dataset_name}: {metric_str}")

        print("-" * 80)

    def _on_training_end(self, dataset_name):
        """训练结束后的操作（针对单个数据集）"""
        print(f"\nTraining completed for {dataset_name}!")
        print(f"Best model saved at epoch {self.best_epoch}")

        # 保存训练历史
        with open(self.save_dir / f'{dataset_name}_train_history.json', 'w') as f:
            json.dump(self.train_history, f, indent=4)

        # 生成训练报告
        self._generate_training_report(dataset_name)

        # 关闭TensorBoard writer
        self.writer.close()

    def _generate_training_report(self, dataset_name):
        """生成训练报告（针对单个数据集）"""
        report = {
            'experiment_name': self.experiment_name,
            'dataset_name': dataset_name,
            'config': self.config,
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics,
            'total_parameters': sum(p.numel() for p in self.student_model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)
        }

        with open(self.save_dir / f'{dataset_name}_training_report.json', 'w') as f:
            json.dump(report, f, indent=4)


# ============================================
# 8. 模型评估器
# ============================================

class ModelEvaluator:
    """模型综合评估器"""

    def __init__(self, model_path: str, config_path: str, device: torch.device):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # 加载模型
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()

        # 创建评估结果目录
        self.eval_dir = Path(model_path).parent / "evaluation"
        self.eval_dir.mkdir(exist_ok=True)

    def _load_model(self, model_path):
        """加载训练好的模型"""
        # 创建模型实例
        num_classes = self.config.get('num_classes', {
            'classification': 5,
            'recognition': 3
        })
        model = ImprovedMultiTaskStudent(num_classes)

        # 加载权重
        if model_path.endswith('.pth'):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

        return model.to(self.device)

    def evaluate_all_datasets(self, test_loaders: Dict[str, DataLoader]):
        """评估所有数据集"""
        results = {}

        for dataset_name, loader in test_loaders.items():
            print(f"\nEvaluating {dataset_name}...")

            if dataset_name in ['BUSI', 'kvasir_seg']:
                results[dataset_name] = self.evaluate_segmentation(loader, dataset_name)
            else:
                results[dataset_name] = self.evaluate_classification(loader, dataset_name)

        # 保存结果
        self._save_results(results)

        # 生成可视化报告
        self._generate_visualizations(results)

        return results

    def evaluate_segmentation(self, loader, dataset_name):
        """详细评估分割任务"""
        all_preds = []
        all_targets = []
        inference_times = []

        with torch.no_grad():
            for images, masks in tqdm(loader, desc="Evaluating"):
                start_time = time.time()

                images = images.to(self.device)
                outputs, _ = self.model(images, dataset_name)
                preds = torch.sigmoid(outputs) > 0.5

                inference_times.append(time.time() - start_time)

                all_preds.append(preds.cpu().numpy())
                all_targets.append(masks.numpy())

        # 合并所有预测
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # 计算详细指标
        results = {
            'IoU': self._calculate_iou(all_preds, all_targets),
            'Dice': self._calculate_dice(all_preds, all_targets),
            'Precision': self._calculate_precision(all_preds, all_targets),
            'Recall': self._calculate_recall(all_preds, all_targets),
            'F1': self._calculate_f1(all_preds, all_targets),
            'Pixel_Accuracy': np.mean(all_preds == all_targets),
            'Mean_Inference_Time': np.mean(inference_times),
            'Std_Inference_Time': np.std(inference_times)
        }

        return results

    def evaluate_classification(self, loader, dataset_name):
        """详细评估分类任务"""
        all_preds = []
        all_targets = []
        all_probs = []
        inference_times = []

        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Evaluating"):
                start_time = time.time()

                images = images.to(self.device)
                outputs, _ = self.model(images, dataset_name)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                inference_times.append(time.time() - start_time)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.numpy())
                all_probs.append(probs.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)

        # 计算详细指标
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            all_targets, all_preds, average=None
        )

        results = {
            'Accuracy': accuracy_score(all_targets, all_preds),
            'Precision_per_class': precision.tolist(),
            'Recall_per_class': recall.tolist(),
            'F1_per_class': f1.tolist(),
            'Support_per_class': support.tolist(),
            'Mean_Inference_Time': np.mean(inference_times),
            'Std_Inference_Time': np.std(inference_times)
        }

        # 计算混淆矩阵
        cm = confusion_matrix(all_targets, all_preds)
        results['Confusion_Matrix'] = cm.tolist()

        # 如果是二分类，计算ROC曲线
        if all_probs.shape[1] == 2:
            fpr, tpr, _ = roc_curve(all_targets, all_probs[:, 1])
            results['ROC_Curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
            results['AUC'] = roc_auc_score(all_targets, all_probs[:, 1])

        # 生成分类报告
        report = classification_report(all_targets, all_preds, output_dict=True)
        results['Classification_Report'] = report

        return results

    def _calculate_iou(self, preds, targets):
        """计算IoU"""
        intersection = np.logical_and(preds, targets).sum()
        union = np.logical_or(preds, targets).sum()
        return intersection / (union + 1e-6)

    def _calculate_dice(self, preds, targets):
        """计算Dice系数"""
        intersection = np.logical_and(preds, targets).sum()
        return 2 * intersection / (preds.sum() + targets.sum() + 1e-6)

    def _calculate_precision(self, preds, targets):
        """计算精确率"""
        true_positive = np.logical_and(preds, targets).sum()
        predicted_positive = preds.sum()
        return true_positive / (predicted_positive + 1e-6)

    def _calculate_recall(self, preds, targets):
        """计算召回率"""
        true_positive = np.logical_and(preds, targets).sum()
        actual_positive = targets.sum()
        return true_positive / (actual_positive + 1e-6)

    def _calculate_f1(self, preds, targets):
        """计算F1分数"""
        precision = self._calculate_precision(preds, targets)
        recall = self._calculate_recall(preds, targets)
        return 2 * (precision * recall) / (precision + recall + 1e-6)

    def _save_results(self, results):
        """保存评估结果"""
        # 保存为JSON
        with open(self.eval_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=4)

        # 创建汇总表格
        summary_data = []
        for dataset, metrics in results.items():
            row = {'Dataset': dataset}
            if 'IoU' in metrics:
                row['Type'] = 'Segmentation'
                row['Primary_Metric'] = f"IoU: {metrics['IoU']:.4f}"
                row['Secondary_Metric'] = f"Dice: {metrics['Dice']:.4f}"
            else:
                row['Type'] = 'Classification'
                row['Primary_Metric'] = f"Acc: {metrics['Accuracy']:.4f}"
                row['Secondary_Metric'] = f"AUC: {metrics.get('AUC', 'N/A')}"
            row[
                'Inference_Time_ms'] = f"{metrics['Mean_Inference_Time'] * 1000:.2f} ± {metrics['Std_Inference_Time'] * 1000:.2f}"
            summary_data.append(row)

        # 保存汇总表格
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.eval_dir / 'evaluation_summary.csv', index=False)
        print("\nEvaluation Summary:")
        print(summary_df.to_string(index=False))

    def _generate_visualizations(self, results):
        """生成可视化报告"""
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()

        # 1. 各数据集性能对比
        datasets = list(results.keys())
        primary_metrics = []
        for dataset in datasets:
            if 'IoU' in results[dataset]:
                primary_metrics.append(results[dataset]['IoU'])
            else:
                primary_metrics.append(results[dataset]['Accuracy'])

        axes[0].bar(datasets, primary_metrics)
        axes[0].set_title('Primary Metrics Across Datasets')
        axes[0].set_ylabel('Score')
        axes[0].set_ylim(0, 1)

        # 2. 推理时间对比
        inference_times = [results[d]['Mean_Inference_Time'] * 1000 for d in datasets]
        axes[1].bar(datasets, inference_times)
        axes[1].set_title('Average Inference Time')
        axes[1].set_ylabel('Time (ms)')

        # 3. 分割任务F1分数
        seg_datasets = [d for d in datasets if d in ['BUSI', 'kvasir_seg']]
        if seg_datasets:
            f1_scores = [results[d]['F1'] for d in seg_datasets]
            axes[2].bar(seg_datasets, f1_scores)
            axes[2].set_title('F1 Scores for Segmentation Tasks')
            axes[2].set_ylim(0, 1)

        # 4. 分类任务混淆矩阵（示例）
        cls_datasets = [d for d in datasets if d not in ['BUSI', 'kvasir_seg']]
        if cls_datasets and 'Confusion_Matrix' in results[cls_datasets[0]]:
            cm = np.array(results[cls_datasets[0]]['Confusion_Matrix'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[3])
            axes[3].set_title(f'Confusion Matrix - {cls_datasets[0]}')

        plt.tight_layout()
        plt.savefig(self.eval_dir / 'evaluation_report.png', dpi=300)
        plt.close()


class DiceLoss(nn.Module):
    """Dice Loss 实现，支持多类别和二值分割"""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # 确保维度一致
        if target.dim() == 3:
            target = target.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

        # 二值分割处理
        if pred.size(1) == 1:
            pred = torch.sigmoid(pred)
            # 添加通道维度
            pred = torch.cat([1 - pred, pred], dim=1)
            target = torch.cat([1 - target, target], dim=1)
        else:
            pred = F.softmax(pred, dim=1)

        # 展平空间维度
        pred = pred.contiguous().view(pred.size(0), pred.size(1), -1)  # (B, C, H*W)
        target = target.contiguous().view(target.size(0), target.size(1), -1)  # (B, C, H*W)

        # 计算交集
        intersection = (pred * target).sum(dim=2)  # (B, C)

        # 计算每个类别的Dice系数
        dice = (2. * intersection + self.smooth) / (
                pred.sum(dim=2) + target.sum(dim=2) + self.smooth
        )

        # 返回平均Dice Loss (1 - Dice)
        return 1 - dice.mean()


# ============================================
# 9. 主函数和使用示例
# ============================================

def start_train():
    """主函数：执行完整的多任务多教师知识蒸馏训练流程（统一模型版本）"""

    # 配置参数
    config = {
        'num_epochs': 200,
        'batch_size': 4,
        'learning_rate': 3e-4,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'temperature': 1.5,
        'feature_weight': 0.01,
        'output_weight': 0.05,
        'task_weight': 0.94,
        'early_stopping_patience': 20,
        'early_stopping_delta': 0.001,
        'num_classes': {
            'classification': 5,  # APTOS2019
            'recognition': 3,  # ISIC2017
            'segmentation': 3  # BUSI (添加分割任务的类别数)
        }
    }

    # 数据集配置（包含所有数据集）
    dataset_config = {
        "datasets": {
            "BUSI": {
                "data_dir": "/root/autodl-tmp/datasets/BUSI",
                "task_type": "segmentation",
                "num_classes": 3,
                "labels": ['benign', 'malignant', 'normal'],
            },
            "ISIC2017": {
                "data_dir": "/root/autodl-tmp/datasets/ISIC2017",
                "task_type": "recognition",
                "num_classes": 3,
                "labels": ["melanoma", "seborrheic_keratosis", "nevus/benign pigmented lesion"]
            },
            "APTOS2019": {
                "data_dir": "/root/autodl-tmp/datasets/APTOS2019",
                "task_type": "classification",
                "num_classes": 5,
                "labels": ['anodr', 'bmilddr', 'cmoderatedr', 'dseveredr', 'eproliferativedr'],
            },
            "kvasir_seg": {
                "data_dir": "/root/autodl-tmp/datasets/Kvasir-SEG",
                "task_type": "segmentation",
                "num_classes": 1,
                "labels": ['polyp']
            }
        },
        "training": config
    }

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 保存配置文件
    config_file = 'data_config.json'
    with open(config_file, 'w') as f:
        json.dump(dataset_config, f, indent=2)
    print(f"Configuration saved to {config_file}")

    # 创建数据加载器（所有数据集）
    print("\nCreating data loaders...")
    manager = DataIntegrationManager(config_file)
    dataloaders = manager.create_dataloaders(batch_size=config['batch_size'])

    # 打印数据集信息
    print("\nDataset loaders created:")
    for mode in ['train', 'val']:
        print(f"\n{mode.upper()} sets:")
        for dataset_name, loader in dataloaders[mode].items():
            print(f"  {dataset_name}: {len(loader)} batches")

    # 加载教师模型
    print("\nLoading teacher models...")
    teacher_models = load_models()
    print(f"Successfully loaded {len(teacher_models)} teacher models:")
    for name in teacher_models.keys():
        print(f"  - {name}")

    # 获取所有数据集名称
    dataset_names = list(dataloaders['train'].keys())
    print(f"\nDatasets to train: {dataset_names}")

    # ========== 核心修改：统一模型训练 ==========
    print("\n" + "=" * 80)
    print("Starting UNIFIED multi-task training (Single Student Model)")
    print("=" * 80)

    # 1. 只创建一次学生模型（在循环外）
    student_model = ImprovedMultiTaskStudent(config['num_classes'])
    student_model = student_model.to(device)
    print(f"\n✅ Unified Student Model Created")
    print(f"   Total parameters: {sum(p.numel() for p in student_model.parameters()) / 1e6:.2f}M")

    # 2. 创建统一的训练器（传入所有数据集的 DataLoader）
    trainer = MultiTeacherDistillationTrainer(
        student_model=student_model,
        teacher_models=teacher_models,
        train_loaders=dataloaders['train'],  # 传入所有训练集
        val_loaders=dataloaders['val'],  # 传入所有验证集
        config=config,
        device=device,
        save_dir="./experiments/multi_teacher_distillation/unified_model"
    )

    # 3. 执行统一训练（在 Trainer 内部实现任务交替）
    print("\n" + "=" * 80)
    print("Training Strategy: Task Interleaving")
    print("=" * 80)

    # 调用训练器的统一训练方法
    trainer.train_unified(dataset_names)

    print("\n" + "=" * 80)
    print("✅ Unified Multi-Task Training Completed!")
    print("=" * 80)

    # 保存最终的统一模型
    final_model_path = "./experiments/multi_teacher_distillation/unified_model/final_unified_model.pth"
    torch.save({
        'model_state_dict': student_model.state_dict(),
        'config': config,
        'dataset_names': dataset_names
    }, final_model_path)
    print(f"\n💾 Final unified model saved to: {final_model_path}")

    return student_model, trainer


if __name__ == "__main__":
    start_train()

