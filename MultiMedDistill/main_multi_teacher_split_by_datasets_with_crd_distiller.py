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
    """跨教师注意力机制"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.student_dim = dim
        # Key和Value不预先创建，将在forward中动态创建
        self.key = None
        self.value = None
        self.query = nn.Conv2d(dim, dim, 1)
        
        # 投影层
        self.q_proj = None
        self.out_proj = None
        
    def forward(self, student_feat: torch.Tensor, 
                teacher_feats: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """使用注意力机制融合多个教师的知识"""
        B, C, H, W = student_feat.shape
        
        # Query来自学生特征
        q = self.query(student_feat).view(B, C, -1).transpose(1, 2)  # B, HW, C
        
        # 获取所有教师特征的通道数
        teacher_channels = set()
        for s_feat_proj, t_feat in teacher_feats:
            t_C = t_feat.shape[1]
            teacher_channels.add(t_C)
        
        # 使用最大的教师通道数作为目标维度
        if len(teacher_channels) > 0:
            common_teacher_dim = max(teacher_channels)
        else:
            common_teacher_dim = C
        
        # 动态创建或更新 Key/Value 层（使用教师特征维度）
        if self.key is None or self.key.out_channels != common_teacher_dim:
            self.key = nn.Conv2d(common_teacher_dim, common_teacher_dim, 1).to(student_feat.device)
            self.value = nn.Conv2d(common_teacher_dim, common_teacher_dim, 1).to(student_feat.device)
        
        # 动态创建或更新投影层
        if C != common_teacher_dim:
            if self.q_proj is None or self.q_proj.out_features != common_teacher_dim:
                self.q_proj = nn.Linear(C, common_teacher_dim).to(student_feat.device)
            
            if self.out_proj is None or self.out_proj.in_features != common_teacher_dim:
                self.out_proj = nn.Linear(common_teacher_dim, C).to(student_feat.device)
        else:
            self.q_proj = None
            self.out_proj = None
        
        # 融合所有教师的知识
        all_values = []
        all_weights = []
        
        for s_feat_proj, t_feat in teacher_feats:
            # s_feat_proj 已经被投影到教师特征空间
            # t_feat 是教师特征
            t_C = t_feat.shape[1]
            
            # 确保特征尺寸匹配
            if t_feat.shape[-2:] != (H, W):
                t_feat = F.interpolate(t_feat, size=(H, W), mode='bilinear', align_corners=False)
            
            # Key和Value使用投影后的学生特征（在教师特征空间）
            k = self.key(s_feat_proj).view(B, t_C, -1).transpose(1, 2)  # B, HW, t_C
            v = self.value(s_feat_proj).view(B, t_C, -1).transpose(1, 2)  # B, HW, t_C
            
            # 计算注意力权重
            if C != t_C and self.q_proj is not None:
                # 如果维度不匹配，将q投影到t_C维度
                q_proj = self.q_proj(q)  # B, HW, t_C
            else:
                q_proj = q
            
            attn = torch.matmul(q_proj, k.transpose(-2, -1)) * (t_C ** -0.5)
            attn = F.softmax(attn, dim=-1)
            
            # 应用注意力
            out = torch.matmul(attn, v)
            all_values.append(out)
            all_weights.append(attn.mean(dim=-1, keepdim=True))  # B, HW, 1
        
        # 加权融合所有教师的输出
        if all_values:
            # 归一化权重
            weights = torch.cat(all_weights, dim=-1)  # B, HW, num_teachers
            weights = F.softmax(weights, dim=-1)
            
            # 融合
            fused = sum(w.unsqueeze(-1) * v for w, v in 
                       zip(weights.split(1, dim=-1), all_values))
            
            # 将融合后的特征投影回学生特征空间
            fused = fused.view(B, H, W, -1)  # [B, H, W, t_C]
            
            t_C = fused.shape[-1]
            if t_C != C and self.out_proj is not None:
                # 重塑为 [B, H*W, t_C] 进行投影
                fused_flat = fused.permute(0, 3, 1, 2).contiguous().view(B, t_C, -1)  # [B, t_C, H*W]
                fused_proj = self.out_proj(fused_flat)  # [B, C, H*W]
                fused = fused_proj.view(B, C, H, W).permute(0, 2, 3, 1)  # [B, H, W, C]
            
            # 残差连接
            return student_feat + fused
        else:
            return student_feat


class CrossTaskKnowledgeDistiller(nn.Module):
    """跨任务知识蒸馏器"""
    
    def __init__(self, student_dims: Dict[str, int], teacher_dims: Dict[str, Dict[str, int]]):
        super().__init__()
        
        # 为每个教师的每层特征创建投影器
        self.projectors = nn.ModuleDict()
        
        for teacher_name, t_dims in teacher_dims.items():
            self.projectors[teacher_name] = nn.ModuleDict()
            for layer_name in student_dims.keys():
                # 获取学生特征维度
                s_dim = student_dims[layer_name]
                
                # 获取教师特征维度（使用对应的层名或默认值）
                if layer_name in t_dims:
                    t_dim = t_dims[layer_name]
                elif 'final' in t_dims:
                    t_dim = t_dims['final']
                else:
                    t_dim = list(t_dims.values())[0]
                
                # 使用残差连接改进投影
                self.projectors[teacher_name][layer_name] = nn.Sequential(
                    nn.Conv2d(s_dim, s_dim, 3, padding=1),
                    nn.BatchNorm2d(s_dim),
                    nn.ReLU(),
                    nn.Conv2d(s_dim, t_dim, 1),
                    nn.BatchNorm2d(t_dim)
                )
        
        # 注意力模块，用于融合不同教师的知识
        # 为所有学生层创建注意力模块
        self.attention_modules = nn.ModuleDict()
        for layer_name in student_dims.keys():
            self.attention_modules[layer_name] = CrossTeacherAttention(student_dims[layer_name])
    
    def forward(self, student_features: Dict[str, torch.Tensor], 
                teacher_features: Dict[str, Dict[str, torch.Tensor]],
                active_teachers: List[str]) -> Dict[str, torch.Tensor]:
        """对齐学生和教师特征"""
        aligned_features = {}
        
        # 层名映射：将学生模型的层名映射到蒸馏器的层名
        layer_mapping = {
            'stem': 'low',
            'block2': 'low',
            'block3': 'mid',
            'block5': 'high',
            'final': 'final'
        }
        
        for layer_name in student_features:
            mapped_layer = layer_mapping.get(layer_name)
            if mapped_layer and mapped_layer in self.attention_modules:
                # 收集所有激活教师的特征
                teacher_feats = []
                for teacher_name in active_teachers:
                    if teacher_name in teacher_features:
                        t_feat = teacher_features[teacher_name]
                        
                        # 如果教师特征是字典，尝试匹配层名
                        if isinstance(t_feat, dict):
                            if layer_name in t_feat:
                                t_feat = t_feat[layer_name]
                            elif mapped_layer in t_feat:
                                t_feat = t_feat[mapped_layer]
                            else:
                                # 如果没有匹配的层名，使用第一个可用的层
                                t_feat = next(iter(t_feat.values()))
                        
                        # 确保特征是2D卷积特征 [B, C, H, W]
                        if t_feat.dim() == 3:
                            # [B, H*W, C] -> [B, C, H, W]
                            B, HW, C = t_feat.shape
                            H = W = int(HW ** 0.5)
                            t_feat = t_feat.transpose(1, 2).reshape(B, C, H, W)
                        
                        # 检查投影器是否存在
                        if teacher_name in self.projectors and layer_name in self.projectors[teacher_name]:
                            # 投影学生特征到教师特征空间
                            s_feat_proj = self.projectors[teacher_name][layer_name](student_features[layer_name])
                            teacher_feats.append((s_feat_proj, t_feat))
                        elif teacher_name in self.projectors and mapped_layer in self.projectors[teacher_name]:
                            # 使用映射的层名
                            s_feat_proj = self.projectors[teacher_name][mapped_layer](student_features[layer_name])
                            teacher_feats.append((s_feat_proj, t_feat))
                
                if teacher_feats:
                    # 使用注意力机制融合知识
                    aligned_feat = self.attention_modules[mapped_layer](
                        student_features[layer_name], teacher_feats
                    )
                    aligned_features[layer_name] = aligned_feat
                else:
                    aligned_features[layer_name] = student_features[layer_name]
            else:
                # 对于没有注意力模块的层，直接返回原始特征
                aligned_features[layer_name] = student_features[layer_name]
        
        return aligned_features


# ============================================
# 3. CRD蒸馏核心组件
# ============================================

class CRDDistiller(nn.Module):
    """
    Classification Regression Distillation (CRD)
    通过回归预测教师和学生特征的距离来优化蒸馏
    """
    
    def __init__(self, feature_dim: int, num_classes: int = None, temperature: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.num_classes = num_classes
        
        # 特征投影器：将特征投影到统一的度量空间
        self.feature_projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 类别原型：每个类别的特征中心
        if num_classes is not None:
            self.class_prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))
            nn.init.xavier_uniform_(self.class_prototypes)
        
        # 回归头：预测学生特征与教师特征的距离
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
    
    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        投影特征到统一空间
        
        Args:
            features: [B, C, H, W] 或 [B, C] 的特征张量
        
        Returns:
            投影后的特征 [B, feature_dim]
        """
        if features.dim() == 4:
            # 全局平均池化
            features = F.adaptive_avg_pool2d(features, 1).view(features.size(0), -1)
        
        # 投影特征
        projected = self.feature_projector(features)
        
        # L2归一化
        projected = F.normalize(projected, p=2, dim=1)
        
        return projected
    
    def compute_distance(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        """
        计算学生特征与教师特征的距离
        
        Args:
            student_feat: [B, D] 学生特征
            teacher_feat: [B, D] 教师特征
        
        Returns:
            距离 [B, 1]
        """
        # 拼接特征
        concat_feat = torch.cat([student_feat, teacher_feat], dim=1)
        
        # 使用回归器预测距离
        distance = self.regressor(concat_feat)
        
        return distance
    
    def compute_contrastive_loss(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor, 
                                labels: torch.Tensor = None) -> torch.Tensor:
        """
        计算对比损失
        
        Args:
            student_feat: [B, D] 学生特征
            teacher_feat: [B, D] 教师特征
            labels: [B] 类别标签（可选）
        
        Returns:
            对比损失
        """
        # 计算相似度矩阵
        student_norm = F.normalize(student_feat, p=2, dim=1)
        teacher_norm = F.normalize(teacher_feat, p=2, dim=1)
        
        similarity = torch.matmul(student_norm, teacher_norm.T) / self.temperature
        
        if labels is not None:
            # 有监督对比损失：同一类别的样本为正样本
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(student_feat.device)
            
            # 排除自身
            mask = mask - torch.eye(mask.size(0)).to(student_feat.device)
            
            # 计算对比损失
            exp_sim = torch.exp(similarity)
            positive_sim = exp_sim * mask
            loss = -torch.log(positive_sim.sum(dim=1) / exp_sim.sum(dim=1) + 1e-8)
            loss = loss.mean()
        else:
            # 无监督对比损失：最小化学生-教师对的距离
            positive_pairs = torch.diag(similarity)
            loss = -positive_pairs.mean()
        
        return loss
    
    def compute_class_prototype_loss(self, student_feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算类别原型损失
        
        Args:
            student_feat: [B, D] 学生特征
            labels: [B] 类别标签
        
        Returns:
            原型损失
        """
        if self.num_classes is None:
            return torch.tensor(0.0, device=student_feat.device)
        
        # 获取每个样本的类别原型
        prototypes = self.class_prototypes[labels]  # [B, D]
        
        # 计算学生特征与原型之间的距离
        distance = F.mse_loss(student_feat, prototypes)
        
        return distance
    
    def forward(self, student_features: Dict[str, torch.Tensor], 
                teacher_features: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        CRD蒸馏前向传播
        
        Args:
            student_features: 学生特征字典 {layer_name: feature_tensor}
            teacher_features: 教师特征字典 {teacher_name: {layer_name: feature_tensor}}
            labels: 类别标签（可选）
        
        Returns:
            损失字典
        """
        losses = {}
        total_loss = 0.0
        
        # 对每个特征层进行CRD蒸馏
        for layer_name, s_feat in student_features.items():
            # 投影学生特征
            s_proj = self.project_features(s_feat)  # [B, D]
            
            layer_losses = []
            
            # 对每个教师模型的特征进行蒸馏
            for teacher_name, t_feats_dict in teacher_features.items():
                if isinstance(t_feats_dict, dict):
                    # 如果教师特征是字典
                    if layer_name in t_feats_dict:
                        t_feat = t_feats_dict[layer_name]
                    elif 'final' in t_feats_dict:
                        t_feat = t_feats_dict['final']
                    else:
                        # 使用第一个可用的特征
                        t_feat = next(iter(t_feats_dict.values()))
                else:
                    # 如果教师特征是单个张量
                    t_feat = t_feats_dict
                
                # 投影教师特征
                t_proj = self.project_features(t_feat)  # [B, D]
                
                # 1. 对比损失：拉近学生和教师特征
                contrastive_loss = self.compute_contrastive_loss(s_proj, t_proj, labels)
                layer_losses.append(contrastive_loss)
                
                # 2. 回归损失：预测距离
                predicted_distance = self.compute_distance(s_proj, t_proj)  # [B, 1]
                # 真实距离是余弦距离的近似
                true_distance = 1.0 - torch.sum(s_proj * t_proj, dim=1, keepdim=True)  # [B, 1]
                regression_loss = F.mse_loss(predicted_distance, true_distance.detach())
                layer_losses.append(regression_loss * 0.5)  # 回归损失权重为0.5
            
            # 平均当前层的损失
            if layer_losses:
                layer_loss = sum(layer_losses) / len(layer_losses)
                losses[f'crd_{layer_name}'] = layer_loss
                total_loss += layer_loss
        
        # 3. 类别原型损失（如果有标签）
        if labels is not None and self.num_classes is not None:
            for layer_name, s_feat in student_features.items():
                s_proj = self.project_features(s_feat)
                proto_loss = self.compute_class_prototype_loss(s_proj, labels)
                losses[f'crd_proto_{layer_name}'] = proto_loss
                total_loss += proto_loss * 0.1  # 原型损失权重为0.1
        
        losses['crd_total'] = total_loss
        
        return losses


class CRDMultiTeacherDistiller(nn.Module):
    """
    多教师CRD蒸馏器
    集成多个教师模型的知识，使用CRD策略进行蒸馏
    """
    
    def __init__(self, student_dims: Dict[str, int], teacher_dims: Dict[str, Dict[str, int]],
                 num_classes: Dict[str, int] = None, temperature: float = 0.1):
        super().__init__()
        
        self.num_classes = num_classes or {'classification': 5, 'recognition': 3}
        self.temperature = temperature
        
        # 为每个学生层创建CRD蒸馏器
        self.crd_distillers = nn.ModuleDict()
        
        for layer_name, dim in student_dims.items():
            # 创建CRD蒸馏器
            self.crd_distillers[layer_name] = CRDDistiller(
                feature_dim=dim,
                num_classes=None,  # 使用全局类别数
                temperature=temperature
            )
        
        # 为每个任务创建特定的CRD蒸馏器（使用类别数）
        self.task_crd_distillers = nn.ModuleDict()
        for task_name, num_cls in self.num_classes.items():
            # 使用final层的维度
            final_dim = student_dims.get('final', student_dims.get('default', 1280))
            self.task_crd_distillers[task_name] = CRDDistiller(
                feature_dim=final_dim,
                num_classes=num_cls,
                temperature=temperature
            )
    
    def forward(self, student_features: Dict[str, torch.Tensor],
                teacher_features: Dict[str, Dict[str, torch.Tensor]],
                dataset_name: str, labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        多教师CRD蒸馏前向传播
        
        Args:
            student_features: 学生特征字典
            teacher_features: 教师特征字典
            dataset_name: 数据集名称
            labels: 类别标签
        
        Returns:
            CRD损失字典
        """
        losses = {}
        
        # 获取任务类型
        if dataset_name in ['APTOS2019']:
            task_name = 'classification'
        elif dataset_name in ['ISIC2017']:
            task_name = 'recognition'
        else:
            task_name = None  # 分割任务不使用任务特定的CRD
        
        # 1. 对所有特征层进行CRD蒸馏
        total_feature_loss = 0.0
        for layer_name, crd in self.crd_distillers.items():
            if layer_name in student_features:
                crd_losses = crd(
                    {layer_name: student_features[layer_name]},
                    teacher_features,
                    labels
                )
                losses.update({f'{task_name}_{k}' if task_name else k: v for k, v in crd_losses.items()})
                total_feature_loss += crd_losses.get('crd_total', 0.0)
        
        # 2. 对分类/识别任务使用任务特定的CRD蒸馏
        if task_name and task_name in self.task_crd_distillers:
            task_crd = self.task_crd_distillers[task_name]
            task_crd_losses = task_crd(
                {'final': student_features.get('final', student_features.get('default'))},
                teacher_features,
                labels
            )
            losses.update({f'{task_name}_{k}': v for k, v in task_crd_losses.items()})
            total_feature_loss += task_crd_losses.get('crd_total', 0.0)
        
        losses['crd_total'] = total_feature_loss
        
        return losses

# ============================================
# 4. 改进的多任务学生模型
# ============================================

class ImprovedMultiTaskStudent(nn.Module):
    """改进的多任务学生模型"""
    
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
        
        # 任务选择器
        feature_dim = self.feature_dims['final']
        self.task_selector = nn.Embedding(3, feature_dim)  # 3个任务类型
        
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
    
    def forward(self, x, dataset_name: str):
        """前向传播"""
        # 清空特征缓存
        self.features = {}
        
        # 任务ID映射
        task_id_map = {
            'BUSI': 0,
            'kvasir_seg': 0,
            'APTOS2019': 1,
            'ISIC2017': 2
        }
        task_id = task_id_map[dataset_name]
        
        # 通过backbone
        features = self.backbone.features(x)
        
        # 获取任务嵌入并添加到特征中
        task_embed = self.task_selector(torch.tensor(task_id).to(x.device))
        task_embed = task_embed.view(1, -1, 1, 1).expand_as(features)
        task_features = features + task_embed
        
        # 根据任务选择解码器
        if task_id == 0:  # 分割任务
            seg_output = self._forward_segmentation(x)
            return seg_output, self.features
        elif task_id == 1:  # 分类任务
            cls_output = self.task_decoders['classification'](task_features)
            return cls_output, self.features
        else:  # 识别任务
            rec_output = self.task_decoders['recognition'](task_features)
            return rec_output, self.features
    
    def _forward_segmentation(self, x):
        """分割任务的前向传播（带跳跃连接）"""
        # 编码器特征
        enc_features = []
        x_in = x
        
        for i, block in enumerate(self.backbone.features):
            x_in = block(x_in)
            if i in [2, 3, 5, 6]:  # 保存中间特征：block2, block3, block5, block6
                enc_features.append(x_in)
        
        # 解码器
        x_dec = enc_features[-1]  # 使用block6的输出（192或232通道）
        for i, decoder_block in enumerate(self.task_decoders['segmentation'][:-1]):
            x_dec = decoder_block(x_dec)
            if i < len(enc_features) - 1:
                # 跳跃连接
                skip = enc_features[-(i+2)]
                # 调整尺寸
                if x_dec.shape[-2:] != skip.shape[-2:]:
                    x_dec = F.interpolate(x_dec, size=skip.shape[-2:], 
                                        mode='bilinear', align_corners=False)
                x_dec = torch.cat([x_dec, skip], dim=1)
        
        # 最终输出层
        output = self.task_decoders['segmentation'][-1](x_dec)

        # 确保输出形状与输入一致
        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return output

# ============================================
# 5. 异构蒸馏损失函数（支持CRD）
# ============================================

class HeterogeneousDistillationLoss(nn.Module):
    """
    优化的异构多教师蒸馏损失，支持传统蒸馏和CRD蒸馏
    """
    
    def __init__(self, temperature: float = 1.5,
                 feature_weight: float = 0.3,
                 output_weight: float = 0.3,
                 task_weight: float = 0.4,
                 crd_weight: float = 0.5,  # CRD蒸馏权重
                 use_crd: bool = True):  # 是否使用CRD蒸馏
        super().__init__()
        self.temperature = temperature
        self.feature_weight = feature_weight
        self.output_weight = output_weight
        self.task_weight = task_weight
        self.crd_weight = crd_weight
        self.use_crd = use_crd
        
        # 任务损失
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.det_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        
    def forward(self, outputs: Dict, targets: torch.Tensor, dataset_name: str, labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        计算总损失，支持传统蒸馏和CRD蒸馏
        
        Args:
            outputs: 包含学生输出、教师输出、特征的字典
            targets: 目标标签
            dataset_name: 数据集名称
            labels: 类别标签（用于CRD蒸馏）
        
        Returns:
            损失字典
        """
        losses = {}
        
        # 1. 任务损失（硬标签）
        student_output = outputs['student_output']
        if dataset_name in ['BUSI', 'kvasir_seg']:
            # 分割任务 - 确保维度正确
            if targets.dim() == 3:
                targets = targets.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
            
            # 确保数据类型匹配
            targets_float = targets.float()
            student_output_float = student_output.float()
            
            # BCE损失
            bce_loss = self.seg_loss(student_output_float, targets_float)
            # Dice损失
            dice_loss = self.dice_loss(student_output_float, targets_float)
            task_loss = 0.5 * bce_loss + 0.5 * dice_loss
        elif dataset_name in ['APTOS2019', 'ISIC2017']:
            # 分类任务
            task_loss = self.cls_loss(student_output.float(), targets)
            labels = targets  # CRD使用分类标签
        losses['task'] = task_loss * self.task_weight
        
        # 2. CRD蒸馏损失（如果启用）
        if self.use_crd and 'crd_losses' in outputs and outputs['crd_losses'] is not None:
            crd_losses = outputs['crd_losses']
            # 加权CRD损失
            losses['crd'] = crd_losses.get('crd_total', 0.0) * self.crd_weight
            # 添加详细的CRD损失
            for key, value in crd_losses.items():
                if key != 'crd_total':
                    losses[f'crd_{key}'] = value * self.crd_weight * 0.1
        
        # 3. 输出级蒸馏损失（使用融合的教师输出）
        if 'teacher_output' in outputs and outputs['teacher_output'] is not None:
            teacher_output = outputs['teacher_output']
            
            # 确保使用单精度
            teacher_output = teacher_output.float()
            student_output = student_output.float()
            
            if dataset_name in ['BUSI', 'kvasir_seg']:
                # 分割任务的蒸馏
                # 教师输出已经是融合后的概率（在_fuse_teacher_outputs中应用了sigmoid）
                # 学生输出需要应用sigmoid
                student_probs = torch.sigmoid(student_output)
                
                # 确保尺寸匹配
                if teacher_output.shape[-2:] != student_output.shape[-2:]:
                    teacher_output = F.interpolate(
                        teacher_output, 
                        size=student_output.shape[-2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # 分割：使用MSE损失（因为教师输出已经是概率）
                output_distill_loss = F.mse_loss(
                    student_probs,
                    teacher_output.detach()
                )
            else:
                # 分类和检测：使用KL散度
                # 教师输出已经是logits（在_fuse_teacher_outputs中转换）
                output_distill_loss = F.kl_div(
                    F.log_softmax(student_output / self.temperature, dim=1),
                    F.softmax(teacher_output.detach() / self.temperature, dim=1),
                    reduction='batchmean'
                ) * (self.temperature ** 2)
            losses['output_distill'] = output_distill_loss * self.output_weight
        
        # 4. 特征级蒸馏损失（使用所有教师的特征）- 仅在非CRD模式下使用
        if not self.use_crd and 'aligned_features' in outputs and 'teacher_features' in outputs:
            feature_losses = []
            
            # 遍历所有教师模型的特征
            for teacher_name, teacher_feats in outputs['teacher_features'].items():
                # 对于每个教师的特征，与对齐后的学生特征计算损失
                if isinstance(teacher_feats, dict):
                    # 如果教师特征是字典（多层特征）
                    for layer_name in outputs['aligned_features']:
                        if layer_name in teacher_feats:
                            s_feat = outputs['aligned_features'][layer_name].float()
                            t_feat = teacher_feats[layer_name].float()
                            
                            # 处理3D张量（Transformer特征）转换为4D
                            if t_feat.dim() == 3:
                                # [B, seq_len, hidden_dim] -> [B, hidden_dim, H, W]
                                B, seq_len, hidden_dim = t_feat.shape
                                H = W = int(seq_len ** 0.5)
                                t_feat = t_feat.transpose(1, 2).reshape(B, hidden_dim, H, W)
                            
                            # 获取学生特征的通道数
                            s_C = s_feat.shape[1]
                            
                            # 将教师特征投影到学生特征空间（如果通道数不匹配）
                            if t_feat.shape[1] != s_C:
                                # 使用1x1卷积将教师特征投影到学生特征空间
                                if not hasattr(self, 'teacher_proj_' + teacher_name + '_' + layer_name):
                                    # 动态创建投影层
                                    proj = nn.Conv2d(t_feat.shape[1], s_C, 1).to(t_feat.device)
                                    setattr(self, 'teacher_proj_' + teacher_name + '_' + layer_name, proj)
                                proj = getattr(self, 'teacher_proj_' + teacher_name + '_' + layer_name)
                                t_feat = proj(t_feat)
                            
                            # 确保特征尺寸匹配
                            if s_feat.shape != t_feat.shape:
                                if len(t_feat.shape) == 4 and len(s_feat.shape) == 4:
                                    t_feat = F.interpolate(t_feat, size=s_feat.shape[-2:], 
                                                         mode='bilinear', align_corners=False)
                            
                            # 计算特征损失
                            feat_loss = F.mse_loss(s_feat, t_feat.detach())
                            feature_losses.append(feat_loss)
                else:
                    # 如果教师特征是单个张量
                    for layer_name, s_feat in outputs['aligned_features'].items():
                        s_feat = s_feat.float()
                        t_feat = teacher_feats.float()
                        
                        # 处理3D张量（Transformer特征）转换为4D
                        if t_feat.dim() == 3:
                            # [B, seq_len, hidden_dim] -> [B, hidden_dim, H, W]
                            B, seq_len, hidden_dim = t_feat.shape
                            H = W = int(seq_len ** 0.5)
                            t_feat = t_feat.transpose(1, 2).reshape(B, hidden_dim, H, W)
                        
                        # 获取学生特征的通道数
                        s_C = s_feat.shape[1]
                        
                        # 将教师特征投影到学生特征空间（如果通道数不匹配）
                        if t_feat.shape[1] != s_C:
                            # 使用1x1卷积将教师特征投影到学生特征空间
                            if not hasattr(self, 'teacher_proj_' + teacher_name + '_' + layer_name):
                                # 动态创建投影层
                                proj = nn.Conv2d(t_feat.shape[1], s_C, 1).to(t_feat.device)
                                setattr(self, 'teacher_proj_' + teacher_name + '_' + layer_name, proj)
                            proj = getattr(self, 'teacher_proj_' + teacher_name + '_' + layer_name)
                            t_feat = proj(t_feat)
                        
                        # 确保特征尺寸匹配
                        if s_feat.shape != t_feat.shape:
                            if len(t_feat.shape) == 4 and len(s_feat.shape) == 4:
                                t_feat = F.interpolate(t_feat, size=s_feat.shape[-2:], 
                                                     mode='bilinear', align_corners=False)
                        
                        # 计算特征损失
                        feat_loss = F.mse_loss(s_feat, t_feat.detach())
                        feature_losses.append(feat_loss)
            
            if feature_losses:
                losses['feature_distill'] = sum(feature_losses) / len(feature_losses) * self.feature_weight
        
        # 总损失
        losses['total'] = sum(losses.values())
        
        return losses


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
        save_dir: str = "./experiments"
    ):
        self.student_model = student_model.to(device)
        self.teacher_models = {k: v.to(device).eval() for k, v in teacher_models.items()}
        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.config = config
        self.device = device
        
        # 创建保存目录
        self.experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = Path(save_dir) / self.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)
        
        # 初始化优化器和调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 损失函数
        self.criterion = HeterogeneousDistillationLoss(
            temperature=config.get('temperature', 2.0),
            feature_weight=config.get('feature_weight', 0.01),
            output_weight=config.get('output_weight', 0.05),
            task_weight=config.get('task_weight', 0.94),
            crd_weight=config.get('crd_weight', 0.5),  # CRD蒸馏权重
            use_crd=config.get('use_crd', True)  # 是否使用CRD蒸馏
        )
        
        # 跨任务知识蒸馏器
        self.cross_task_distiller = self._create_cross_task_distiller()
        
        # CRD多教师蒸馏器
        self.crd_distiller = self._create_crd_distiller()
        
        # 训练记录
        self.train_history = {
            'losses': {},
            'metrics': {},
            'learning_rates': []
        }
        
        # TensorBoard
        self.writer = SummaryWriter(self.save_dir / "tensorboard")
        
        # 早停机制
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 15),
            min_delta=config.get('early_stopping_delta', 0.001)
        )
        
        # 最佳模型跟踪
        self.best_metrics = {}
        self.best_epoch = 0
        
        # 任务映射
        self.task_mapping = {
            'BUSI': ('segmentation_busi', 'USFM'),
            'kvasir_seg': ('segmentation_kvasir', 'MedSAM'),
            'APTOS2019': ('classification', 'RETFound'),
            'ISIC2017': ('recognition', 'BioMedPhrase')
        }
        
    def _create_optimizer(self):
        """创建优化器（使用差异化学习率）"""
        param_groups = [
            {'params': self.student_model.backbone.parameters(), 
            'lr': self.config['learning_rate'] * 0.1},
            {'params': self.student_model.task_decoders.parameters(), 
            'lr': self.config['learning_rate']},
            # 添加任务选择器参数
            {'params': self.student_model.task_selector.parameters(), 
            'lr': self.config['learning_rate']}
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
        """创建跨任务蒸馏器"""
        # 获取教师模型的特征维度
        teacher_dims = {}
        for name, teacher in self.teacher_models.items():
            # 根据实际的教师特征维度设置
            if name == 'MedSAM':
                # MedSAM特征：[B, 256, 64, 64]
                teacher_dims[name] = {
                    'stem': 256, 'block2': 256, 'block3': 256, 
                    'block5': 256, 'final': 256
                }
            elif name == 'RETFound_MAE':
                # RETFound_MAE特征：[B, 196, 768]
                teacher_dims[name] = {
                    'stem': 768, 'block2': 768, 'block3': 768, 
                    'block5': 768, 'final': 768
                }
            elif name == 'USFM':
                # USFM特征：可能没有，使用默认值
                teacher_dims[name] = {
                    'stem': 256, 'block2': 512, 'block3': 512, 
                    'block5': 1024, 'final': 2048
                }
            else:
                # 默认维度
                teacher_dims[name] = {
                    'stem': 256, 'block2': 512, 'block3': 512, 
                    'block5': 1024, 'final': 2048
                }
        
        # 使用学生模型的实际特征维度
        student_dims = self.student_model.feature_dims if hasattr(self.student_model, 'feature_dims') else {
            'default': 1280, 'stem': 32, 'block1': 16, 
            'block2': 24, 'block3': 40, 'block4': 80, 
            'block5': 112, 'block6': 192, 'block7': 320, 
            'final': 1280
        }
        
        print(f"Creating CrossTaskKnowledgeDistiller:")
        print(f"  Student dims: {student_dims}")
        print(f"  Teacher dims: {teacher_dims}")
        
        return CrossTaskKnowledgeDistiller(student_dims, teacher_dims).to(self.device)
    
    def _create_crd_distiller(self):
        """创建CRD多教师蒸馏器"""
        # 获取教师模型的特征维度
        teacher_dims = {}
        for name, teacher in self.teacher_models.items():
            if name == 'MedSAM':
                teacher_dims[name] = {
                    'stem': 256, 'block2': 256, 'block3': 256, 
                    'block5': 256, 'final': 256
                }
            elif name == 'RETFound_MAE':
                teacher_dims[name] = {
                    'stem': 768, 'block2': 768, 'block3': 768, 
                    'block5': 768, 'final': 768
                }
            elif name == 'USFM':
                teacher_dims[name] = {
                    'stem': 256, 'block2': 512, 'block3': 512, 
                    'block5': 1024, 'final': 2048
                }
            else:
                teacher_dims[name] = {
                    'stem': 256, 'block2': 512, 'block3': 512, 
                    'block5': 1024, 'final': 2048
                }
        
        # 使用学生模型的实际特征维度
        student_dims = self.student_model.feature_dims if hasattr(self.student_model, 'feature_dims') else {
            'default': 1280, 'stem': 32, 'block1': 16, 
            'block2': 24, 'block3': 40, 'block4': 80, 
            'block5': 112, 'block6': 192, 'block7': 320, 
            'final': 1280
        }
        
        # 获取类别数
        num_classes = self.config.get('num_classes', {
            'classification': 5,
            'recognition': 3
        })
        
        print(f"Creating CRDMultiTeacherDistiller:")
        print(f"  Student dims: {student_dims}")
        print(f"  Teacher dims: {teacher_dims}")
        print(f"  Num classes: {num_classes}")
        
        crd_distiller = CRDMultiTeacherDistiller(
            student_dims=student_dims,
            teacher_dims=teacher_dims,
            num_classes=num_classes,
            temperature=self.config.get('crd_temperature', 0.1)
        ).to(self.device)
        
        return crd_distiller
    
    def train(self, dataset_name: str):
        """针对单个数据集的训练流程"""
        print(f"\n{'='*80}")
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
        """训练一个epoch（针对单个数据集）"""
        self.student_model.train()
        
        loader = self.train_loaders[dataset_name]
        print(f"Training on dataset: {dataset_name}")
        
        losses = {'total': 0, 'task': 0, 'output_distill': 0, 'feature_distill': 0}
        num_batches = 0
        
        # 根据任务类型调整学习率
        if dataset_name in ['BUSI', 'kvasir_seg']:
            lr = self.config['learning_rate'] * 1.5  # 分割任务更高学习率
        else:
            lr = self.config['learning_rate']
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # 进度条
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} - {dataset_name}")
        
        for batch_idx, (images, targets_list) in enumerate(pbar):
            try:
                images = images.to(self.device)
                
                # 准备该批次的targets (根据任务类型)
                if dataset_name in ['BUSI', 'kvasir_seg']:  # 分割任务
                    # 提取掩码并转换为张量
                    mask_list = []
                    for target in targets_list:
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
                    
                    targets = torch.stack(mask_list).to(self.device)
                elif dataset_name in ['APTOS2019', 'ISIC2017']:  # 分类任务
                    # 提取标签
                    label_list = []
                    for idx, target in enumerate(targets_list):
                        try:
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
                                label_list.append(int(label))
                            else:
                                label_list.append(0)
                        except Exception as e:
                            label_list.append(0)
                    
                    targets = torch.tensor(label_list, dtype=torch.long).to(self.device)
                
                # 前向传播（使用所有教师模型）
                outputs = self._forward_pass(images, targets, dataset_name, None, verbose=False)
                
                # 计算损失（传递labels用于CRD）
                labels = targets if dataset_name in ['APTOS2019', 'ISIC2017'] else None
                loss_dict = self.criterion(outputs, targets, dataset_name, labels=labels)
                
                # 梯度累积
                accumulation_steps = 4
                loss_dict['total'] = loss_dict['total'] / accumulation_steps
                
                # 反向传播
                loss_dict['total'].backward()
    
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # 记录损失
                for k, v in loss_dict.items():
                    losses[k] += v.item()
                num_batches += 1
                
                # 更新进度条
                pbar.set_postfix({k: v/num_batches for k, v in losses.items()})
            
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                self.optimizer.zero_grad()
                continue
            finally:
                gc.collect()
                torch.cuda.empty_cache()
    
        # 计算平均损失
        if num_batches > 0:
            epoch_losses = {k: v/num_batches for k, v in losses.items()}
        else:
            epoch_losses = {'total': 0, 'task': 0, 'output_distill': 0, 'feature_distill': 0}
    
        return epoch_losses    
    def _forward_pass(self, images, targets, dataset_name, teacher_model, verbose=False):
        """执行前向传播 - 支持传统蒸馏和CRD蒸馏"""
        outputs = {}
        
        # 学生模型预测
        student_output, student_features = self.student_model(images, dataset_name)
        
        outputs['student_output'] = student_output
        outputs['student_features'] = student_features
        
        # 收集所有教师模型的输出和特征
        all_teacher_outputs = {}
        all_teacher_features = {}
        
        # 遍历所有教师模型
        with torch.no_grad():
            for teacher_name, teacher_model in self.teacher_models.items():
                try:
                    # 获取教师输出
                    teacher_output, teacher_features = self._get_teacher_output(
                        teacher_model, teacher_name, images, dataset_name
                    )
                    
                    # 根据任务类型决定是否使用教师输出
                    if dataset_name in ['APTOS2019', 'ISIC2017']:
                        # 分类任务：教师模型通常不提供分类输出
                        # 只使用特征进行蒸馏，不使用输出蒸馏
                        if teacher_features is not None:
                            all_teacher_features[teacher_name] = teacher_features
                            if verbose:
                                print(f"  ✓ {teacher_name}: Using features (shape: {teacher_features.shape})")
                        else:
                            if verbose:
                                print(f"  ✗ {teacher_name}: No features available")
                        # 跳过输出
                        teacher_output = None
                    elif dataset_name in ['BUSI', 'kvasir_seg']:
                        # 分割任务：可以使用教师的分割输出
                        if teacher_output is not None:
                            # 检查输出形状是否合理
                            if teacher_output.dim() in [3, 4]:
                                all_teacher_outputs[teacher_name] = teacher_output
                                if verbose:
                                    print(f"  ✓ {teacher_name}: Using output (shape: {teacher_output.shape})")
                            else:
                                if verbose:
                                    print(f"  ✗ {teacher_name}: Unexpected output shape: {teacher_output.shape}")
                        else:
                            if verbose:
                                print(f"  ✗ {teacher_name}: No output available")
                        
                        if teacher_features is not None:
                            all_teacher_features[teacher_name] = teacher_features
                            if verbose:
                                print(f"  ✓ {teacher_name}: Using features (shape: {teacher_features.shape})")
                        else:
                            if verbose:
                                print(f"  ✗ {teacher_name}: No features available")
                        
                except Exception as e:
                    if verbose:
                        print(f"  ✗ {teacher_name}: Error - {e}")
                    continue
        
        # 融合所有教师的输出
        if all_teacher_outputs:
            outputs['teacher_output'] = self._fuse_teacher_outputs(
                all_teacher_outputs, dataset_name, student_output.shape
            )
            if verbose:
                print(f"  ✓ Fused {len(all_teacher_outputs)} teacher outputs")
        
        # CRD蒸馏损失（如果启用）
        if self.criterion.use_crd and all_teacher_features and hasattr(self, 'crd_distiller'):
            if verbose:
                print(f"  Computing CRD distillation loss...")
            
            # 获取标签用于CRD（分类任务）
            labels = None
            if dataset_name in ['APTOS2019', 'ISIC2017']:
                labels = targets  # 分类任务使用targets作为标签
            
            try:
                # 计算CRD损失
                crd_losses = self.crd_distiller(
                    student_features=student_features,
                    teacher_features=all_teacher_features,
                    dataset_name=dataset_name,
                    labels=labels
                )
                outputs['crd_losses'] = crd_losses
                if verbose:
                    print(f"  ✓ CRD losses: {crd_losses}")
            except Exception as e:
                if verbose:
                    print(f"  ✗ CRD distillation error: {e}")
                outputs['crd_losses'] = {'crd_total': torch.tensor(0.0, device=self.device)}
        
        # 传统特征蒸馏（如果CRD未启用）
        if not self.criterion.use_crd and all_teacher_features and hasattr(self, 'cross_task_distiller'):
            if verbose:
                print(f"  Student features: {list(student_features.keys())}")
                print(f"  Teacher features: {list(all_teacher_features.keys())}")
                for teacher_name, feats in all_teacher_features.items():
                    if isinstance(feats, dict):
                        print(f"    {teacher_name}: {list(feats.keys())}")
                    else:
                        print(f"    {teacher_name}: single tensor shape {feats.shape}")
            
            # 确保输入特征都是单精度
            student_features_float = {k: v.float() for k, v in student_features.items()}
            teacher_features_float = {k: v.float() for k, v in all_teacher_features.items()}
            
            # 使用所有激活的教师进行特征对齐
            aligned_features = self.cross_task_distiller(
                student_features_float, 
                teacher_features_float,
                list(all_teacher_features.keys())
            )
            
            # 确保输出是单精度
            aligned_features = {k: v.float() for k, v in aligned_features.items()}
            outputs['aligned_features'] = aligned_features
            outputs['teacher_features'] = teacher_features_float
            if verbose:
                print(f"  ✓ Fused {len(all_teacher_features)} teacher features, aligned: {list(aligned_features.keys())}")
        elif verbose:
            print(f"  ✗ No feature fusion (missing features or distiller)")
        
        return outputs
    
    def _get_teacher_output(self, teacher_model, teacher_name, images, dataset_name):
        """获取单个教师模型的输出和特征"""
        teacher_output = None
        teacher_features = None
        
        # MedSAM需要特定格式的输入（字典列表）
        if teacher_name == 'MedSAM' and hasattr(teacher_model, 'model'):
            # 将张量转换为MedSAM期望的格式
            mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
            denorm_images = images * std + mean
            denorm_images = denorm_images * 255.0
            
            # 转换为 numpy 数组 (HxWxC)
            numpy_images = denorm_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            
            medsam_input = [{"image": img} for img in numpy_images]
            try:
                teacher_output = teacher_model(medsam_input, multimask_output=False)
                if teacher_output is not None:
                    teacher_output = teacher_output.float()
            except Exception as e:
                print(f"MedSAM inference error: {e}")
                teacher_output = None
            
            # 获取特征
            try:
                medsam_feat_input = [
                    {"image": img.permute(1, 2, 0).cpu().numpy()}
                    for img in images
                ]
                teacher_features = teacher_model.extract_features(medsam_feat_input)
                if teacher_features is not None:
                    teacher_features = teacher_features.float()
            except Exception as e:
                print(f"MedSAM feature extraction error: {e}")
                teacher_features = None
        
        elif teacher_name == 'RETFound_MAE':
            # RETFound_MAE需要224x224的输入
            try:
                # 调整图像尺寸到224x224
                images_224 = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
                
                # RETFound_MAE是MAE模型，主要用于特征提取
                if hasattr(teacher_model, 'forward'):
                    result = teacher_model(images_224)
                    # RETFound_MAE可能返回tuple，需要提取正确的输出
                    if isinstance(result, tuple):
                        # 通常返回 (loss, pred, mask) 或类似的结构
                        # 我们需要提取特征或预测
                        if len(result) > 1:
                            teacher_output = result[1]  # 通常第二个元素是预测或特征
                        else:
                            teacher_output = result[0]
                    else:
                        teacher_output = result
                    
                    # 确保teacher_output是tensor而不是tuple
                    if isinstance(teacher_output, tuple):
                        teacher_output = teacher_output[0] if len(teacher_output) > 0 else None
                    
                    if teacher_output is not None and isinstance(teacher_output, torch.Tensor):
                        teacher_output = teacher_output.float()
                    else:
                        teacher_output = None
                
                # 获取特征
                if hasattr(teacher_model, 'forward'):
                    result = teacher_model(images_224)
                    if isinstance(result, tuple):
                        if len(result) > 1:
                            teacher_features = result[1]
                        else:
                            teacher_features = result[0]
                    else:
                        teacher_features = result
                    
                    # 确保teacher_features是tensor而不是tuple
                    if isinstance(teacher_features, tuple):
                        teacher_features = teacher_features[0] if len(teacher_features) > 0 else None
                    
                    if teacher_features is not None and isinstance(teacher_features, torch.Tensor):
                        teacher_features = teacher_features.float()
                    else:
                        teacher_features = None
                
            except Exception as e:
                print(f"RETFound_MAE inference error: {e}")
                teacher_output = None
                teacher_features = None
        
        elif teacher_name == 'BioMedPrase':
            # BioMedPrase需要特殊处理
            try:
                # BioMedPrase期望的输入格式可能是字典或列表
                # 先尝试直接传递
                if hasattr(teacher_model, 'forward'):
                    # 尝试多种输入格式
                    try:
                        # 格式1: 直接传递张量
                        teacher_output = teacher_model(images)
                    except:
                        try:
                            # 格式2: 字典格式
                            teacher_output = teacher_model({'image': images})
                        except:
                            try:
                                # 格式3: 列表格式
                                teacher_output = teacher_model([{'image': img} for img in images])
                            except:
                                # 格式4: 仅特征提取
                                teacher_output = None
                    
                    if teacher_output is not None:
                        teacher_output = teacher_output.float()
                
                # 获取特征
                if hasattr(teacher_model, 'forward'):
                    try:
                        teacher_features = teacher_model(images)
                        if teacher_features is not None:
                            teacher_features = teacher_features.float()
                    except:
                        teacher_features = None
                
            except Exception as e:
                print(f"BioMedPrase inference error: {e}")
                teacher_output = None
                teacher_features = None
        
        elif teacher_name == 'USFM':
            # USFM模型处理
            try:
                if hasattr(teacher_model, 'forward'):
                    teacher_output = teacher_model(images)
                    if teacher_output is not None:
                        teacher_output = teacher_output.float()
                
                # 获取特征
                if hasattr(teacher_model, 'model'):
                    if hasattr(teacher_model.model, 'encoder'):
                        # 从编码器获取特征
                        with torch.no_grad():
                            teacher_features = teacher_model.model.encoder(images)
                        if teacher_features is not None:
                            teacher_features = teacher_features.float()
                
            except Exception as e:
                print(f"USFM inference error: {e}")
                teacher_output = None
                teacher_features = None
        
        else:
            # 其他教师模型的标准处理
            try:
                # 尝试获取输出
                if hasattr(teacher_model, 'forward'):
                    teacher_output = teacher_model(images)
                    if teacher_output is not None:
                        teacher_output = teacher_output.float()
                
                # 尝试获取特征
                if hasattr(teacher_model, 'extract_features'):
                    teacher_features = teacher_model.extract_features(images)
                    if teacher_features is not None:
                        teacher_features = teacher_features.float()
                elif hasattr(teacher_model, 'model'):
                    # 对于包装模型，尝试从内部模型获取特征
                    if hasattr(teacher_model.model, 'extract_features'):
                        teacher_features = teacher_model.model.extract_features(images)
                        if teacher_features is not None:
                            teacher_features = teacher_features.float()
                
            except Exception as e:
                print(f"Teacher {teacher_name} inference error: {e}")
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
    
    def _validate_epoch(self, epoch: int, dataset_name: str) -> Dict[str, float]:
        """验证一个epoch（针对单个数据集）"""
        print(f"Starting validation for epoch {epoch + 1}")
        self.student_model.eval()
        
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
        print(f"\nEpoch {epoch+1}/{self.config['num_epochs']} - {dataset_name} - LR: {lr:.6f}")
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
            row['Inference_Time_ms'] = f"{metrics['Mean_Inference_Time']*1000:.2f} ± {metrics['Std_Inference_Time']*1000:.2f}"
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
        inference_times = [results[d]['Mean_Inference_Time']*1000 for d in datasets]
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
    """主函数：执行完整的多任务多教师知识蒸馏训练流程（支持CRD）"""
    
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
        'crd_weight': 0.5,  # CRD蒸馏权重
        'crd_temperature': 0.1,  # CRD温度参数
        'use_crd': True,  # 启用CRD蒸馏
        'early_stopping_patience': 20,
        'early_stopping_delta': 0.001,
        'num_classes': {
            'classification': 5,  # APTOS2019
            'recognition': 3      # ISIC2017
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
                "labels":["melanoma", "seborrheic_keratosis", "nevus/benign pigmented lesion"]
            },
            "APTOS2019": {
                "data_dir": "/root/autodl-tmp/datasets/APTOS2019",
                "task_type": "classification",
                "num_classes": 5,
                "labels":['anodr', 'bmilddr', 'cmoderatedr', 'dseveredr', 'eproliferativedr'],
            },
            "kvasir_seg":{
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
    
    # 创建学生模型
    student_model = ImprovedMultiTaskStudent(config['num_classes'])
    student_model = student_model.to(device)
    print(f"Student model parameters: {sum(p.numel() for p in student_model.parameters()) / 1e6:.2f}M")
    
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
    
    # 为每个数据集独立训练
    print("\n" + "=" * 80)
    print("Starting independent training for each dataset")
    print("=" * 80)
    
    for dataset_name in dataset_names:
        print(f"\n{'#'*80}")
        print(f"# Training dataset: {dataset_name}")
        print(f"{'#'*80}")
        
        # 为每个数据集创建独立的学生模型
        student_model = ImprovedMultiTaskStudent(config['num_classes'])
        student_model = student_model.to(device)
        print(f"Student model parameters: {sum(p.numel() for p in student_model.parameters()) / 1e6:.2f}M")
        
        # 创建训练器（只包含当前数据集的加载器）
        trainer = MultiTeacherDistillationTrainer(
            student_model=student_model,
            teacher_models=teacher_models,
            train_loaders={dataset_name: dataloaders['train'][dataset_name]},
            val_loaders={dataset_name: dataloaders['val'][dataset_name]},
            config=config,
            device=device,
            save_dir=f"./experiments/multi_teacher_distillation/{dataset_name}"
        )
        
        # 开始训练当前数据集
        trainer.train(dataset_name)
        
        # 清理内存
        del student_model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 80)
    print("All datasets training completed!")
    print("=" * 80)


if __name__ == "__main__":
    start_train() 
    
