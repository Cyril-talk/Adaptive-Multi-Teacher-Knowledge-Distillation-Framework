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
        self.query = nn.Conv2d(dim, dim, 1)
        self.key = nn.Conv2d(dim, dim, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.scale = dim ** -0.5
        
    def forward(self, student_feat: torch.Tensor, 
                teacher_feats: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """使用注意力机制融合多个教师的知识"""
        B, C, H, W = student_feat.shape
        
        # Query来自学生特征
        q = self.query(student_feat).view(B, C, -1).transpose(1, 2)  # B, HW, C
        
        # 融合所有教师的知识
        all_values = []
        all_weights = []
        
        for s_feat_proj, t_feat in teacher_feats:
            # 确保特征尺寸匹配
            if t_feat.shape[-2:] != (H, W):
                t_feat = F.interpolate(t_feat, size=(H, W), mode='bilinear', align_corners=False)
            
            k = self.key(t_feat).view(B, C, -1).transpose(1, 2)  # B, HW, C
            v = self.value(t_feat).view(B, C, -1).transpose(1, 2)  # B, HW, C
            
            # 计算注意力权重
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
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
            fused = fused.view(B, H, W, C).permute(0, 3, 1, 2)
            
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
            for layer_name, t_dim in t_dims.items():
                s_dim = student_dims.get(layer_name, student_dims.get('default', 1280))
                
                # 使用残差连接改进投影
                self.projectors[teacher_name][layer_name] = nn.Sequential(
                    nn.Conv2d(s_dim, s_dim, 3, padding=1),
                    nn.BatchNorm2d(s_dim),
                    nn.ReLU(),
                    nn.Conv2d(s_dim, t_dim, 1),
                    nn.BatchNorm2d(t_dim)
                )
        
        # 注意力模块，用于融合不同教师的知识
        self.attention_modules = nn.ModuleDict()
        for layer_name in ['low', 'mid', 'high', 'final']:
            self.attention_modules[layer_name] = CrossTeacherAttention(
                student_dims.get(layer_name, student_dims.get('default', 1280))
            )
    
    def forward(self, student_features: Dict[str, torch.Tensor], 
                teacher_features: Dict[str, Dict[str, torch.Tensor]],
                active_teachers: List[str]) -> Dict[str, torch.Tensor]:
        """对齐学生和教师特征"""
        aligned_features = {}
        
        for layer_name in student_features:
            if layer_name in self.attention_modules:
                # 收集所有激活教师的特征
                teacher_feats = []
                for teacher_name in active_teachers:
                    if teacher_name in teacher_features and layer_name in teacher_features[teacher_name]:
                        t_feat = teacher_features[teacher_name][layer_name]
                        # 投影学生特征到教师特征空间
                        s_feat_proj = self.projectors[teacher_name][layer_name](student_features[layer_name])
                        teacher_feats.append((s_feat_proj, t_feat))
                
                if teacher_feats:
                    # 使用注意力机制融合知识
                    aligned_feat = self.attention_modules[layer_name](
                        student_features[layer_name], teacher_feats
                    )
                    aligned_features[layer_name] = aligned_feat
                else:
                    aligned_features[layer_name] = student_features[layer_name]
        
        return aligned_features


# ============================================
# 3. 改进的多任务学生模型
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
# 4. 异构蒸馏损失函数
# ============================================

class HeterogeneousDistillationLoss(nn.Module):
    """优化的异构多教师蒸馏损失"""
    
    def __init__(self, temperature: float = 1.5,  # 降低温度
                 feature_weight: float = 0.3,       # 降低特征蒸馏权重
                 output_weight: float = 0.3,        # 降低输出蒸馏权重
                 task_weight: float = 0.4):         # 提高任务损失权重
        super().__init__()
        self.temperature = temperature
        self.feature_weight = feature_weight
        self.output_weight = output_weight
        self.task_weight = task_weight
        
        # 任务损失
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.det_loss = nn.CrossEntropyLoss()  # 简化的检测损失
        self.dice_loss = DiceLoss()
        
    def forward(self, outputs: Dict, targets: torch.Tensor, dataset_name: str) -> Dict[str, torch.Tensor]:
        """计算总损失"""
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
        losses['task'] = task_loss * self.task_weight
        
        # 2. 输出级蒸馏损失
        if 'teacher_output' in outputs and outputs['teacher_output'] is not None:
            teacher_output = outputs['teacher_output']
            
            # 确保使用单精度
            teacher_output = teacher_output.float()
            student_output = student_output.float()
            
            if dataset_name in ['BUSI', 'kvasir_seg']:
                # 分割任务的蒸馏 - 调整教师输出尺寸以匹配学生输出
                if teacher_output.dim() == 4 and teacher_output.shape[1] > 1:
                    # 如果是多通道输出，取第一个通道
                    teacher_output = teacher_output[:, 0:1, :, :]
                    
                if teacher_output.shape[-2:] != student_output.shape[-2:]:
                    teacher_output = F.interpolate(
                        teacher_output, 
                        size=student_output.shape[-2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # 分割：使用MSE
                output_distill_loss = F.mse_loss(
                    torch.sigmoid(student_output),
                    torch.sigmoid(teacher_output).detach()
                )
            else:
                # 分类和检测：使用KL散度
                output_distill_loss = F.kl_div(
                    F.log_softmax(student_output / self.temperature, dim=1),
                    F.softmax(teacher_output.detach() / self.temperature, dim=1),
                    reduction='batchmean'
                ) * (self.temperature ** 2)
            losses['output_distill'] = output_distill_loss * self.output_weight
        
        # 3. 特征级蒸馏损失
        if 'aligned_features' in outputs and 'teacher_features' in outputs:
            feature_losses = []
            for layer_name in outputs['aligned_features']:
                if layer_name in outputs['teacher_features']:
                    s_feat = outputs['aligned_features'][layer_name].float()
                    t_feat = outputs['teacher_features'][layer_name].float()
                    
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
            temperature=config.get('temperature', 2.0),  # 降低温度
            feature_weight=config.get('feature_weight', 0.1),  # 降低特征蒸馏权重
            output_weight=config.get('output_weight', 0.2),  # 降低输出蒸馏权重
            task_weight=config.get('task_weight', 0.7)  # 提高任务损失权重
        )
        
        # 跨任务知识蒸馏器
        self.cross_task_distiller = self._create_cross_task_distiller()
        
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
            # 这里需要根据实际教师模型调整
            if hasattr(teacher, 'get_feature_dims'):
                teacher_dims[name] = teacher.get_feature_dims()
            else:
                # 默认维度
                teacher_dims[name] = {'low': 256, 'mid': 512, 'high': 1024, 'final': 2048}
        
        student_dims = self.student_model.feature_dims if hasattr(self.student_model, 'feature_dims') else {
            'default': 1280, 'low': 320, 'mid': 512, 'high': 1024, 'final': 1280
        }
        
        return CrossTaskKnowledgeDistiller(student_dims, teacher_dims).to(self.device)
    
    def train(self):
        """完整的训练流程"""
        print(f"Starting training for {self.config['num_epochs']} epochs")
        print(f"Experiment: {self.experiment_name}")
        print("-" * 80)
        
        for epoch in range(self.config['num_epochs']):
            # 训练阶段
            train_losses = self._train_epoch(epoch)
            
            # 验证阶段
            val_metrics = self._validate_epoch(epoch)
            
            # 更新学习率
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                avg_loss = np.mean([losses['total'] for losses in train_losses.values()])
                self.scheduler.step(avg_loss)
            else:
                self.scheduler.step()
            
            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.train_history['learning_rates'].append(current_lr)
            
            # 打印epoch总结
            self._print_epoch_summary(epoch, train_losses, val_metrics, current_lr)
            
            # 保存检查点
            self._save_checkpoint(epoch, val_metrics)
            
            # 早停检查
            avg_metric = self._calculate_average_metric(val_metrics)
            self.early_stopping(avg_metric)
            if self.early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        # 训练结束后的操作
        self._on_training_end()
    
    def _train_epoch(self, epoch: int) -> Dict[str, Dict[str, float]]:
        """训练一个epoch"""
        self.student_model.train()
        epoch_losses = {}
        
        # 遍历所有数据集
        for dataset_name, loader in self.train_loaders.items():
            print(f"Training on dataset: {dataset_name}")
            losses = {'total': 0, 'task': 0, 'output_distill': 0, 'feature_distill': 0}
            num_batches = 0
            
            if dataset_name in ['BUSI', 'kvasir_seg']:
                lr = self.config['learning_rate'] * 1.5  # 分割任务更高学习率
            else:
                lr = self.config['learning_rate']
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # 获取对应的教师模型
            _, teacher_name = self.task_mapping.get(dataset_name, (None, None))
            teacher_model = self.teacher_models.get(teacher_name)
            
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
                                mask_list.append(target['mask'])
                            else:
                                _, _, H, W = images.shape
                                mask_list.append(torch.zeros((H, W), dtype=torch.long))
                        
                        targets = torch.stack(mask_list).to(self.device)
                    elif dataset_name in ['APTOS2019', 'ISIC2017']:  # 分类任务
                        # 提取标签
                        label_list = []
                        for target in targets_list:
                            if 'class_label' in target:
                                label_list.append(target['class_label'])
                            else:
                                label_list.append(0)
                        
                        targets = torch.tensor(label_list, dtype=torch.long).to(self.device)
                    
                    # 前向传播
                    outputs = self._forward_pass(images, targets, dataset_name, teacher_model)
                    
                    # 计算损失
                    loss_dict = self.criterion(outputs, targets, dataset_name)
                    
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
                epoch_losses[dataset_name] = {k: v/num_batches for k, v in losses.items()}

        print(f"Epoch[{epoch + 1}] Losses: ", epoch_losses)
        return epoch_losses
    
    def _forward_pass(self, images, targets, dataset_name, teacher_model):
        """执行前向传播"""
        outputs = {}
        
        # 学生模型预测
        student_output, student_features = self.student_model(images, dataset_name)
        outputs['student_output'] = student_output
        outputs['student_features'] = student_features
        
        # 教师模型预测
        if teacher_model is not None:
            with torch.no_grad():
                # 获取教师输出
                _, teacher_name = self.task_mapping.get(dataset_name, (None, None))
                
                # MedSAM需要特定格式的输入（字典列表）
                if teacher_name == 'MedSAM' and hasattr(teacher_model, 'model'):
                    # 将张量转换为MedSAM期望的格式
                    # 反归一化到 [0, 255]
                    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
                    denorm_images = images * std + mean
                    denorm_images = denorm_images * 255.0
                    
                    # 转换为 numpy 数组 (HxWxC)
                    numpy_images = denorm_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                    
                    medsam_input = [{"image": img} for img in numpy_images]
                    try:
                        teacher_output = teacher_model(medsam_input, multimask_output=False)
                    except Exception as e:
                        print(f"MedSAM inference error: {e}")
                        teacher_output = None
                else:
                    try:
                        teacher_output = teacher_model(images)
                    except Exception as e:
                        print(f"Teacher model inference error: {e}")
                        teacher_output = None
                    
                # 确保教师输出是单精度
                if teacher_output is not None:
                    teacher_output = teacher_output.float()
                outputs['teacher_output'] = teacher_output
                
                # 获取教师特征（如果支持）
                if hasattr(teacher_model, 'extract_features'):
                    # MedSAM的特征提取也需要特殊处理
                    if teacher_name == 'MedSAM':
                        try:
                            medsam_feat_input = [
                                {"image": img.permute(1, 2, 0).cpu().numpy()}
                                for img in images
                            ]
                            teacher_features = teacher_model.extract_features(medsam_feat_input)
                        except Exception as e:
                            print(f"MedSAM feature extraction error: {e}")
                            teacher_features = None
                    else:
                        try:
                            teacher_features = teacher_model.extract_features(images)
                        except Exception as e:
                            print(f"Teacher feature extraction error: {e}")
                            teacher_features = None
                        
                    # 确保教师特征是单精度
                    if teacher_features is not None:
                        teacher_features = teacher_features.float()
                    outputs['teacher_features'] = teacher_features
        
        # 特征对齐（如果有教师特征）
        if hasattr(self, 'cross_task_distiller') and 'teacher_features' in outputs and outputs['teacher_features'] is not None:
            # 确保输入特征都是单精度
            student_features_float = {k: v.float() for k, v in student_features.items()}
            teacher_features_float = {dataset_name: outputs['teacher_features']}
            
            aligned_features = self.cross_task_distiller(
                student_features_float, 
                teacher_features_float,
                [dataset_name]
            )
            
            # 确保输出是单精度
            aligned_features = {k: v.float() for k, v in aligned_features.items()}
            outputs['aligned_features'] = aligned_features
        
        return outputs
    
    def _validate_epoch(self, epoch: int) -> Dict[str, Dict[str, float]]:
        """验证一个epoch"""
        print(f"Starting validation for epoch {epoch + 1}")
        self.student_model.eval()
        val_metrics = {}
        
        with torch.no_grad():
            # 遍历所有数据集
            for dataset_name, loader in self.val_loaders.items():
                try:
                    print(f"Validating dataset: {dataset_name}")
                    if dataset_name in ['BUSI', 'kvasir_seg']:
                        # 分割任务评估
                        metrics = self._evaluate_segmentation(loader, dataset_name)
                    else:
                        # 分类任务评估
                        metrics = self._evaluate_classification(loader, dataset_name)
                    
                    val_metrics[dataset_name] = metrics
                    print(f"Validation metrics for {dataset_name}: {metrics}")
                
                except Exception as e:
                    print(f"Error evaluating dataset {dataset_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # 记录错误但继续验证其他数据集
                    val_metrics[dataset_name] = {"error": str(e)}
        
        # 记录到TensorBoard
        self._log_val_metrics(epoch, val_metrics)
        
        return val_metrics
    
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
                        mask_list.append(target['mask'])
                    else:
                        # 如果没有掩码，创建一个空掩码
                        _, _, H, W = images.shape
                        mask_list.append(torch.zeros((H, W), dtype=torch.long))
                
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

                # 添加调试信息
                if batch_idx == 0:
                    print(f"Debug - Images shape: {images.shape}")
                    print(f"Debug - Outputs shape: {outputs.shape}")
                    print(f"Debug - Preds shape: {preds.shape}")
                    print(f"Debug - Masks shape: {masks.shape}")
                    print(f"Debug - Preds range: [{preds.min():.3f}, {preds.max():.3f}]")
                    print(f"Debug - Masks range: [{masks.min():.3f}, {masks.max():.3f}]")
        
        # 合并所有预测
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # 添加数值稳定性检查
        print(f"Evaluation - Preds shape: {all_preds.shape}, Targets shape: {all_targets.shape}")
        print(f"Evaluation - Preds unique values: {np.unique(all_preds)}")
        print(f"Evaluation - Targets unique values: {np.unique(all_targets)}")
        
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
                        labels.append(target['class_label'])
                    elif 'labels' in target and len(target['labels']) > 0:
                        # 对于检测任务，使用第一个标签
                        labels.append(target['labels'][0])
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
    
    def _calculate_average_metric(self, val_metrics):
        """计算平均验证指标（用于早停）"""
        total_score = 0
        count = 0
        
        for dataset_name, metrics in val_metrics.items():
            if dataset_name in ['BUSI', 'kvasir_seg']:
                # 使用IoU作为主要指标
                total_score += metrics['IoU']
            else:
                # 使用准确率作为主要指标
                total_score += metrics['Accuracy']
            count += 1
        
        return total_score / count if count > 0 else 0
    
    def _save_checkpoint(self, epoch, val_metrics):
        """保存检查点"""
        # 计算当前平均指标
        current_score = self._calculate_average_metric(val_metrics)
        
        # 检查是否是最佳模型
        is_best = False
        if not self.best_metrics or current_score > self._calculate_average_metric(self.best_metrics):
            self.best_metrics = val_metrics
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
            'config': self.config
        }
        
        # 保存最新检查点
        torch.save(checkpoint, self.save_dir / 'last_checkpoint.pth')
        
        # 如果是最佳模型，额外保存
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_checkpoint.pth')
            # 单独保存最佳模型权重（便于部署）
            torch.save(self.student_model.state_dict(), self.save_dir / 'best_model.pth')
            print(f"New best model saved! Average score: {current_score:.4f}")
    
    def _log_train_metrics(self, epoch, losses):
        """记录训练指标到TensorBoard"""
        for dataset_name, loss_dict in losses.items():
            for loss_name, loss_value in loss_dict.items():
                self.writer.add_scalar(f'Train/{dataset_name}/{loss_name}', loss_value, epoch)
    
    def _log_val_metrics(self, epoch, metrics):
        """记录验证指标到TensorBoard"""
        for dataset_name, metric_dict in metrics.items():
            for metric_name, metric_value in metric_dict.items():
                self.writer.add_scalar(f'Val/{dataset_name}/{metric_name}', metric_value, epoch)
    
    def _print_epoch_summary(self, epoch, train_losses, val_metrics, lr):
        """打印epoch总结"""
        print(f"\nEpoch {epoch+1}/{self.config['num_epochs']} - LR: {lr:.6f}")
        print("-" * 80)
        
        # 打印训练损失
        print("Training Losses:")
        for dataset, losses in train_losses.items():
            loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
            print(f"  {dataset}: {loss_str}")
        
        # 打印验证指标
        print("\nValidation Metrics:")
        for dataset, metrics in val_metrics.items():
            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            print(f"  {dataset}: {metric_str}")
        
        print("-" * 80)
    
    def _on_training_end(self):
        """训练结束后的操作"""
        print("\nTraining completed!")
        print(f"Best model saved at epoch {self.best_epoch}")
        
        # 保存训练历史
        with open(self.save_dir / 'train_history.json', 'w') as f:
            json.dump(self.train_history, f, indent=4)
        
        # 生成训练报告
        self._generate_training_report()
        
        # 关闭TensorBoard writer
        self.writer.close()
    
    def _generate_training_report(self):
        """生成训练报告"""
        report = {
            'experiment_name': self.experiment_name,
            'config': self.config,
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics,
            'total_parameters': sum(p.numel() for p in self.student_model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)
        }
        
        with open(self.save_dir / 'training_report.json', 'w') as f:
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

def start_train(dataset_name):
    """主函数：执行完整的训练和评估流程"""
    
    # 配置参数
    config = {
        'num_epochs': 200,  # 增加训练轮数
        'batch_size': 4,    # 增加批次大小
        'learning_rate': 3e-4,  # 降低学习率
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'temperature': 1.5,  # 降低温度
        'feature_weight': 0.2,  # 降低特征蒸馏权重
        'output_weight': 0.2,  # 降低输出蒸馏权重
        'task_weight': 0.6,  # 提高任务损失权重
        'early_stopping_patience': 20,
        'early_stopping_delta': 0.001,
        'num_classes': {
            'classification': 5,  # APTOS2019
            'recognition': 3      # ISIC2017
        }
    }

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

    dataset_config['datasets'] = {dataset_name: dataset_config['datasets'][dataset_name]}
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    student_model = ImprovedMultiTaskStudent(config['num_classes'])
    print(f"Student model parameters: {sum(p.numel() for p in student_model.parameters()) / 1e6:.2f}M")
    
    # 数据加载器

    # 保存配置文件
    with open(f'{dataset_name}_data_config.json', 'w') as f:
        json.dump(dataset_config, f, indent=2)
        
    # 创建数据加载器 (现在返回三层字典结构)
    manager = DataIntegrationManager(f'{dataset_name}_data_config.json')
    dataloaders = manager.create_dataloaders(batch_size=dataset_config['training']['batch_size'])

    # 教师模型
    teacher_models = load_models()
    
    # 开始训练
    trainer = MultiTeacherDistillationTrainer(
        student_model=student_model,
        teacher_models=teacher_models,
        train_loaders=dataloaders['train'],
        val_loaders=dataloaders['val'],
        config=config,
        device=device,
        save_dir=f"./experiments/{dataset_name}"
    )
    
    trainer.train()
    

if __name__ == "__main__":
    datasets = ["BUSI", "kvasir_seg", "APTOS2019", "ISIC2017"]
    start_train(dataset_name=datasets[3]) # 选择对应的数据集 
    
