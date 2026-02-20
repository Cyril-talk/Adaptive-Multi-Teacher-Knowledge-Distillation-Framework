

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
import yaml
from datetime import datetime

# 导入 zl.py 中的所有组件
from zl import (
    Config,
    setup_logger,
    StudentModel,
    TeacherGate,
    ReturnDecoder,
    LossLevelGating,
    CRDLoss,
    ReconstructionLoss,
    FeatureDistillationLoss,
    OutputDistillationLoss,
    MultiMedDistillSystem,
    train_one_epoch,
    evaluate,
    Visualizer,
    create_dataloaders,
    create_teacher_models,
    MultiTaskDataset,
    BUSIDataset,
    NIHChestXrayDataset
)


# ==================== 扩展配置类 ====================
class TrainingConfig(Config):
    """扩展的训练配置"""

    def __init__(self, config_path=None):
        super().__init__()

        # 如果提供了配置文件，加载覆盖
        if config_path and Path(config_path).exists():
            self.load_from_yaml(config_path)

        # 训练特定配置
        self.use_amp = True  # 混合精度训练
        self.gradient_accumulation_steps = 1
        self.early_stopping_patience = 10
        self.save_frequency = 5  # 每5个epoch保存一次

        # 分布式训练
        self.distributed = False
        self.local_rank = -1

        # 实验追踪
        self.experiment_name = f"multimed_distill_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.use_wandb = False  # 是否使用 Weights & Biases

    def load_from_yaml(self, config_path):
        """从 YAML 文件加载配置"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save_to_yaml(self, save_path):
        """保存配置到 YAML 文件"""
        config_dict = {k: v for k, v in self.__dict__.items()
                       if not k.startswith('_') and not callable(v)}

        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# ==================== 数据集注册器 ====================
class DatasetRegistry:
    """数据集注册和管理"""

    DATASETS = {
        'busi': BUSIDataset,
        'nih': NIHChestXrayDataset,
        # 可以继续添加其他数据集
    }

    @classmethod
    def create_dataset(cls, name, root_dir, split, transform):
        """创建指定数据集"""
        if name not in cls.DATASETS:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(cls.DATASETS.keys())}")

        dataset_class = cls.DATASETS[name]
        return dataset_class(root_dir=root_dir, split=split, transform=transform)

    @classmethod
    def register_dataset(cls, name, dataset_class):
        """注册新数据集"""
        cls.DATASETS[name] = dataset_class


# ==================== 增强的数据加载器创建 ====================
def create_enhanced_dataloaders(config, logger):
    """
    创建增强的数据加载器（支持任务感知采样）
    """
    from torch.utils.data import Sampler
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    # 训练数据增强（论文标准）
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # 验证数据增强
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # 创建数据集
    train_datasets = {}
    val_datasets = {}

    # 获取所有任务配置
    all_tasks = {
        **config.segmentation_tasks,
        **config.classification_tasks,
        **config.recognition_tasks
    }

    for task_name in all_tasks.keys():
        try:
            # 训练集
            train_datasets[task_name] = DatasetRegistry.create_dataset(
                name=task_name,
                root_dir=Path(config.data_root) / task_name.upper(),
                split='train',
                transform=train_transform
            )

            # 验证集
            val_datasets[task_name] = DatasetRegistry.create_dataset(
                name=task_name,
                root_dir=Path(config.data_root) / task_name.upper(),
                split='val',
                transform=val_transform
            )

            logger.info(f"✓ Loaded {task_name}: "
                        f"train={len(train_datasets[task_name])}, "
                        f"val={len(val_datasets[task_name])}")

        except Exception as e:
            logger.warning(f"✗ Failed to load {task_name}: {e}")

    # 创建多任务数据集
    train_dataset = MultiTaskDataset(train_datasets)
    val_dataset = MultiTaskDataset(val_datasets)

    # 任务感知批采样器
    class TaskBatchSampler(Sampler):
        """确保每个batch来自同一任务"""

        def __init__(self, dataset, batch_size):
            self.dataset = dataset
            self.batch_size = batch_size

            # 按任务分组索引
            self.task_indices = {i: [] for i in range(len(dataset.task_names))}
            for idx in range(len(dataset)):
                task_id = np.searchsorted(dataset.cumulative_lengths[1:], idx, side='right')
                self.task_indices[task_id].append(idx)

        def __iter__(self):
            # 随机打乱每个任务的索引
            for task_id in self.task_indices.keys():
                indices = self.task_indices[task_id].copy()
                np.random.shuffle(indices)

                # 生成批次
                for i in range(0, len(indices), self.batch_size):
                    batch = indices[i:i + self.batch_size]
                    if len(batch) == self.batch_size:  # 只返回完整批次
                        yield batch

        def __len__(self):
            return sum(len(indices) // self.batch_size
                       for indices in self.task_indices.values())

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=TaskBatchSampler(train_dataset, config.batch_size),
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=TaskBatchSampler(val_dataset, config.batch_size),
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader


# ==================== 学习率调度器（带Warmup） ====================
class WarmupCosineScheduler:
    """带 Warmup 的余弦退火调度器"""

    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 initial_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Warmup 阶段：线性增长
            lr = self.initial_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine Annealing 阶段
            progress = (self.current_epoch - self.warmup_epochs) / (
                    self.total_epochs - self.warmup_epochs
            )
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (
                    1 + np.cos(np.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_epoch += 1
        return lr

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


# ==================== 混合精度训练引擎 ====================
def train_one_epoch_amp(model, dataloader, optimizer, scheduler, scaler,
                        device, epoch, logger, config):
    """
    带混合精度的训练循环
    """
    from torch.cuda.amp import autocast
    from tqdm import tqdm

    model.train()

    # 初始化指标
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
        # 数据转移
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        task_ids = batch['task_id'].to(device, non_blocking=True)
        task_name = batch['task_name'][0]

        # 混合精度前向传播
        with autocast(enabled=config.use_amp):
            total_loss, loss_dict, loss_weights, predictions = model(
                images, task_ids, task_name, labels
            )

            # 梯度累积
            total_loss = total_loss / config.gradient_accumulation_steps

        # 混合精度反向传播
        scaler.scale(total_loss).backward()

        # 梯度累积步骤
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 优化器步骤
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # 记录指标
        metrics['total_loss'].append(total_loss.item() * config.gradient_accumulation_steps)
        for k in ['task', 'feat', 'out', 'rec', 'crd']:
            metrics[f'{k}_loss'].append(loss_dict[k].item())
        metrics['loss_weights'].append(loss_weights.mean(0).detach().cpu().numpy())

        # 更新进度条
        pbar.set_postfix({
            'Loss': f"{total_loss.item() * config.gradient_accumulation_steps:.4f}",
            'LR': f"{optimizer.param_groups[0]['lr']:.6f}"
        })

    # 学习率调度
    current_lr = scheduler.step()

    # 更新输出蒸馏温度
    model.out_loss.update_temperature(epoch, config.num_epochs)

    # 计算平均指标
    avg_metrics = {k: np.mean(v) for k, v in metrics.items() if k != 'loss_weights'}
    avg_metrics['loss_weights'] = np.mean(metrics['loss_weights'], axis=0)
    avg_metrics['learning_rate'] = current_lr

    return avg_metrics


# ==================== 早停机制 ====================
class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=10, min_delta=0.0001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


# ==================== 主训练函数 ====================
def main(args):
    """主训练流程"""

    # 1. 初始化配置
    config = TrainingConfig(args.config)

    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # 2. 设置日志
    log_file = Path(config.log_dir) / f'{config.experiment_name}.log'
    logger = setup_logger(log_file)

    logger.info("=" * 80)
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info("=" * 80)
    logger.info(f"Device: {config.device}")
    logger.info(f"Mixed Precision: {config.use_amp}")
    logger.info(f"Gradient Accumulation: {config.gradient_accumulation_steps}")

    # 保存配置
    config.save_to_yaml(Path(config.log_dir) / f'{config.experiment_name}_config.yaml')

    # 3. 创建数据加载器
    logger.info("\n" + "=" * 80)
    logger.info("Creating Dataloaders...")
    logger.info("=" * 80)

    train_loader, val_loader = create_enhanced_dataloaders(config, logger)

    logger.info(f"✓ Train batches: {len(train_loader)}")
    logger.info(f"✓ Val batches: {len(val_loader)}")

    # 4. 创建教师模型
    logger.info("\n" + "=" * 80)
    logger.info("Creating Teacher Models...")
    logger.info("=" * 80)

    # 合并所有任务配置
    all_tasks_config = {
        **{k: v['classes'] for k, v in config.segmentation_tasks.items()},
        **{k: v['classes'] for k, v in config.classification_tasks.items()},
        **{k: v['classes'] for k, v in config.recognition_tasks.items()}
    }

    # 临时修改 config.num_classes_dict 以兼容 create_teacher_models
    config.num_classes_dict = all_tasks_config

    teachers = create_teacher_models(config)
    for name, teacher in teachers.items():
        teacher = teacher.to(config.device)
        logger.info(f"✓ Teacher '{name}' loaded")

    # 5. 创建蒸馏系统
    logger.info("\n" + "=" * 80)
    logger.info("Creating Distillation System...")
    logger.info("=" * 80)

    model = MultiMedDistillSystem(config, teachers).to(config.device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.student.parameters())
    trainable_params = sum(p.numel() for p in model.student.parameters() if p.requires_grad)

    logger.info(f"✓ Student Model Parameters:")
    logger.info(f"  - Total: {total_params / 1e6:.2f}M")
    logger.info(f"  - Trainable: {trainable_params / 1e6:.2f}M")

    # 6. 优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        total_epochs=config.num_epochs,
        initial_lr=config.learning_rate
    )

    # 混合精度 Scaler
    from torch.cuda.amp import GradScaler
    scaler = GradScaler(enabled=config.use_amp)

    # 早停
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        mode='max'
    )

    # 7. 训练循环
    logger.info("\n" + "=" * 80)
    logger.info("Starting Training...")
    logger.info("=" * 80)

    best_accuracy = 0.0
    train_history = {
        'total_loss': [], 'task_loss': [], 'feat_loss': [],
        'out_loss': [], 'rec_loss': [], 'crd_loss': [],
        'learning_rate': []
    }
    val_history = {'accuracy': []}
    loss_weights_history = []

    for epoch in range(1, config.num_epochs + 1):
        # 训练
        train_metrics = train_one_epoch_amp(
            model, train_loader, optimizer, scheduler, scaler,
            config.device, epoch, logger, config
        )

        # 记录训练指标
        for k in train_history.keys():
            if k in train_metrics:
                train_history[k].append(train_metrics[k])
        loss_weights_history.append(train_metrics['loss_weights'])

        # 评估
        if epoch % config.save_frequency == 0 or epoch == config.num_epochs:
            val_results = evaluate(model, val_loader, config.device, logger)
            val_history['accuracy'].append(val_results['accuracy'])

            # 保存最佳模型
            if val_results['accuracy'] > best_accuracy:
                best_accuracy = val_results['accuracy']

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.__dict__,
                    'scaler_state_dict': scaler.state_dict(),
                    'accuracy': best_accuracy,
                    'config': config.__dict__
                }

                torch.save(
                    checkpoint,
                    Path(config.checkpoint_dir) / f'{config.experiment_name}_best.pth'
                )

                logger.info(f"✓ New best model saved! Accuracy: {best_accuracy:.4f}")

            # 可视化
            if epoch % 10 == 0:
                vis = Visualizer()

                # t-SNE
                vis.plot_tsne(
                    val_results['features'],
                    val_results['labels'],
                    Path(config.vis_dir) / f'{config.experiment_name}_tsne_epoch_{epoch}.png'
                )

                # 损失权重演化
                vis.plot_loss_weights_evolution(
                    loss_weights_history,
                    Path(config.vis_dir) / f'{config.experiment_name}_weights_epoch_{epoch}.png'
                )

            # 早停检查
            if early_stopping(val_results['accuracy']):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # 日志输出
        logger.info(
            f"Epoch {epoch}/{config.num_epochs} | "
            f"Train Loss: {train_metrics['total_loss']:.4f} | "
            f"Val Acc: {val_history['accuracy'][-1] if val_history['accuracy'] else 'N/A':.4f} | "
            f"LR: {train_metrics['learning_rate']:.6f}"
        )

    # 8. 保存最终模型
    final_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'best_accuracy': best_accuracy,
        'train_history': train_history,
        'val_history': val_history
    }

    torch.save(
        final_checkpoint,
        Path(config.checkpoint_dir) / f'{config.experiment_name}_final.pth'
    )

    # 9. 最终可视化
    vis = Visualizer()
    vis.plot_loss_curves(
        train_history,
        val_history,
        Path(config.vis_dir) / f'{config.experiment_name}_curves.png'
    )

    # 10. 性能基准测试
    logger.info("\n" + "=" * 80)
    logger.info("Performance Benchmarking")
    logger.info("=" * 80)

    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(config.device)

    # 延迟测试
    import time
    warmup, test_runs = 10, 100

    for _ in range(warmup):
        with torch.no_grad():
            _ = model.student(dummy_input, list(all_tasks_config.keys())[0])

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(test_runs):
        with torch.no_grad():
            _ = model.student(dummy_input, list(all_tasks_config.keys())[0])

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    avg_time = (time.time() - start_time) / test_runs * 1000
    logger.info(f"✓ Average Inference Time: {avg_time:.2f} ms")

    # FLOPs 计算
    try:
        from thop import profile
        flops, params = profile(
            model.student,
            inputs=(dummy_input, list(all_tasks_config.keys())[0])
        )
        logger.info(f"✓ FLOPs: {flops / 1e9:.2f} G")
        logger.info(f"✓ Parameters: {params / 1e6:.2f} M")
    except ImportError:
        logger.warning("⚠ thop not installed. Skipping FLOPs calculation.")

    # 11. 总结
    logger.info("\n" + "=" * 80)
    logger.info("Training Completed!")
    logger.info("=" * 80)
    logger.info(f"✓ Best Validation Accuracy: {best_accuracy:.4f}")
    logger.info(f"✓ Total Epochs: {epoch}")
    logger.info(f"✓ Experiment: {config.experiment_name}")
    logger.info("=" * 80)


# ==================== 命令行接口 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiMedDistill Training")

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML file'
    )

    parser.add_argument(
        '--data_root',
        type=str,
        default='./data',
        help='Root directory of datasets'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training'
    )

    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Initial learning rate'
    )

    parser.add_argument(
        '--use_amp',
        action='store_true',
        help='Use automatic mixed precision'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    args = parser.parse_args()

    # 运行训练
    main(args)
