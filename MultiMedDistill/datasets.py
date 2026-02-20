# ===== datasets.py 修复版本 =====
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from typing import Dict, List, Tuple, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2

class UnifiedMedicalDataset(Dataset):
    """
    统一的医学图像数据集类，支持多任务学习
    """
    def __init__(self, 
                 data_configs: Dict,
                 transform=None,
                 mode='train',
                 target_size=(512, 512)):
        """
        Args:
            data_configs: 包含各数据集配置的字典
            transform: 数据增强变换
            mode: 'train', 'val', 'test'
            target_size: 统一的图像尺寸
        """
        self.data_configs = data_configs
        self.transform = transform
        self.mode = mode
        self.target_size = target_size
        
        # 整合所有数据集的样本
        self.samples = self._integrate_datasets()
        
    def _integrate_datasets(self) -> List[Dict]:
        """整合所有数据集的样本信息"""
        all_samples = []
        
        for dataset_name, config in self.data_configs.items():
            dataset_samples = self._load_dataset_samples(dataset_name, config)
            all_samples.extend(dataset_samples)
            
        return all_samples
    
    def _load_dataset_samples(self, dataset_name: str, config: Dict) -> List[Dict]:
        """加载单个数据集的样本"""
        samples = []
        
        if dataset_name == 'BUSI':
            samples = self._load_busi_samples(config)
        elif dataset_name == 'ISIC2017':
            samples = self._load_isic_samples(config)
        elif dataset_name == 'APTOS2019':
            samples = self._load_aptos_samples(config)
        elif dataset_name == "kvasir_seg":
            samples = self._load_kvasir_seg_samples(config)
            
        return samples
    
    def _load_busi_samples(self, config: Dict) -> List[Dict]:
        """加载BUSI数据集 - 分割任务"""
        samples = []
        data_dir = config['data_dir']
        labels = config['labels']
        
        for split_dir in ['train', 'val', 'test']:
            split_path = os.path.join(data_dir, split_dir)
            if not os.path.exists(split_path):
                continue
                
            # 遍历所有类别目录
            for label_idx, label_name in enumerate(labels):
                class_dir = os.path.join(split_path, label_name)
                if not os.path.exists(class_dir):
                    continue
                    
                # 遍历该类别下的所有文件
                for file in os.listdir(class_dir):
                    if file.endswith('.png') and not file.endswith('_mask.png'):
                        img_path = os.path.join(class_dir, file)
                        base_name = os.path.splitext(file)[0]
                        mask_path = os.path.join(class_dir, f"{base_name}_mask.png")
                        
                        # 确保掩码文件存在
                        if not os.path.exists(mask_path):
                            continue
                        
                        samples.append({
                            'dataset': 'BUSI',
                            'task_type': 'segmentation',
                            'image_path': img_path,
                            'mask_path': mask_path,
                            'label': label_idx,  # 使用类别索引
                            'bbox': None,
                        })
        return samples
    
    def _load_isic_samples(self, config: Dict) -> List[Dict]:
        """加载ISIC2017数据集 - 识别任务"""
        samples = []
        data_dir = config['data_dir']
        
        # 遍历ISIC2017数据集文件
        for split_dir in ['Training', 'Validation', 'Test']:
            if "Training" in split_dir:
                split_dir = "ISIC-2017_Training_Data"
            elif "Validation" in split_dir:
                split_dir = "ISIC-2017_Validation_Data"
            else:
                split_dir = "ISIC-2017_Test_v2_Data"
            file_dir = os.path.join(data_dir, split_dir)
            if not os.path.exists(file_dir):
                continue
            
            for _, file in enumerate(os.listdir(file_dir)):
                if "_superpixels" in file or ".csv" in file:
                    continue
                
                img_path = os.path.join(file_dir, file)
                mask_path = os.path.join(file_dir.replace("_Data", "_Part1_GroundTruth"), file.replace(".jpg", "_segmentation.png"))
                annotation_path = os.path.join(file_dir.replace("_Data", "_Annotations"), file.replace(".jpg", "_annotation.json"))
                if not (os.path.exists(mask_path) and os.path.exists(annotation_path)):
                    continue 

                try:
                    with open(annotation_path, 'r', encoding="utf-8") as f:
                            annotations = json.load(f)

                    # 提取边界框和类别
                    bboxes = []
                    labels = []
                    for ann in annotations:
                        bbox = ann['ann']  # [x, y, width, height]
                        category_id = ann['category_id']
                        bboxes.append(bbox)
                        labels.append(category_id)

                    # 只保留有单个标签的样本
                    if len(labels) == 1:
                        samples.append({
                            'dataset': 'ISIC2017',
                            'task_type': 'recognition', 
                            'image_path': img_path,
                            'mask_path': mask_path,
                            'label': labels[0],  # 使用单个标签，而不是列表
                            'bbox': bboxes,
                        })
                    elif len(labels) > 1:
                        # 如果有多个标签，取第一个并打印警告
                        samples.append({
                            'dataset': 'ISIC2017',
                            'task_type': 'recognition', 
                            'image_path': img_path,
                            'mask_path': mask_path,
                            'label': labels[0],  # 使用第一个标签
                            'bbox': bboxes,
                        })
                except Exception as e:
                    print(f"Warning: Error loading annotation for {file}: {e}")
                    continue
        
        # 读取标签文件
        label_file = os.path.join(data_dir, f'{self.mode}_labels.csv')
        if os.path.exists(label_file):
            try:
                labels_df = pd.read_csv(label_file)
                
                for _, row in labels_df.iterrows():
                    img_name = row['image_id']
                    img_path = os.path.join(data_dir, 'images', f'{img_name}.jpg')
                    
                    if os.path.exists(img_path):
                        # ISIC2017通常是二分类或多分类
                        label = row['melanoma'] if 'melanoma' in row else row['label']
                        
                        # 确保label是简单的值，不是张量
                        if isinstance(label, torch.Tensor):
                            label = label.item() if label.numel() == 1 else int(label.flatten()[0].item())
                        
                        samples.append({
                            'dataset': 'ISIC2017',
                            'task_type': 'recognition',
                            'image_path': img_path,
                            'mask_path': None,
                            'label': int(label),
                            'bbox': None,
                        })
            except Exception as e:
                print(f"Warning: Error reading label file {label_file}: {e}")
        
        return samples
    
    def _load_aptos_samples(self, config: Dict) -> List[Dict]:
        """加载APTOS2019数据集 - 分类任务"""
        samples = []
        data_dir = config['data_dir']
        
        for split_dir in ['train', 'val', 'test']:
            split_path = os.path.join(data_dir, split_dir)
            if not os.path.exists(split_path):
                continue

            labels = config['labels']
            for label_index, label in enumerate(labels):
                file_dir = os.path.join(split_path, label)
                if not os.path.exists(file_dir):
                    continue
                    
                for file in os.listdir(file_dir):
                    if file.endswith('.png'):
                        img_path = os.path.join(file_dir, file)
                        if os.path.exists(img_path):
                            samples.append({
                                'dataset': 'APTOS2019',
                                'task_type': 'classification',
                                'image_path': img_path,
                                'mask_path': None,
                                'label': label_index,  
                                'bbox': None,   
                            })
        
        return samples
    
    def _load_kvasir_seg_samples(self, config: Dict) -> List[Dict]:
        """加载Kvasir_seg数据集"""
        samples = []
        data_dir = config['data_dir']
        
        images_path = os.path.join(data_dir, 'images')
        masks_path = os.path.join(data_dir, 'masks')
        
        if not os.path.exists(images_path) or not os.path.exists(masks_path):
            return samples
            
        # 读取边界框信息
        boxes_dict = {}
        bbox_file = os.path.join(data_dir, 'kavsir_bboxes.json')
        if os.path.exists(bbox_file):
            try:
                with open(bbox_file, 'r', encoding='UTF-8') as file:
                    boxes_dict = json.loads(file.read())
            except:
                pass
        
        for file in os.listdir(images_path):
            if not file.endswith(('.jpg', '.png')):
                continue
                
            img_path = os.path.join(images_path, file)
            mask_path = os.path.join(masks_path, file)
            
            if not (os.path.exists(img_path) and os.path.exists(mask_path)):
                continue
            
            # 提取边界框信息
            bboxes = []
            labels = []
            file_key = file.split('.')[0]
            temp_dict = boxes_dict.get(file_key, {})
            
            if temp_dict and 'bbox' in temp_dict:
                for b_info in temp_dict['bbox']:
                    if all(key in b_info for key in ['xmin', 'ymin', 'xmax', 'ymax']):
                        bbox = [b_info['xmin'], b_info['ymin'], 
                               b_info['xmax'] - b_info['xmin'], 
                               b_info['ymax'] - b_info['ymin']]
                        category_id = b_info.get('label', 0)
                        bboxes.append(bbox)
                        labels.append(category_id)

            samples.append({
                'dataset': 'kvasir_seg',
                'task_type': 'segmentation', 
                'image_path': img_path,
                'mask_path': mask_path,
                'label': labels if labels else [0],
                'bbox': bboxes if bboxes else None,
            })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # 加载图像
            image = Image.open(sample['image_path']).convert('RGB')
            image = np.array(image)
            
            # 确保图像是正确的形状
            if len(image.shape) != 3 or image.shape[2] != 3:
                # 如果不是RGB图像，创建一个RGB图像
                if len(image.shape) == 2:
                    image = np.stack([image, image, image], axis=2)
                elif len(image.shape) == 3 and image.shape[2] == 1:
                    image = np.concatenate([image, image, image], axis=2)
                elif len(image.shape) == 3 and image.shape[2] > 3:
                    image = image[:, :, :3]
            
        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {e}")
            # 返回默认图像
            image = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
        
        # 根据任务类型准备标签
        targets = self._prepare_targets(sample)
        
        # 统一尺寸
        original_h, original_w = image.shape[:2]
        if original_h != self.target_size[0] or original_w != self.target_size[1]:
            image = cv2.resize(image, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)
        
        # 应用数据增强
        if self.transform:
            if sample['task_type'] == 'segmentation' and 'mask' in targets and targets['mask'] is not None:
                try:
                    mask_np = targets['mask'].numpy() if isinstance(targets['mask'], torch.Tensor) else targets['mask']
                    
                    # 确保掩码和图像尺寸一致
                    if mask_np.shape[:2] != image.shape[:2]:
                        mask_np = cv2.resize(mask_np, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    # 应用增强（禁用形状检查）
                    transformed = self.transform(image=image, mask=mask_np)
                    image = transformed['image']
                    targets['mask'] = transformed['mask'].long()
                except Exception as e:
                    print(f"Transform error for {sample['image_path']}: {e}")
                    # 如果增强失败，使用原始图像和掩码
                    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            else:
                try:
                    transformed = self.transform(image=image)
                    image = transformed['image']
                except Exception as e:
                    print(f"Transform error for {sample['image_path']}: {e}")
                    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            # 如果没有变换，手动转换
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, targets
    
    def _prepare_targets(self, sample: Dict) -> Dict:
        """准备不同任务的目标标签"""
        targets = {
            'task_type': sample['task_type'],
            'dataset': sample['dataset']
        }
        
        
        if sample['task_type'] == 'segmentation':
            if sample['mask_path'] and os.path.exists(sample['mask_path']):
                try:
                    mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
                    mask = mask[:, :, None] if mask.ndim == 2 else mask
                    if mask is not None:
                        # 标准化分割标签为二值 (0, 1)
                        mask = (mask > 0).astype(np.uint8)
                        
                        # 统一尺寸
                        if mask.shape[:2] != self.target_size:
                            mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
                        
                        targets['mask'] = torch.from_numpy(mask).long()
                    else:
                        targets['mask'] = torch.zeros(self.target_size, dtype=torch.long)
                except Exception as e:
                    print(f"Error loading mask {sample['mask_path']}: {e}")
                    targets['mask'] = torch.zeros(self.target_size, dtype=torch.long)
            else:
                targets['mask'] = torch.zeros(self.target_size, dtype=torch.long)
                
        elif sample['task_type'] == 'classification':
            targets['class_label'] = torch.tensor(sample['label'], dtype=torch.long)
            
        elif sample['task_type'] == 'recognition':
            # 识别任务（检测任务）
            if sample['bbox'] and sample['label']:
                targets['boxes'] = torch.tensor(sample['bbox'], dtype=torch.float32)
                # 确保labels是1D张量
                labels = sample['label']
                if isinstance(labels, list):
                    targets['labels'] = torch.tensor(labels, dtype=torch.long)
                else:
                    targets['labels'] = torch.tensor([labels], dtype=torch.long)
            else:
                targets['boxes'] = torch.empty((0, 4), dtype=torch.float32)
                targets['labels'] = torch.empty((0,), dtype=torch.long)
            
            # 主要类别标签 - 确保是标量
            if sample['label'] is not None:
                label = sample['label']
                try:
                    if isinstance(label, torch.Tensor):
                        # 如果是张量，检查维度
                        if label.numel() == 1:
                            # 标量张量
                            targets['class_label'] = torch.tensor(label.item(), dtype=torch.long)
                        elif label.dim() == 1:
                            # 1D张量，取第一个元素
                            targets['class_label'] = torch.tensor(label[0].item(), dtype=torch.long)
                        else:
                            # 多维张量，展平后取第一个元素
                            targets['class_label'] = torch.tensor(label.flatten()[0].item(), dtype=torch.long)
                    elif isinstance(label, list):
                        # 如果是列表，取第一个元素
                        if len(label) > 0:
                            first_label = label[0]
                            if isinstance(first_label, torch.Tensor):
                                # 列表中的元素也是张量
                                targets['class_label'] = torch.tensor(first_label.item(), dtype=torch.long)
                            else:
                                targets['class_label'] = torch.tensor(int(first_label), dtype=torch.long)
                        else:
                            targets['class_label'] = torch.tensor(0, dtype=torch.long)
                    elif isinstance(label, (int, np.integer)):
                        targets['class_label'] = torch.tensor(int(label), dtype=torch.long)
                    else:
                        # 其他情况，尝试转换为标量
                        targets['class_label'] = torch.tensor(int(label), dtype=torch.long)
                except Exception as e:
                    print(f"Warning: Error processing label for recognition task: {e}, using default label 0")
                    targets['class_label'] = torch.tensor(0, dtype=torch.long)
            else:
                targets['class_label'] = torch.tensor(0, dtype=torch.long)
        
        return targets


class DataIntegrationManager:
    """数据整合管理器"""
    
    def __init__(self, config_file: str):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
    
    def create_transforms(self) -> Dict:
        """创建数据增强变换"""
        train_transform = A.Compose([
                    A.Resize(512, 512),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Affine(scale=(0.85, 1.15), translate_percent=0.1, rotate=(-25, 25), p=0.5),  # 替代ShiftScaleRotate
                    A.RandomBrightnessContrast(p=0.3),
                    A.GaussianBlur(p=0.2),
                    A.CoarseDropout(num_holes=8, max_h_size=32, max_w_size=32, p=0.3),  # 修复参数名
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ], is_check_shapes=False)  # 关键：禁用形状检查
        
        val_transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], is_check_shapes=False)  # 关键：禁用形状检查
        
        return {'train': train_transform, 'val': val_transform, 'test': val_transform}
    
    def create_dataloaders(self, batch_size: int = 8) -> Dict:
        """创建数据加载器"""
        transforms = self.create_transforms()
        dataloaders = {'train': {}, 'val': {}, 'test': {}}
        
        # 为每个数据集创建单独的DataLoader
        for dataset_name, config in self.config['datasets'].items():
            for mode in ['train', 'val', 'test']:
                # 创建只包含当前数据集的子数据集
                dataset_config = {
                    dataset_name: config
                }
                
                dataset = UnifiedMedicalDataset(
                    data_configs=dataset_config,
                    transform=transforms[mode],
                    mode=mode,
                    target_size=(512, 512)
                )
                
                if len(dataset) > 0:  # 只有当数据集不为空时才创建DataLoader
                    # 创建DataLoader
                    dataloaders[mode][dataset_name] = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=(mode == 'train'),
                        num_workers=8,  # 减少worker数量
                        collate_fn=self.custom_collate_fn,
                        pin_memory=True,
                        drop_last=False
                    )
        
        return dataloaders
    
    def custom_collate_fn(self, batch):
        """自定义批次整理函数，处理不同任务的数据"""
        images = []
        targets = []  # 存储每个样本的targets字典
        
        for item in batch:
            images.append(item[0])
            targets.append(item[1])
        
        return (
            torch.stack(images),
            targets  # 列表，包含每个样本的targets字典
        )

    def analyze_dataset_statistics(self):
        """分析数据集统计信息"""
        stats = {}
        
        for mode in ['train', 'val', 'test']:
            try:
                dataset = UnifiedMedicalDataset(
                    data_configs=self.config['datasets'],
                    transform=None,
                    mode=mode,
                    target_size=(512, 512)
                )
                
                task_counts = {}
                dataset_counts = {}
                
                for sample in dataset.samples:
                    task_type = sample['task_type']
                    dataset_name = sample['dataset']
                    
                    task_counts[task_type] = task_counts.get(task_type, 0) + 1
                    dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1
                
                stats[mode] = {
                    'total_samples': len(dataset.samples),
                    'task_distribution': task_counts,
                    'dataset_distribution': dataset_counts
                }
            except Exception as e:
                print(f"Error analyzing {mode} dataset: {e}")
                stats[mode] = {
                    'total_samples': 0,
                    'task_distribution': {},
                    'dataset_distribution': {}
                }
        
        return stats


# 使用示例
if __name__ == "__main__":
    # datasets 配置文件内容
    config = {
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
        "training": {
            "batch_size": 8,
            "num_epochs": 100,
            "learning_rate": 0.001
        },
    }

    # 保存配置文件
    with open('data_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # 创建数据管理器
    manager = DataIntegrationManager('data_config.json')
    
    # 创建数据加载器
    dataloaders = manager.create_dataloaders(batch_size=8)
    
    # 分析数据集统计
    stats = manager.analyze_dataset_statistics()
    print("Dataset Statistics:")
    for mode, mode_stats in stats.items():
        print(f"\n{mode.upper()} Set:")
        print(f"  Total samples: {mode_stats['total_samples']}")
        print(f"  Task distribution: {mode_stats['task_distribution']}")
        print(f"  Dataset distribution: {mode_stats['dataset_distribution']}")
