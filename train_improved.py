"""
改进的YOLOv10训练脚本 - 针对冰箱场景优化
专门解决小物体检测和误检问题
"""

import os
import yaml
from ultralytics import YOLO
import torch
import numpy as np
from pathlib import Path

class ImprovedYOLOTrainer:
    def __init__(self, data_yaml_path, model_size='n'):
        """
        初始化改进的训练器
        
        Args:
            data_yaml_path: 数据集配置文件路径
            model_size: 模型大小 ('n', 's', 'm', 'l')
        """
        self.data_yaml = data_yaml_path
        self.model_size = model_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def create_improved_config(self):
        """创建改进的训练配置"""
        config = {
            # 基础配置
            'model': f'yolov10{self.model_size}.pt',
            'data': self.data_yaml,
            'epochs': 100,  # 增加训练轮数
            'imgsz': 1280,  # 提高分辨率以检测小物体
            'batch': 8,  # 由于分辨率提高，可能需要减小batch size
            
            # 优化器配置
            'optimizer': 'AdamW',
            'lr0': 0.001,  # 降低初始学习率
            'lrf': 0.0001,  # 最终学习率
            'momentum': 0.937,
            'weight_decay': 0.0005,
            
            # 数据增强 - 专门针对小物体和冰箱场景
            'augment': True,
            'degrees': 10.0,  # 适度旋转
            'translate': 0.2,  # 平移
            'scale': 0.9,  # 增加尺度变化
            'shear': 5.0,  # 剪切
            'perspective': 0.0001,  # 轻微透视变换
            'flipud': 0.1,  # 垂直翻转（冰箱物品可能倒放）
            'fliplr': 0.5,  # 水平翻转
            'mosaic': 1.0,  # Mosaic增强
            'mixup': 0.2,  # Mixup增强
            'copy_paste': 0.3,  # Copy-paste增强对小物体有帮助
            
            # 训练策略
            'close_mosaic': 15,  # 最后15轮关闭mosaic
            'warmup_epochs': 5,  # 增加warmup
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # 损失函数权重 - 针对小物体调整
            'box': 10.0,  # 增加边界框损失权重
            'cls': 0.5,  # 分类损失
            'dfl': 1.5,  # DFL损失
            
            # 其他重要参数
            'patience': 30,  # 早停耐心值
            'save': True,
            'cache': True,  # 缓存图像加速训练
            'pretrained': True,
            'multi_scale': True,  # 多尺度训练
            'single_cls': False,
            'rect': False,  # 不使用矩形训练
            'cos_lr': True,  # 使用余弦学习率调度
            'label_smoothing': 0.1,  # 标签平滑
            'dropout': 0.1,  # 添加dropout
            
            # 验证参数
            'val': True,
            'plots': True,
            'save_json': True,
            
            # 设备
            'device': self.device,
            'workers': 8,
            'project': 'fridge_detection_improved',
            'name': f'yolo{self.model_size}_1280_enhanced',
            'exist_ok': False,
            'seed': 42,
            'deterministic': True,
        }
        return config
    
    def train_with_class_weights(self):
        """使用类别权重训练"""
        # 加载模型
        model = YOLO(f'yolov10{self.model_size}.pt')
        
        # 获取改进的配置
        config = self.create_improved_config()
        
        # 计算类别权重（根据您的数据集统计）
        class_weights = self.calculate_class_weights()
        if class_weights is not None:
            config['cls_weights'] = class_weights
        
        # 开始训练
        results = model.train(**config)
        
        return results
    
    def calculate_class_weights(self):
        """
        计算类别权重以平衡数据集
        对于实例较少的类别给予更高权重
        """
        # 根据您提供的数据统计
        class_instances = {
            'Cooked_Rice': 151,
            'Egg': 429,
            'Tofu': 946,
            'Onion': 782,
            'Carrot': 1104,
            'Tomato': 2372,
            'Capsicum': 504,
            'Chicken_Breast': 434,
            'Prawn': 636,
            'Butter': 372,
            'Potato': 1179,
            'Beef_Steak': 250
        }
        
        # 计算权重（反比例）
        total_instances = sum(class_instances.values())
        num_classes = len(class_instances)
        
        weights = {}
        for class_name, count in class_instances.items():
            # 使用平方根来避免权重差异过大
            weight = np.sqrt(total_instances / (num_classes * count))
            weights[class_name] = weight
        
        # 归一化权重
        max_weight = max(weights.values())
        normalized_weights = {k: v/max_weight * 2.0 for k, v in weights.items()}
        
        print("类别权重:")
        for class_name, weight in normalized_weights.items():
            print(f"  {class_name}: {weight:.3f}")
        
        # 转换为列表（按类别ID顺序）
        weight_list = list(normalized_weights.values())
        return weight_list
    
    def train_multi_scale(self):
        """多尺度训练策略"""
        scales = [640, 960, 1280]
        models = []
        
        for scale in scales:
            print(f"\n训练尺度: {scale}x{scale}")
            config = self.create_improved_config()
            config['imgsz'] = scale
            config['name'] = f'yolo{self.model_size}_{scale}'
            
            model = YOLO(f'yolov10{self.model_size}.pt')
            results = model.train(**config)
            models.append(model)
        
        return models
    
    def fine_tune_for_small_objects(self, base_model_path):
        """
        专门针对小物体进行微调
        """
        # 加载基础模型
        model = YOLO(base_model_path)
        
        config = {
            'data': self.data_yaml,
            'epochs': 30,  # 微调轮数
            'imgsz': 1280,
            'batch': 4,
            
            # 专门针对小物体的配置
            'lr0': 0.0001,  # 更小的学习率
            'box': 15.0,  # 更高的box损失权重
            'anchor_t': 2.0,  # 降低anchor阈值，接受更多小anchor
            
            # 强化数据增强
            'copy_paste': 0.5,  # 增加copy-paste
            'scale': 1.2,  # 更大的尺度变化
            'mosaic': 1.0,
            
            'project': 'fridge_detection_small_objects',
            'name': f'yolo{self.model_size}_small_finetune',
        }
        
        results = model.train(**config)
        return results

class DataAugmentationPipeline:
    """数据增强管道"""
    
    @staticmethod
    def augment_for_fridge_scene(image, labels):
        """
        专门针对冰箱场景的数据增强
        
        Args:
            image: 输入图像
            labels: YOLO格式标签
        """
        import albumentations as A
        import cv2
        
        # 创建增强管道
        transform = A.Compose([
            # 光照变化（模拟冰箱内不同光照）
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5
            ),
            
            # 模糊（模拟冰箱玻璃门或雾气）
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1),
                A.MotionBlur(blur_limit=(3, 7), p=1),
                A.GlassBlur(sigma=0.7, max_delta=2, iterations=2, p=1),
            ], p=0.3),
            
            # 噪声（模拟低光照噪声）
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
            ], p=0.2),
            
            # 阴影（模拟物品遮挡产生的阴影）
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=3,
                shadow_dimension=5,
                p=0.3
            ),
            
            # 网格失真（模拟冰箱架子造成的视觉效果）
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
            
            # 色彩变化
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.3
            ),
            
            # CLAHE（增强对比度，有助于小物体检测）
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        # 应用增强
        class_labels = [label[0] for label in labels]
        bboxes = [label[1:5] for label in labels]
        
        transformed = transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        
        return transformed['image'], transformed['bboxes'], transformed['class_labels']

def main():
    """主训练流程"""
    
    # 配置路径
    data_yaml = '/content/data.yaml'  # 您的数据集配置
    
    # 创建训练器
    trainer = ImprovedYOLOTrainer(data_yaml, model_size='s')  # 使用更大的模型
    
    # 方案1: 标准改进训练
    print("开始改进训练...")
    results = trainer.train_with_class_weights()
    
    # 方案2: 多尺度训练（可选）
    # print("开始多尺度训练...")
    # models = trainer.train_multi_scale()
    
    # 方案3: 小物体微调（如果已有基础模型）
    # print("针对小物体微调...")
    # results = trainer.fine_tune_for_small_objects('path/to/base/model.pt')
    
    print("训练完成！")
    
    # 保存最佳模型路径
    best_model_path = results.best
    print(f"最佳模型保存在: {best_model_path}")
    
    return best_model_path

if __name__ == "__main__":
    main()