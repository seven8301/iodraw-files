# 冰箱食材检测模型改进行动计划

## 📊 当前问题分析

基于您的反馈：
1. **小物体检测不到** - 特别是Prawn(虾)、Egg(鸡蛋)等小食材
2. **误检问题** - 把非食物物品误认为食材

您的训练数据显示：
- 训练效果最好的：Egg (mAP 0.949), Cooked_Rice (0.933), Chicken_Breast (0.909)
- 效果较差的：Carrot (0.516), Prawn (0.529) - 这两个确实容易漏检
- 整体mAP50: 0.83 (理论上不错，但实际应用有差距)

## 🎯 立即可以做的改进（今天就能完成）

### 1. 调整推理参数（5分钟）
```python
# 在您的推理代码中调整这些参数
model.predict(
    source=image,
    conf=0.25,  # 降低置信度阈值，从默认0.5降到0.25
    iou=0.45,   # 降低IOU阈值，从0.7降到0.45
    imgsz=1280, # 提高推理分辨率，从640提高到1280
    augment=True, # 开启测试时增强
    agnostic_nms=False, # 类别感知的NMS
    max_det=300  # 增加最大检测数量
)
```

### 2. 实施两阶段检测策略（30分钟）
```python
def two_stage_detection(image_path):
    """两阶段检测：先检测大物体，再放大检测小物体"""
    
    # 第一阶段：正常检测
    results_stage1 = model.predict(
        source=image_path,
        conf=0.4,  # 较高置信度
        imgsz=640
    )
    
    # 第二阶段：高分辨率检测小物体
    results_stage2 = model.predict(
        source=image_path,
        conf=0.25,  # 更低置信度捕获小物体
        imgsz=1280,  # 更高分辨率
        augment=True
    )
    
    # 合并结果，去重
    return merge_results(results_stage1, results_stage2)
```

### 3. 添加类别特定处理（1小时）
```python
# 针对容易漏检和误检的类别设置不同阈值
class_configs = {
    'Prawn': {'conf': 0.2, 'min_area': 400},     # 虾-降低阈值
    'Carrot': {'conf': 0.25, 'min_area': 600},   # 胡萝卜-降低阈值
    'Onion': {'conf': 0.5, 'min_area': 800},     # 洋葱-提高阈值减少误检
    'Capsicum': {'conf': 0.5, 'min_area': 800},  # 辣椒-提高阈值减少误检
}
```

## 📸 短期改进方案（1-3天）

### 1. 收集冰箱特定数据（第1天）

**具体步骤：**
1. 拍摄50-100张您实际冰箱的照片
2. 包含不同情况：
   - 不同光照（开灯/关灯/半开）
   - 不同角度（正面/侧面/俯视）
   - 不同摆放（整齐/杂乱/堆叠）
   - 特别注意拍摄小物体的特写

3. 使用您已有的模型进行预标注：
```python
# 自动预标注脚本
for image in fridge_images:
    results = model.predict(image, conf=0.2)
    # 保存为YOLO格式标签，然后手动修正
    results.save_txt('labels/')
```

4. 手动修正标注，特别注意：
   - 添加漏检的小物体
   - 删除误检的标注
   - 添加"negative"类别（非食物物品）

### 2. 数据增强策略（第2天）

创建专门的增强脚本：
```python
import albumentations as A

# 冰箱场景专用增强
transform = A.Compose([
    # 模拟冰箱光照
    A.RandomBrightnessContrast(
        brightness_limit=(-0.3, 0.2),  # 冰箱内通常较暗
        contrast_limit=(-0.2, 0.3),
        p=0.8
    ),
    
    # 模拟冰箱门反光
    A.RandomSunFlare(
        flare_roi=(0, 0, 0.3, 0.3),
        angle_lower=0,
        angle_upper=1,
        num_flare_circles_lower=1,
        num_flare_circles_upper=2,
        src_radius=30,
        p=0.3
    ),
    
    # 模拟物品遮挡
    A.CoarseDropout(
        max_holes=3,
        max_height=50,
        max_width=50,
        fill_value=0,
        p=0.3
    ),
    
    # 专门增强小物体
    A.RandomSizedBBoxSafeCrop(
        height=640,
        width=640,
        erosion_rate=0.0,
        interpolation=cv2.INTER_LINEAR,
        p=0.5
    ),
])
```

### 3. 微调模型（第3天）

使用新数据微调：
```bash
yolo train \
    model=/content/drive/MyDrive/yolo_runs75/foodlab_v10/weights/best.pt \
    data=fridge_data.yaml \
    epochs=30 \
    imgsz=960 \
    batch=8 \
    lr0=0.0001 \
    cos_lr=True \
    close_mosaic=10 \
    copy_paste=0.3 \
    mixup=0.15 \
    mosaic=1.0 \
    hsv_h=0.015 \
    hsv_s=0.7 \
    hsv_v=0.4 \
    degrees=10.0 \
    translate=0.2 \
    scale=0.9 \
    flipud=0.1 \
    fliplr=0.5 \
    cache=True \
    patience=20
```

## 🚀 中期优化方案（1周）

### 1. 尝试更大的模型
```bash
# 使用YOLOv10s（小型）代替nano
yolo train model=yolov10s.pt data=data.yaml epochs=100 imgsz=960

# 或者YOLOv10m（中型）如果GPU允许
yolo train model=yolov10m.pt data=data.yaml epochs=100 imgsz=960
```

### 2. 实施级联检测
```python
class CascadeDetector:
    def __init__(self):
        self.detector1 = YOLO('yolov10n.pt')  # 快速初筛
        self.detector2 = YOLO('yolov10s.pt')  # 精确检测
        
    def detect(self, image):
        # 第一级：快速检测
        stage1 = self.detector1(image, conf=0.3)
        
        # 第二级：对可疑区域精确检测
        roi_results = []
        for box in stage1.boxes:
            roi = crop_roi(image, box)
            stage2 = self.detector2(roi, conf=0.4)
            roi_results.append(stage2)
            
        return combine_results(stage1, roi_results)
```

### 3. 添加后处理规则
```python
def post_process_rules(detections):
    """基于领域知识的后处理"""
    
    # 规则1：冰箱中butter(黄油)通常是方形包装
    for det in detections:
        if det.class_name == 'Butter':
            aspect_ratio = det.width / det.height
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                det.confidence *= 0.5  # 降低置信度
    
    # 规则2：Prawn(虾)通常成群出现
    prawn_count = sum(1 for d in detections if d.class_name == 'Prawn')
    if prawn_count == 1:  # 只检测到一只虾，可能是误检
        for det in detections:
            if det.class_name == 'Prawn':
                det.confidence *= 0.7
    
    # 规则3：某些食材不会太大
    max_sizes = {
        'Egg': 0.1,  # 占图像最大10%
        'Prawn': 0.05,
        'Butter': 0.15
    }
    
    image_area = 1280 * 960
    for det in detections:
        if det.class_name in max_sizes:
            det_area = det.width * det.height
            if det_area > image_area * max_sizes[det.class_name]:
                det.confidence *= 0.3  # 大幅降低置信度
    
    return detections
```

## 📝 测试验证方案

### 创建测试集
1. 准备20张冰箱照片作为测试集
2. 手动标注ground truth
3. 包含各种难例：
   - 小物体密集区域
   - 物品重叠区域
   - 光照不良区域
   - 易误检物品

### 评估指标
```python
def evaluate_fridge_detection(model, test_images):
    metrics = {
        'small_object_recall': 0,  # 小物体召回率
        'false_positive_rate': 0,  # 误检率
        'per_class_ap': {},        # 每类AP
    }
    
    for image in test_images:
        pred = model(image)
        gt = load_ground_truth(image)
        
        # 计算小物体召回率（<32x32像素）
        small_recalls = calculate_small_object_recall(pred, gt)
        
        # 计算误检率
        fp_rate = calculate_false_positive_rate(pred, gt)
        
        # 更新指标
        metrics['small_object_recall'] += small_recalls
        metrics['false_positive_rate'] += fp_rate
    
    return metrics
```

## ⚡ 快速测试命令

在Google Colab中运行：
```python
# 1. 测试不同置信度阈值
for conf in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    results = model.predict(
        '/content/test_fridge.jpg',
        conf=conf,
        imgsz=1280,
        augment=True
    )
    print(f"Conf={conf}: 检测到 {len(results[0].boxes)} 个物体")
    results[0].save(f'result_conf_{conf}.jpg')

# 2. 测试不同分辨率
for imgsz in [640, 960, 1280]:
    results = model.predict(
        '/content/test_fridge.jpg',
        conf=0.3,
        imgsz=imgsz
    )
    print(f"Size={imgsz}: 检测到 {len(results[0].boxes)} 个物体")

# 3. 对比增强前后
results_normal = model.predict('/content/test_fridge.jpg', augment=False)
results_augment = model.predict('/content/test_fridge.jpg', augment=True)
print(f"Normal: {len(results_normal[0].boxes)} vs Augment: {len(results_augment[0].boxes)}")
```

## 🎯 建议的执行顺序

1. **立即（10分钟）**：调整推理参数，测试效果
2. **今天（2小时）**：实施两阶段检测和类别特定阈值
3. **明天（4小时）**：收集和标注冰箱数据
4. **后天（3小时）**：数据增强并微调模型
5. **本周末**：测试更大模型和级联检测

## 💡 关键建议

1. **最重要的是数据**：您需要更多真实冰箱场景的训练数据
2. **分辨率很关键**：小物体检测必须用高分辨率（至少960，最好1280）
3. **后处理规则有效**：基于领域知识的规则可以显著减少误检
4. **测试时增强(TTA)值得尝试**：虽然慢一些但准确率会提高

您可以先从调整推理参数开始，这是最快见效的方法。然后逐步实施其他改进。有任何问题随时问我！