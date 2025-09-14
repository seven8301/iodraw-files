"""
优化的推理管道 - 解决误检和漏检问题
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict, Tuple
import json
from pathlib import Path

class OptimizedInferencePipeline:
    def __init__(self, model_path: str):
        """
        初始化优化的推理管道
        
        Args:
            model_path: YOLO模型路径
        """
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 优化的置信度阈值（基于您的测试反馈调整）
        self.confidence_thresholds = {
            0: 0.5,   # Cooked_Rice
            1: 0.45,  # Egg
            2: 0.55,  # Tofu
            3: 0.6,   # Onion - 容易误检，提高阈值
            4: 0.6,   # Carrot - 容易误检，提高阈值
            5: 0.5,   # Tomato
            6: 0.6,   # Capsicum - 容易误检，提高阈值
            7: 0.5,   # Chicken_Breast
            8: 0.65,  # Prawn - 小物体，提高阈值避免误检
            9: 0.55,  # Butter
            10: 0.6,  # Potato - 容易误检，提高阈值
            11: 0.5   # Beef_Steak
        }
        
        # 类别名称映射
        self.class_names = [
            'Cooked_Rice', 'Egg', 'Tofu', 'Onion', 'Carrot', 'Tomato',
            'Capsicum', 'Chicken_Breast', 'Prawn', 'Butter', 'Potato', 'Beef_Steak'
        ]
        
        # 小物体面积阈值（像素）
        self.small_object_threshold = 32 * 32
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像以提高检测效果
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        # 1. 增强对比度（CLAHE）
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 2. 降噪
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        # 3. 锐化（有助于小物体检测）
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened
    
    def multi_scale_detection(self, image: np.ndarray) -> List[Dict]:
        """
        多尺度检测以提高小物体检测率
        
        Args:
            image: 输入图像
            
        Returns:
            检测结果列表
        """
        scales = [1.0, 1.5, 2.0]  # 多个尺度
        all_detections = []
        
        original_height, original_width = image.shape[:2]
        
        for scale in scales:
            # 缩放图像
            scaled_width = int(original_width * scale)
            scaled_height = int(original_height * scale)
            
            # 限制最大尺寸避免内存溢出
            if scaled_width > 2560 or scaled_height > 2560:
                continue
                
            scaled_image = cv2.resize(image, (scaled_width, scaled_height))
            
            # 检测
            results = self.model(scaled_image, conf=0.3, iou=0.5, augment=True)
            
            # 将坐标转换回原始尺度
            for r in results:
                if r.boxes is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    scores = r.boxes.conf.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, score, cls in zip(boxes, scores, classes):
                        # 转换坐标
                        box = box / scale
                        
                        all_detections.append({
                            'box': box,
                            'score': score,
                            'class': cls,
                            'scale': scale
                        })
        
        return all_detections
    
    def filter_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        过滤检测结果，减少误检
        
        Args:
            detections: 原始检测结果
            
        Returns:
            过滤后的检测结果
        """
        filtered = []
        
        for det in detections:
            cls = det['class']
            score = det['score']
            box = det['box']
            
            # 1. 应用类别特定的置信度阈值
            if score < self.confidence_thresholds.get(cls, 0.5):
                continue
            
            # 2. 计算边界框面积
            area = (box[2] - box[0]) * (box[3] - box[1])
            
            # 3. 对小物体应用更严格的过滤
            if area < self.small_object_threshold:
                # 小物体需要更高的置信度
                if score < self.confidence_thresholds.get(cls, 0.5) + 0.1:
                    continue
            
            # 4. 过滤异常大小的检测框
            image_area = 1280 * 1280  # 假设图像大小
            if area > image_area * 0.5:  # 检测框超过图像一半
                continue
            
            # 5. 过滤边缘检测（可能是误检）
            margin = 10
            if (box[0] < margin or box[1] < margin or 
                box[2] > 1280 - margin or box[3] > 1280 - margin):
                # 边缘检测需要更高置信度
                if score < self.confidence_thresholds.get(cls, 0.5) + 0.15:
                    continue
            
            filtered.append(det)
        
        return filtered
    
    def nms_with_class(self, detections: List[Dict], iou_threshold: float = 0.4) -> List[Dict]:
        """
        类别感知的NMS
        
        Args:
            detections: 检测结果
            iou_threshold: IOU阈值
            
        Returns:
            NMS后的检测结果
        """
        if not detections:
            return []
        
        # 按类别分组
        class_groups = {}
        for det in detections:
            cls = det['class']
            if cls not in class_groups:
                class_groups[cls] = []
            class_groups[cls].append(det)
        
        final_detections = []
        
        # 对每个类别单独进行NMS
        for cls, dets in class_groups.items():
            # 按置信度排序
            dets = sorted(dets, key=lambda x: x['score'], reverse=True)
            
            keep = []
            while dets:
                # 保留置信度最高的
                keep.append(dets[0])
                dets = dets[1:]
                
                if not dets:
                    break
                
                # 计算IOU并过滤
                kept_box = keep[-1]['box']
                dets = [d for d in dets if self.calculate_iou(kept_box, d['box']) < iou_threshold]
            
            final_detections.extend(keep)
        
        return final_detections
    
    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """计算两个边界框的IOU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def detect_with_tta(self, image: np.ndarray) -> List[Dict]:
        """
        使用测试时增强(TTA)进行检测
        
        Args:
            image: 输入图像
            
        Returns:
            检测结果
        """
        all_detections = []
        
        # 原始图像检测
        results = self.model(image, conf=0.3, iou=0.5)
        all_detections.extend(self.parse_results(results, transform='none'))
        
        # 水平翻转
        flipped = cv2.flip(image, 1)
        results = self.model(flipped, conf=0.3, iou=0.5)
        all_detections.extend(self.parse_results(results, transform='flip_horizontal'))
        
        # 不同亮度
        for gamma in [0.8, 1.2]:
            adjusted = self.adjust_gamma(image, gamma)
            results = self.model(adjusted, conf=0.3, iou=0.5)
            all_detections.extend(self.parse_results(results, transform=f'gamma_{gamma}'))
        
        # 合并和投票
        final_detections = self.ensemble_detections(all_detections)
        
        return final_detections
    
    def adjust_gamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """调整图像gamma值"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def parse_results(self, results, transform: str) -> List[Dict]:
        """解析YOLO结果"""
        detections = []
        for r in results:
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)
                
                for box, score, cls in zip(boxes, scores, classes):
                    # 如果是翻转的，需要转换坐标
                    if transform == 'flip_horizontal':
                        # 假设图像宽度为1280
                        box[0], box[2] = 1280 - box[2], 1280 - box[0]
                    
                    detections.append({
                        'box': box,
                        'score': score,
                        'class': cls,
                        'transform': transform
                    })
        return detections
    
    def ensemble_detections(self, all_detections: List[Dict]) -> List[Dict]:
        """
        集成多个检测结果
        
        Args:
            all_detections: 所有检测结果
            
        Returns:
            集成后的结果
        """
        # 简单投票策略：如果同一位置被多次检测到，增加置信度
        final_detections = []
        processed = set()
        
        for i, det1 in enumerate(all_detections):
            if i in processed:
                continue
                
            # 找到所有重叠的检测
            overlapping = [det1]
            for j, det2 in enumerate(all_detections[i+1:], i+1):
                if j not in processed and det1['class'] == det2['class']:
                    iou = self.calculate_iou(det1['box'], det2['box'])
                    if iou > 0.5:
                        overlapping.append(det2)
                        processed.add(j)
            
            # 如果有多个重叠检测，取平均
            if len(overlapping) > 1:
                avg_box = np.mean([d['box'] for d in overlapping], axis=0)
                avg_score = np.mean([d['score'] for d in overlapping])
                # 增加置信度奖励
                bonus = min(0.1 * (len(overlapping) - 1), 0.3)
                final_detections.append({
                    'box': avg_box,
                    'score': min(avg_score + bonus, 1.0),
                    'class': det1['class']
                })
            else:
                final_detections.append(det1)
        
        return final_detections
    
    def detect(self, image_path: str, use_tta: bool = False, 
               use_multi_scale: bool = False) -> Dict:
        """
        主检测函数
        
        Args:
            image_path: 图像路径
            use_tta: 是否使用测试时增强
            use_multi_scale: 是否使用多尺度检测
            
        Returns:
            检测结果字典
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return {'error': 'Failed to load image'}
        
        # 预处理
        processed_image = self.preprocess_image(image)
        
        # 检测
        if use_tta:
            detections = self.detect_with_tta(processed_image)
        elif use_multi_scale:
            detections = self.multi_scale_detection(processed_image)
        else:
            results = self.model(processed_image, conf=0.3, iou=0.5, augment=True)
            detections = self.parse_results(results, transform='none')
        
        # 过滤
        filtered = self.filter_detections(detections)
        
        # NMS
        final = self.nms_with_class(filtered)
        
        # 格式化输出
        output = {
            'image_path': image_path,
            'detections': [],
            'ingredients': set()
        }
        
        for det in final:
            class_name = self.class_names[det['class']]
            output['detections'].append({
                'class': class_name,
                'confidence': float(det['score']),
                'bbox': det['box'].tolist()
            })
            output['ingredients'].add(class_name)
        
        output['ingredients'] = list(output['ingredients'])
        
        return output
    
    def visualize_results(self, image_path: str, detections: Dict, save_path: str = None):
        """
        可视化检测结果
        
        Args:
            image_path: 原始图像路径
            detections: 检测结果
            save_path: 保存路径
        """
        image = cv2.imread(image_path)
        
        # 绘制检测框
        for det in detections['detections']:
            box = det['bbox']
            x1, y1, x2, y2 = map(int, box)
            
            # 选择颜色
            color = (0, 255, 0) if det['confidence'] > 0.7 else (0, 165, 255)
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{det['class']}: {det['confidence']:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - label_size[1] - 5), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示或保存
        if save_path:
            cv2.imwrite(save_path, image)
        else:
            cv2.imshow('Detection Results', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return image


def test_pipeline():
    """测试优化的推理管道"""
    
    # 初始化管道
    pipeline = OptimizedInferencePipeline('/content/drive/MyDrive/yolo_runs75/foodlab_v10/weights/best.pt')
    
    # 测试图像
    test_image = '/path/to/your/fridge/image.jpg'
    
    # 标准检测
    print("标准检测...")
    results = pipeline.detect(test_image)
    print(f"检测到的食材: {results['ingredients']}")
    
    # 使用TTA检测（更准确但更慢）
    print("\n使用TTA检测...")
    results_tta = pipeline.detect(test_image, use_tta=True)
    print(f"检测到的食材 (TTA): {results_tta['ingredients']}")
    
    # 使用多尺度检测（针对小物体）
    print("\n使用多尺度检测...")
    results_ms = pipeline.detect(test_image, use_multi_scale=True)
    print(f"检测到的食材 (多尺度): {results_ms['ingredients']}")
    
    # 可视化结果
    pipeline.visualize_results(test_image, results, 'detection_results.jpg')
    
    return results


if __name__ == "__main__":
    test_pipeline()