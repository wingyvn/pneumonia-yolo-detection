"""
YOLO 检测器封装模块
====================

将 YOLO 模型的加载和推理封装为简洁的接口，
供 GUI 层调用，避免 UI 代码直接依赖 ultralytics 细节。
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


# 类别名称（与 data.yaml 保持一致）
CLASS_NAMES = ['Pneumonia Bacteria', 'Pneumonia Virus', 'Sick', 'healthy', 'tuberculosis']

# 类别对应的中文名称（用于界面显示）
CLASS_NAMES_CN = ['细菌性肺炎', '病毒性肺炎', '患病', '健康', '肺结核']

# 类别对应颜色（RGB 格式，用于界面标识）
CLASS_COLORS = [
    (229, 57, 53),    # 红 — 细菌性肺炎
    (30, 136, 229),   # 蓝 — 病毒性肺炎
    (253, 216, 53),   # 黄 — 患病
    (67, 160, 71),    # 绿 — 健康
    (142, 36, 170),   # 紫 — 肺结核
]


class PneumoniaDetector:
    """
    肺炎检测器：封装 YOLO 模型的加载与推理。

    用法:
        detector = PneumoniaDetector('runs/detect/pneumonia_exp1/weights/best.pt')
        detections, annotated_image = detector.detect(pil_image, conf=0.25, iou=0.45)
    """

    def __init__(self, model_path: str = None):
        """
        初始化检测器。

        Args:
            model_path: 模型权重文件路径 (.pt)，为 None 时不加载模型
        """
        self.model = None
        self.model_path = None

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        """
        加载或切换模型。

        Args:
            model_path: .pt 模型文件路径

        Returns:
            是否加载成功
        """
        try:
            self.model = YOLO(model_path)
            self.model_path = model_path
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False

    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None

    def detect(self, image: Image.Image, conf: float = 0.25, iou: float = 0.45) -> tuple:
        """
        对单张图片执行检测。

        Args:
            image: PIL Image 对象
            conf: 置信度阈值（0~1），低于此值的检测框被过滤
            iou: IoU 阈值（0~1），用于 NMS 去重

        Returns:
            (detections, annotated_image)
            - detections: 列表，每个元素为字典:
                {
                    'class_id': int,          # 类别索引
                    'class_name': str,        # 英文类别名
                    'class_name_cn': str,     # 中文类别名
                    'confidence': float,      # 置信度
                    'bbox': [x1, y1, x2, y2], # 边界框像素坐标
                    'bbox_norm': [cx, cy, w, h], # 归一化坐标（用于可视化）
                    'area_ratio': float       # 检测框面积占图像面积的比例
                }
            - annotated_image: numpy 数组 (RGB)，带有标注框的图像
        """
        if not self.is_loaded():
            return [], None

        # 执行推理
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            verbose=False
        )

        detections = []
        annotated_image = None

        # 获取图像尺寸用于计算面积比
        img_w, img_h = image.size
        img_area = img_w * img_h

        for r in results:
            # 生成标注图（YOLO 返回 BGR，转为 RGB）
            annotated_image = r.plot()[:, :, ::-1].copy()

            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # 计算归一化中心坐标和宽高
                cx = (x1 + x2) / 2 / img_w
                cy = (y1 + y2) / 2 / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h

                # 计算面积占比
                box_area = (x2 - x1) * (y2 - y1)
                area_ratio = box_area / img_area if img_area > 0 else 0

                detections.append({
                    'class_id': cls_id,
                    'class_name': CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'class_{cls_id}',
                    'class_name_cn': CLASS_NAMES_CN[cls_id] if cls_id < len(CLASS_NAMES_CN) else f'类别_{cls_id}',
                    'confidence': round(conf_val, 4),
                    'bbox': [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    'bbox_norm': [round(cx, 4), round(cy, 4), round(bw, 4), round(bh, 4)],
                    'area_ratio': round(area_ratio, 4)
                })

        return detections, annotated_image

    def annotate_image(self, image_path: str, detections: list) -> np.ndarray | None:
        """
        基于已保存的检测结果重新绘制标注图，避免在批量检测信号中传输大数组。

        Args:
            image_path: 原图路径
            detections: detect() 返回的检测结果列表

        Returns:
            RGB numpy 数组，失败时返回 None
        """
        if not os.path.exists(image_path):
            return None

        try:
            image = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(image)
            font = self._load_label_font(image.size)

            for det in detections:
                cls_id = det.get('class_id', -1)
                color = CLASS_COLORS[cls_id] if 0 <= cls_id < len(CLASS_COLORS) else (150, 150, 150)
                bbox = det.get('bbox', [])
                if len(bbox) != 4:
                    continue

                x1, y1, x2, y2 = bbox
                label = f"{det.get('class_name', 'unknown')} {det.get('confidence', 0):.2f}"

                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                if font is not None:
                    left, top, right, bottom = draw.textbbox((x1, y1), label, font=font)
                    text_w = right - left
                    text_h = bottom - top
                else:
                    text_w = int(max(48, len(label) * 7))
                    text_h = 14

                text_y = max(0, y1 - text_h - 6)
                draw.rectangle([x1, text_y, x1 + text_w + 8, text_y + text_h + 6], fill=color)
                draw.text((x1 + 4, text_y + 3), label, fill='white', font=font)

            return np.array(image)
        except Exception:
            return None

    @staticmethod
    def _load_label_font(image_size: tuple) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """优先加载 Windows 常见字体，失败时退回 PIL 默认字体。"""
        font_size = max(14, int(min(image_size) * 0.025))
        for font_name in ['msyh.ttc', 'simhei.ttf', 'arial.ttf']:
            try:
                return ImageFont.truetype(font_name, font_size)
            except OSError:
                continue
        return ImageFont.load_default()


def find_available_models(runs_dir: str = './runs/detect') -> dict:
    """
    自动扫描 runs/detect 目录下所有可用的模型权重文件。

    Returns:
        字典 {显示名称: 文件路径}，例如:
        {'pneumonia_exp1/best.pt': 'runs/detect/pneumonia_exp1/weights/best.pt'}
    """
    models = {}

    if not os.path.isdir(runs_dir):
        return models

    for exp_name in os.listdir(runs_dir):
        weights_dir = os.path.join(runs_dir, exp_name, 'weights')
        if os.path.isdir(weights_dir):
            for weight_file in ['best.pt', 'last.pt']:
                weight_path = os.path.join(weights_dir, weight_file)
                if os.path.exists(weight_path):
                    display_name = f"{exp_name}/{weight_file}"
                    models[display_name] = weight_path

    return models
