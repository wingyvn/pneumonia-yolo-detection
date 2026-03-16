import os
import shutil
from pathlib import Path
import yaml

def verify_dataset_structure():
    """
    验证数据集结构是否完整
    """
    required_dirs = ['images/train', 'images/val', 'images/test',
                    'labels/train', 'labels/val', 'labels/test']
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"警告: {dir_path} 目录不存在")
            return False
    return True

def check_image_label_pairs():
    """
    检查图像和标签是否一一对应
    """
    splits = ['train', 'val', 'test']
    for split in splits:
        images_dir = f"images/{split}"
        labels_dir = f"labels/{split}"
        
        image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))}
        label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith('.txt')}
        
        missing_labels = image_files - label_files
        if missing_labels:
            print(f"{split}集中以下图像缺少标签文件:")
            for img in missing_labels:
                print(f"- {img}")

def validate_label_format():
    """
    验证标签格式是否符合YOLO格式
    """
    splits = ['train', 'val', 'test']
    for split in splits:
        labels_dir = f"labels/{split}"
        for label_file in os.listdir(labels_dir):
            if label_file.endswith('.txt'):
                file_path = os.path.join(labels_dir, label_file)
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines, 1):
                            parts = line.strip().split()
                            if len(parts) != 5:
                                print(f"错误: {file_path} 第{i}行格式不正确")
                                continue
                            
                            # 验证数值范围
                            try:
                                class_id = int(parts[0])
                                x, y, w, h = map(float, parts[1:])
                                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                                    print(f"警告: {file_path} 第{i}行坐标超出范围[0,1]")
                            except ValueError:
                                print(f"错误: {file_path} 第{i}行包含非数字值")
                except Exception as e:
                    print(f"无法读取文件 {file_path}: {str(e)}")

def main():
    """
    主函数：运行所有验证
    """
    print("开始验证数据集...")
    
    # 验证数据集结构
    if not verify_dataset_structure():
        print("数据集结构验证失败")
        return
    
    # 检查图像和标签对应关系
    print("\n检查图像和标签对应关系...")
    check_image_label_pairs()
    
    # 验证标签格式
    print("\n验证标签格式...")
    validate_label_format()
    
    print("\n数据集验证完成！")

if __name__ == "__main__":
    main()