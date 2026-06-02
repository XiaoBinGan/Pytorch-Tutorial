"""最小单样本推理入口。

用法：
    python infer.py 路径\到\图片.png

这个脚本负责：
- 加载 best_model.pt
- 读取一张本地图片
- 做与 MNIST 兼容的预处理
- 输出预测类别
"""

import os
import sys
import torch
from PIL import Image
from torchvision import transforms

from config import CHECKPOINT_DIR, NUM_CLASSES
from models import MNISTNet


def load_model():
    """加载训练好的最佳模型。"""
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"未找到 {best_model_path}，请先运行 python train.py 训练并保存模型。"
        )

    model = MNISTNet(num_classes=NUM_CLASSES)
    state_dict = torch.load(best_model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_image(image_path):
    """读取单张图片并转成模型能接受的输入张量。"""
    transform = transforms.Compose([
        transforms.Grayscale(),      # 转单通道灰度图
        transforms.Resize((28, 28)), # 调整到 MNIST 模型要求的尺寸
        transforms.ToTensor(),       # 转成张量并缩放到 [0, 1]
    ])

    image = Image.open(image_path)
    return transform(image).unsqueeze(0)


def predict(model, x):
    """执行推理并返回预测类别 id。"""
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
    return pred


def main():
    if len(sys.argv) < 2:
        raise SystemExit("用法: python infer.py 路径\\到\\图片.png")

    image_path = sys.argv[1]
    model = load_model()
    x = load_image(image_path)
    pred = predict(model, x)

    print(f"输入图片: {image_path}")
    print(f"输入 shape: {tuple(x.shape)}")
    print(f"预测结果: {pred}")


if __name__ == "__main__":
    main()
