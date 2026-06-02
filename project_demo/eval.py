"""独立评估入口。

这个脚本负责：
- 加载训练好的 best_model.pt
- 在测试集上重新做一次客观评估
- 输出最终准确率

把评估单独拆出来的价值是：
- 不依赖训练脚本当场状态
- 更贴近真实项目里的独立验证流程
"""

import os
import torch

from config import CHECKPOINT_DIR, DEVICE, NUM_CLASSES
from datasets import build_loaders
from models import MNISTNet


def main():
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"未找到 {best_model_path}，请先运行 python train.py 生成模型权重。"
        )

    _, _, test_loader = build_loaders()

    model = MNISTNet(num_classes=NUM_CLASSES).to(DEVICE)
    state_dict = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    print(f"评估设备: {DEVICE}")
    print(f"test_accuracy = {correct / total:.4f}")


if __name__ == "__main__":
    main()
