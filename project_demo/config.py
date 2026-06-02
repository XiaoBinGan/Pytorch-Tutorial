"""project_demo 的统一配置文件。

这个文件集中管理：
- 数据目录
- 模型权重目录
- 输出目录
- 训练超参数
- 设备选择
- 验证集划分与 early stopping 参数

后面如果你要迁移到别的任务，优先改这里，
而不是去 train.py / eval.py / infer.py 里到处搜路径。
"""

import os
import torch

# 项目根目录：以当前 config.py 所在目录为基准，避免依赖命令行当前工作目录
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# 各类目录统一集中管理
DATA_DIR = os.path.join(PROJECT_DIR, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

# 训练超参数
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 5
NUM_CLASSES = 10
VAL_RATIO = 0.1
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 2

# 统一设备选择：有 GPU 就用，没有就自动退回 CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 为了让 train.py 第一次运行就不因为目录不存在而报错，这里顺手创建目录
for path in [DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR]:
    os.makedirs(path, exist_ok=True)
