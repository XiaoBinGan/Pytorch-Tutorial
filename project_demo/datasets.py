"""数据读取与 DataLoader 构建。

当前模板使用 MNIST 作为最小可运行示例，原因是：
- torchvision 内置支持，适合教学和快速验证
- 训练、评估、推理链路都容易搭起来
- 后面要替换成自己的数据集时，优先改这个文件

这次升级后，训练集会进一步拆成：
- train_loader
- val_loader

这样训练脚本就能真正区分：
- 训练指标
- 验证指标
- 最终测试指标
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from config import DATA_DIR, BATCH_SIZE, RANDOM_SEED, VAL_RATIO


def get_transforms():
    """返回训练和评估阶段共用的基础变换。"""
    return transforms.ToTensor()


def build_datasets():
    """构建原始训练集和测试集。"""
    transform = get_transforms()

    full_train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )

    return full_train_dataset, test_dataset


def split_train_val(full_train_dataset):
    """把原始训练集拆成训练集和验证集。"""
    total_size = len(full_train_dataset)
    val_size = int(total_size * VAL_RATIO)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator,
    )

    return train_dataset, val_dataset


def build_loaders():
    """构建训练、验证、测试三个 DataLoader。"""
    full_train_dataset, test_dataset = build_datasets()
    train_dataset, val_dataset = split_train_val(full_train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader
