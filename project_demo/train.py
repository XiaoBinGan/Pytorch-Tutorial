"""训练主入口。

这个脚本现在负责：
- 构建训练 / 验证 / 测试数据加载器
- 实例化模型、损失函数、优化器
- 执行训练循环
- 用验证集选择更好的模型
- 使用 val_loss 做 early stopping
- 保存 last_checkpoint.pt 和 best_model.pt
- 输出 history.json 和训练曲线图
"""

import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    CHECKPOINT_DIR,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LR,
    NUM_CLASSES,
    OUTPUT_DIR,
)
from datasets import build_loaders
from models import MNISTNet


def evaluate(model, data_loader, criterion, device):
    """在验证集或测试集上计算平均 loss 和 accuracy。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / len(data_loader)
    acc = correct / total
    return avg_loss, acc


def save_history(history, output_dir):
    """把训练历史保存成 JSON，便于后续分析。"""
    history_path = os.path.join(output_dir, "history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"训练历史已保存到: {history_path}")


def plot_history(history, output_dir):
    """保存 loss / accuracy 曲线图。"""
    epochs = history["epoch"]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], marker="o", label="train_loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_acc"], marker="o", label="val_acc")
    plt.plot(epochs, history["test_acc"], marker="o", label="test_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Validation / Test Accuracy")
    plt.legend()

    plt.tight_layout()
    figure_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(figure_path, dpi=150)
    plt.close()
    print(f"训练曲线已保存到: {figure_path}")


def main():
    train_loader, val_loader, test_loader = build_loaders()

    model = MNISTNet(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    bad_epochs = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "test_acc": [],
    }

    print(f"当前设备: {DEVICE}")
    print(f"训练集 batch 数: {len(train_loader)}")
    print(f"验证集 batch 数: {len(val_loader)}")
    print(f"测试集 batch 数: {len(test_loader)}")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 只在最前面打印少量 batch 信息，帮助快速检查数据和 shape
            if batch_idx < 2 and epoch == 0:
                print(
                    f"batch={batch_idx+1}, x.shape={tuple(x.shape)}, "
                    f"logits.shape={tuple(logits.shape)}, loss={loss.item():.4f}"
                )

        train_loss = running_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        _, test_acc = evaluate(model, test_loader, criterion, DEVICE)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)

        print(
            f"epoch={epoch+1}, train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, test_acc={test_acc:.4f}"
        )

        # 保存最新断点，便于中断后恢复训练
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "history": history,
            },
            os.path.join(CHECKPOINT_DIR, "last_checkpoint.pt"),
        )

        # 以验证集 loss 为准保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            print("验证集表现提升，已更新 best_model.pt")
        else:
            bad_epochs += 1
            print(f"验证集未改善，bad_epochs={bad_epochs}")

        # early stopping：连续若干轮验证集不改善则提前停止
        if bad_epochs >= EARLY_STOPPING_PATIENCE:
            print("触发 early stopping，提前结束训练。")
            break

    save_history(history, OUTPUT_DIR)
    plot_history(history, OUTPUT_DIR)
    print("训练完成。")


if __name__ == "__main__":
    main()
