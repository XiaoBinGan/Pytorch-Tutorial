# EP07｜分类任务：手写数字识别


<p align="center">
 <img src="https://pytorch.org/assets/images/pytorch-logo.png" width="400" alt="PyTorch">
</p>

<h1 align="center">PyTorch 你不学</h1>
<h2 align="center">EP07 分类任务：手写数字识别</h2>
<h4 align="center">吴佳浩 · 著</h4>
<h3 align="center">15 篇 · 本地 Windows 实战 · RTX 5090 实测</h3>

<p align="center">
 <img src="https://img.shields.io/badge/PyTorch-2.12.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
 <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
 <img src="https://img.shields.io/badge/GPU-RTX%205090-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="NVIDIA">
 <img src="https://img.shields.io/badge/OS-Windows%2011-0078D6?style=for-the-badge&logo=windows&logoColor=white" alt="Windows">
 <img src="https://img.shields.io/badge/Chapters-15-FF6B35?style=for-the-badge&logo=bookstack&logoColor=white" alt="15 Chapters">
</p>

<p align="center">
  <b>[Thu 2026-06-11 02:04 GMT+8] 作者：吴佳浩 | 撰稿：2026-05-25 | 实测：RTX 5090 + 96GB + Windows 11</b>
</p>


## 本篇你会学到什么

这一篇承接训练循环，第一次把前面的基础能力放进一个完整、标准、可复现的图像分类任务里。学完这一篇，你应该能够：

- 用 `torchvision` 自带的 MNIST 数据集完成一个完整分类任务
- 理解图像分类任务中输入 shape、类别数、预测结果之间的关系
- 知道一个完整图像分类流程如何从数据、模型、训练、评估串起来
- 学会把前面学过的 Tensor、Module、DataLoader、训练循环组合成一个真正像样的项目案例
- 能看懂分类任务里 accuracy 是怎么计算出来的
- 能在本地 Windows 环境把第一个可运行的图像分类实验跑通

---

## 一、为什么 MNIST 是入门分类任务的经典选择

如果你第一次做图像分类，直接上大数据集通常会很痛苦。因为你会同时遇到：

- 数据量大
- 模型复杂
- 训练时间长
- 调试成本高
- 报错来源多

而 MNIST 很适合作为入门任务，原因在于：

- 数据集规模适中
- 图片尺寸统一（28×28）
- 类别明确（0~9 共 10 类）
- 数据格式干净
- 社区资料多
- 本地 CPU 环境也能跑起来

所以学 MNIST 的重点不是“我以后就只做手写数字识别”，而是：

> **用一个足够简单、足够标准的任务，把完整分类流程真正跑通。**

这一步非常重要，因为它会把你前面学过的知识真正串起来：

- Tensor
- DataLoader
- `nn.Module`
- loss
- optimizer
- 训练循环
- 预测与评估

---

## 二、核心理论讲解

### 1. 图像分类任务在做什么

图像分类的目标是：

> 给定一张图片，判断它属于哪一个类别。

在 MNIST 里：

- 输入是一张灰度图片
- 输出是 `0~9` 这 10 个数字中的一个

所以这是一个**10 分类问题**。

### 2. MNIST 数据长什么样

MNIST 中每张图片：

- 高度：28
- 宽度：28
- 通道数：1（灰度图）

经过 `ToTensor()` 转换后，单张图片通常会变成：

- shape = `(1, 28, 28)`

如果一个 batch 有 64 张图片，那么输入 shape 常见是：

- `(64, 1, 28, 28)`

这里四个维度分别表示：

- batch size
- channel
- height
- width

### 3. 为什么很多入门模型要先 `Flatten`

如果你用的是最基础的全连接网络，而不是 CNN，那么线性层需要接收一维特征向量。

但图像原本是二维（再加通道维），所以你常常需要先把：

- `(1, 28, 28)`

展平成：

- `(784,)`

因为：

- `28 × 28 = 784`

所以很多入门版 MNIST 模型都会先用 `nn.Flatten()`。

### 4. 模型输出为什么是 10 维

因为这是 10 分类任务。

模型最后一层通常输出 shape：

- `(batch_size, 10)`

这 10 个值不是最终标签，而是对 10 个类别的原始打分（logits）。

后续再通过：

```python
preds = torch.argmax(logits, dim=1)
```

取分数最高的类别，作为最终预测结果。

### 5. Accuracy 是怎么计算的

分类任务里，最基础的指标通常是准确率（accuracy）：

> 预测正确的样本数 / 总样本数

例如：

- 测试集中一共有 10000 张图片
- 预测对了 9200 张

那么准确率就是：

- `92%`

它是最直观的分类任务指标。

---

## 三、先建立一个直觉理解

你可以把 MNIST 分类任务理解成一个“让模型学会看图认数字”的过程。

流程大致是：

1. 模型看到很多带标签的数字图片
2. 每看一批图片，就先做预测
3. 再把预测和真实数字对比，计算错误程度
4. 根据错误反向传播更新参数
5. 经过很多轮之后，模型逐渐学会识别数字形状模式

这里真正重要的不是“模型会不会认 7 和 9”，而是你要建立起这种项目级理解：

- 数据如何被读取
- 图片如何变成 Tensor
- Tensor 如何进入模型
- 模型输出如何变成预测类别
- 训练和评估是如何分开的

---

## 四、真实项目里怎么用

### 场景 1：图像分类流程入门模板

很多更复杂的图像分类项目，本质上都沿用了和 MNIST 类似的骨架：

- 准备数据集
- DataLoader 批量读取
- 定义模型
- 定义损失函数和优化器
- 训练多个 epoch
- 在验证集 / 测试集上评估准确率

所以你今天学的是 MNIST，实际上是在学图像分类项目的基础模板。

### 场景 2：后续替换成更真实的数据集

以后你可能会做：

- 猫狗分类
- 工业缺陷检测中的分类子任务
- 医学影像分类
- 文档图片分类

虽然数据内容变了，但流程骨架并没有本质变化。

这也是为什么这篇非常重要：

> **它是第一个真正把前面知识组合成完整项目流的节点。**

---

## 五、流程图 / 结构图（Mermaid）

下面这张图展示了 MNIST 分类项目的完整数据流：

```mermaid
flowchart LR
    A[MNIST 图片数据集] --> B[ToTensor 转成 Tensor]
    B --> C[DataLoader 组装 batch]
    C --> D[模型前向传播]
    D --> E[输出 10 类 logits]
    E --> F[CrossEntropyLoss 计算损失]
    F --> G[反向传播与优化器更新]
    G --> H[测试集评估 accuracy]
```

看这张图时，你要意识到：

- 分类任务不是只有“模型”这一步
- 数据读取、训练、评估是完整闭环

---

## 六、从零写一个最小可运行示例

下面我们直接写一个完整可运行的 MNIST 手写数字识别示例。

这个版本特意保持“足够真实，但不过度复杂”，方便你在本地 Windows 环境直接跑起来。

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 1. 统一设备管理
# 如果本机支持 CUDA，就优先使用 GPU；否则自动退回 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("当前设备:", device)

# 2. 定义数据预处理
# ToTensor() 会把 PIL 图片或 ndarray 转成 Tensor，
# 并把像素值从 [0, 255] 缩放到 [0, 1]
transform = transforms.ToTensor()

# 3. 准备训练集和测试集
# root 可以根据你的本地目录习惯调整
# 第一次运行时，如果 download=True，会自动下载数据
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# 4. 使用 DataLoader 组织 batch
# 训练集通常 shuffle=True，测试集一般不打乱
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)


class MNISTNet(nn.Module):
    """
    一个最基础的 MNIST 全连接分类模型。

    输入：
        shape = (batch_size, 1, 28, 28)

    流程：
        先展平为 784 维，再经过两层线性层完成 10 分类

    输出：
        shape = (batch_size, 10)
        表示 10 个类别的原始打分（logits）
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 把图像从 (1, 28, 28) 展平成 784 维向量
            nn.Flatten(),

            # 第一层全连接：784 -> 128
            nn.Linear(28 * 28, 128),

            # 非线性激活函数
            nn.ReLU(),

            # 输出层：128 -> 10，对应 10 个数字类别
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


# 5. 实例化模型、损失函数、优化器
model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 6. 训练 3 个 epoch 做演示
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        # 把图片和标签迁移到统一设备
        images = images.to(device)
        labels = labels.to(device)

        # 清空上一轮梯度
        optimizer.zero_grad()

        # 前向传播
        logits = model(images)

        # 计算分类损失
        loss = criterion(logits, labels)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        total_loss += loss.item()

        # 打印前几个 batch 的信息，帮助确认 shape 和训练是否正常
        if batch_idx < 2:
            print(
                f"[epoch {epoch+1} | batch {batch_idx+1}] "
                f"images.shape={images.shape}, labels.shape={labels.shape}, "
                f"logits.shape={logits.shape}, loss={loss.item():.4f}"
            )

    avg_loss = total_loss / len(train_loader)
    print(f"epoch={epoch+1}, avg_loss={avg_loss:.4f}")


# 7. 在测试集上评估准确率
model.eval()
correct = 0
total = 0

# 评估阶段不需要计算梯度，可以节省内存和计算开销
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        # 取每条样本打分最高的类别，作为预测结果
        preds = torch.argmax(logits, dim=1)

        # 统计预测正确的样本数
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"test accuracy = {accuracy:.4f}")
```

---

## 七、拆解代码执行过程

### 1. `ToTensor()` 做了什么

这一点非常关键。

`transforms.ToTensor()` 不只是“换个数据类型”，它还做了两件对训练很重要的事：

- 把图片转成 PyTorch Tensor
- 把像素值缩放到 `[0, 1]`

这让后续模型更容易处理输入数据。

### 2. 为什么训练集和测试集要分开

这是机器学习的基本原则。

- **训练集**：用来学习参数
- **测试集**：用来评估模型是否真的学到了泛化能力

如果你只看训练集表现，很可能会误以为模型很强，实际上它只是记住了训练数据。

### 3. `nn.Flatten()` 的作用

单张 MNIST 图片原本 shape 是：

- `(1, 28, 28)`

但全连接层 `nn.Linear` 期望输入最后一维是特征向量，所以需要先把图像展平成：

- `(784,)`

如果是一个 batch，那么就会变成：

- `(batch_size, 784)`

### 4. 模型输出为什么是 `(batch_size, 10)`

因为这是 10 分类任务。

每条样本输出 10 个类别打分。举例来说：

- 第 0 维分数代表“这张图是数字 0”的置信打分
- 第 1 维分数代表“这张图是数字 1”的打分
- ……
- 第 9 维分数代表“这张图是数字 9”的打分

### 5. 评估阶段为什么要 `model.eval()` 和 `torch.no_grad()`

这是非常重要的工程习惯。

- `model.eval()`：告诉模型进入评估模式
- `torch.no_grad()`：关闭梯度计算，减少内存和计算开销

虽然这个简单模型里没有 Dropout / BatchNorm，但养成这个习惯非常重要，因为你后面做更复杂模型时一定会用到。

---

## 八、运行结果应该怎么看

运行后，建议重点看以下几个方面。

### 1. 图片输入 shape 是否正确

你应该看到类似：

- `images.shape = (64, 1, 28, 28)`

这表示：

- batch size = 64
- 灰度图通道数 = 1
- 图片尺寸 = 28 × 28

### 2. 模型输出 shape 是否正确

你应该看到：

- `logits.shape = (64, 10)`

这说明模型确实在做 10 分类。

### 3. loss 是否能够正常下降

刚开始 loss 可能比较大，比如：

- `2.1`
- `1.7`
- `1.2`

随着训练推进，通常会逐步下降。

MNIST 是比较容易学的任务，所以如果训练流程正确，loss 通常会比较明显地往下走。

### 4. accuracy 是否达到一个合理水平

即使用这个非常基础的全连接模型，跑几轮之后通常也能得到一个还不错的准确率。

如果准确率非常低，优先检查：

- 数据是否正常加载
- 模型输出维度是否正确
- 标签是否正确
- loss 和 optimizer 是否配置对了

---

## 九、常见错误与排查

### 问题 1：输入 shape 不对

典型报错往往和线性层维度不匹配有关。

例如你忘了 `Flatten()`，就可能导致：

- 模型以为输入是 784 维向量
- 实际却收到 `(1, 28, 28)` 的图像张量

排查建议：

- 在前向传播前打印 `images.shape`
- 在模型输出后打印 `logits.shape`

### 问题 2：把 logits 当成最终概率

模型输出的是 logits，不是概率。入门阶段最简单的做法是：

- 训练时直接把 logits 交给 `CrossEntropyLoss`
- 预测时再用 `argmax` 取类别

不要在训练前手动乱加 softmax，除非你很清楚损失函数期望什么输入。

### 问题 3：忘记切换评估模式

虽然这个例子里问题可能不明显，但在更复杂模型里，如果评估时不写：

```python
model.eval()
```

结果可能不稳定。

### 问题 4：Windows 本地下载数据慢或失败

第一次运行会下载数据到本地 `data` 目录。

如果网络不稳定，可能出现：

- 下载慢
- 下载失败
- 需要重试

这通常不是代码逻辑问题，而是网络和数据下载问题。

### 问题 5：CPU 训练太慢

如果你本机没有 GPU，MNIST 依然能跑，但速度会慢一些。入门阶段这是完全可以接受的，因为我们的重点是把完整流程学会，而不是追求极致速度。

---

## 十、本篇小结

这一篇非常关键，因为它是你第一次真正把前面的知识串成一个完整项目：

- 用 `torchvision` 加载数据
- 用 DataLoader 组织 batch
- 用 `nn.Module` 定义模型
- 用损失函数和优化器训练
- 用测试集评估 accuracy

你学到的不只是“怎么识别手写数字”，更重要的是：

> **一个标准图像分类任务从数据到评估的完整骨架。**

后面你做更复杂的图像项目时，大概率仍然会沿用这个流程，只是数据更复杂、模型更强而已。

---

## 十一、练习题

### 练习 1：修改 batch size
把 `batch_size` 从 `64` 改成：

- `32`
- `128`

观察训练打印信息和每轮训练节奏有什么变化。

### 练习 2：修改隐藏层大小
把全连接隐藏层从 `128` 改成 `256`，观察参数量和训练表现有没有变化。

### 练习 3：打印单张图片 shape
从 `train_dataset[0]` 取出一张图片和标签，打印：

- 图片 shape
- 标签值

确认你真正理解单条样本和 batch 样本的区别。

### 练习 4：手动计算 accuracy
在测试集评估阶段，尝试自己打印：

- 总样本数
- 正确样本数
- 最终 accuracy

确认你理解准确率的计算过程。

### 练习 5：思考题
为什么说 MNIST 重要的不是“认数字本身”，而是“让你第一次把完整分类项目流程跑通”？

试着用自己的话总结一下。

---

## 下一篇预告

下一篇我们会进一步升级图像模型本身，进入 **卷积神经网络 CNN 入门**。

到那时你会开始理解：

- 为什么图像不能总是简单拍平成向量处理
- 卷积层和池化层是怎么逐步提取图像局部特征的
- 为什么 CNN 会成为视觉任务最经典的基础模型
