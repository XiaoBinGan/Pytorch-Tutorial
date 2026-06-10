# EP11｜迁移学习：基于 torchvision 预训练模型


<p align="center">
 <img src="https://pytorch.org/assets/images/pytorch-logo.png" width="400" alt="PyTorch">
</p>

<h1 align="center">PyTorch 你不学</h1>
<h2 align="center">EP11 迁移学习：基于 torchvision 预训练模型</h2>
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

这一篇承接前面的图像分类与模型工程能力，开始进入更贴近真实业务的训练策略。学完这一篇，你应该能够：

- 理解迁移学习为什么在中小数据集场景下很常用
- 区分“从零训练”和“使用预训练模型微调”的核心差别
- 学会用 `torchvision` 加载 `resnet18` 等预训练模型
- 学会替换最后一层分类头，让模型适配自己的类别数
- 学会冻结部分参数，只训练分类头
- 分清“在线加载预训练权重”和“离线只验证结构能跑”的区别
- 能检查哪些参数在训练，哪些参数被冻结

---

## 一、为什么很多真实项目不从零开始训练

如果你自己有一个很大的图像数据集、很多算力和足够训练时间，那么从零训练当然是一条路。

但现实里更常见的情况是：

- 数据量不大，可能只有几千到几万张图片
- 算力有限，未必有很强的 GPU 环境
- 项目周期紧，更希望先把可用效果做出来

这时，迁移学习往往是非常划算的选择。

它的核心思路是：

> **先利用一个已经在大规模数据集上学到通用视觉特征的模型，再把它迁移到你的具体任务上。**

比如 `torchvision` 中常见的 `resnet18`，通常已经在 ImageNet 这样的通用图像数据集上学过很多基础视觉模式：

- 边缘
- 纹理
- 颜色块
- 局部结构
- 中层语义模式

如果你的任务和自然图像有一定相似性，那么这些能力通常是有迁移价值的。

---

## 二、核心理论讲解

### 1. 什么是迁移学习

迁移学习（Transfer Learning）可以简单理解成：

> **把一个模型在旧任务上学到的知识，迁移到新任务上继续使用。**

在图像分类场景里，最常见的形式就是：

- 主干网络使用预训练参数
- 最后一层分类头替换成适合你当前任务类别数的新层
- 然后根据需要选择：
  - 只训练分类头
  - 或进一步微调整个模型

### 2. 为什么预训练模型有用

因为很多视觉任务共享一些底层模式。

例如不管你识别的是：

- 猫狗
- 花卉
- 工业零件
- 某些医学图像中的基础纹理模式

模型前几层提取的很多低层特征常常是相通的，比如边缘、纹理、局部形状。

这意味着：

- 你不必从随机参数开始重新学一遍最基础的视觉表示
- 可以直接站在已有知识上继续适配你的任务

### 3. 什么是“替换最后一层分类头”

像 `resnet18` 这类模型，原始设计通常对应 ImageNet 的 1000 分类任务。

所以它最后一层全连接层输出维度通常是：

- 1000

如果你自己的任务只有 2 类、5 类、10 类，就必须把最后一层换掉。

例如：

```python
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
```

这表示：

- 前面的特征提取主干保留
- 最后分类头改成适配你自己的类别数

### 4. 什么是冻结参数

冻结参数的意思是：

- 不让这些参数参与梯度更新
- 训练时保持它们不变

在 PyTorch 中常见写法是：

```python
param.requires_grad = False
```

为什么要冻结？

因为在小数据集场景里，直接把整个大模型全量微调，往往：

- 更容易过拟合
- 更耗时
- 更吃显存

所以一个很常见的入门策略是：

- 先冻结主干特征层
- 只训练新换上的分类头

### 5. 什么是“结构能跑”和“迁移学习真的生效”

这是非常容易混淆的点。

如果你写：

```python
model = resnet18(weights=None)
```

这只是说明：

- 你成功构建了模型结构
- 可以验证替换分类头的代码没写崩

但这**不等于真正完成了迁移学习**。

真正迁移学习的价值来自：

- 加载预训练权重

如果没有这些预训练参数，模型本质上还是随机初始化，只是结构长得像 ResNet 而已。

---

## 三、先建立一个直觉理解

你可以把迁移学习想成“请一个已经有通用经验的人来做新工作”。

这个人过去已经学会了很多基础能力，比如：

- 识别边缘
- 看懂形状
- 分辨纹理

现在你只需要让他：

- 适应你的新业务类别
- 学会最后的具体分类标准

而不是让他从“什么是边缘”开始重新学。

所以迁移学习的本质，不是偷懒，而是：

> **复用已有通用知识，减少从零学习的成本。**

---

## 四、真实项目里怎么用

### 场景 1：你只有几千张图片

这是迁移学习最典型的使用场景。

如果你数据量不大，从零训练一个较大的 CNN 往往不划算。这时用预训练模型微调，通常效果更稳、收敛更快。

### 场景 2：你先想快速做出一个可用 baseline

很多项目初期，不是为了立刻做到 SOTA，而是先做一个“可用且靠谱”的基线结果。

迁移学习非常适合这个目标。

### 场景 3：算力有限

如果你算力一般，冻结大部分特征层、只训练分类头，通常会比从零训练轻松很多。

---

## 五、流程图 / 结构图（Mermaid）

下面这张图展示了迁移学习的基本流程：

```mermaid
flowchart LR
    A[加载预训练模型主干] --> B[替换最后分类层]
    B --> C[选择是否冻结主干参数]
    C --> D[准备自己的数据集]
    D --> E[训练分类头或微调整个模型]
    E --> F[在验证集上评估效果]
```

这张图里最关键的决策点是：

- 你是只训练分类头
- 还是进一步微调整个模型

这通常取决于：

- 数据量大小
- 任务与预训练任务的相似程度
- 算力预算

---

## 六、在线版：真正加载预训练权重

这是最常见、最标准的迁移学习写法之一。

```python
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# 加载带有预训练权重的 resnet18
# 第一次使用时，torchvision 可能会联网下载权重文件
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# 获取最后一层全连接层的输入特征数
num_features = model.fc.in_features

# 把原本用于 1000 分类的输出层，替换成你自己的类别数
# 这里假设你的任务是 2 分类
model.fc = nn.Linear(num_features, 2)

print(model.fc)
```

这段代码的核心含义是：

- 主干网络参数来自预训练
- 任务相关的最后分类头由你自己重新定义

---

## 七、离线版：只验证结构，不下载权重

如果你当前机器不方便联网，或者你只是想先确认代码结构是否正确，可以这样写：

```python
import torch.nn as nn
from torchvision.models import resnet18

# weights=None 表示不加载预训练参数
# 这样可以在离线环境下先验证结构能否跑通
model = resnet18(weights=None)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

print(model.fc)
```

这段代码的意义是：

- 验证模型结构替换逻辑没写错
- 不需要联网下载权重

但一定要清楚：

> `weights=None` 只是“结构验证”，不是“真正迁移学习完成”。

---

## 八、冻结特征层，只训练分类头

下面演示一个非常常见的做法：

- 冻结主干网络
- 只训练最后的分类层

```python
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# 先把所有参数都冻结
for param in model.parameters():
    param.requires_grad = False

# 再单独解冻最后的分类头
for param in model.fc.parameters():
    param.requires_grad = True
```

这样做之后：

- 大部分主干参数不会更新
- 训练主要集中在新换上的 `fc` 层

对于中小数据集来说，这通常是一个很好的起点。

---

## 九、如何确认冻结是否真的生效

很多人写完冻结代码后，实际上并没有真正检查。

最直接的方法就是打印所有可训练参数：

```python
trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
print(trainable_params)
```

如果你只训练分类头，常见输出会主要集中在：

- `fc.weight`
- `fc.bias`

这才说明冻结逻辑确实生效了。

---

## 十、构建优化器时要注意什么

如果你冻结了大部分参数，优化器最好只接收真正可训练的参数。

```python
import torch.optim as optim

optimizer = optim.Adam(
    (p for p in model.parameters() if p.requires_grad),
    lr=1e-3
)
```

这样做的好处是：

- 优化器只管理需要更新的参数
- 逻辑更清楚
- 更符合“只训练分类头”的真实意图

---

## 十一、一个更完整的迁移学习最小示例

下面给一个更接近真实工程的最小示例，演示：

- 加载预训练模型
- 替换分类头
- 冻结主干
- 构造优化器

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights

# 选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载预训练 resnet18
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# 替换最后分类层，改成 2 分类
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# 冻结主干参数
for param in model.parameters():
    param.requires_grad = False

# 只训练最后分类头
for param in model.fc.parameters():
    param.requires_grad = True

# 把模型迁移到目标设备
model = model.to(device)

# 只把可训练参数交给优化器
optimizer = optim.Adam(
    (p for p in model.parameters() if p.requires_grad),
    lr=1e-3
)

# 交叉熵损失适合多分类 / 二分类 logits 场景
criterion = nn.CrossEntropyLoss()

# 打印可训练参数名，确认冻结是否真的生效
for name, param in model.named_parameters():
    if param.requires_grad:
        print("trainable:", name, param.shape)
```

---

## 十二、运行结果应该怎么看

### 1. 替换后的分类头是否维度正确

如果你的任务是 2 分类，那么最后一层应类似：

- `Linear(in_features=..., out_features=2)`

### 2. 冻结后可训练参数是否只剩分类头

如果打印结果里仍然有大量卷积层参数是可训练的，说明冻结逻辑没写对。

### 3. 离线结构验证和真实迁移学习要分开理解

如果你用的是：

```python
weights=None
```

那你只能说明：

- 结构正确
- 代码不崩

不能说明：

- 迁移学习效果已经验证通过

### 4. 如果训练很快但效果一般，要想想是否需要解冻更多层

只训练分类头是一个稳妥起点，但不是所有任务的最佳终点。

如果任务和 ImageNet 差异较大，有时需要进一步微调更多层。

---

## 十三、本地环境注意事项

### 1. 在线模式可能触发权重下载

第一次使用：

```python
model = resnet18(weights=ResNet18_Weights.DEFAULT)
```

`torchvision` 可能会下载权重到本地缓存。这不是异常，是正常行为。

### 2. 离线环境先用 `weights=None` 做冒烟测试

如果当前环境不方便联网，先验证：

- 模型结构能否正确实例化
- 分类头是否替换成功
- 冻结逻辑是否写对

这是很合理的第一步。

### 3. 不要把“代码能跑”误当成“效果已经验证”

这点我再强调一次，因为非常容易误判。

- 结构能跑，只说明代码层面没崩
- 迁移学习有效，要看你是否真正加载了预训练权重，以及验证集效果是否改善

---

## 十四、常见错误与排查

### 问题 1：忘记替换最后分类层

如果你的任务不是 1000 分类，却没有替换 `model.fc`，那模型输出维度就不对。

### 问题 2：冻结逻辑写了，但优化器仍然接收全部参数

虽然通常不至于完全错误，但逻辑上不够清楚。更推荐只把 `requires_grad=True` 的参数交给优化器。

### 问题 3：把 `weights=None` 当成预训练完成

这只是随机初始化结构，不是真正的迁移学习效果。

### 问题 4：任务和预训练分布差异太大，却仍然只训分类头

如果你的数据和 ImageNet 差异特别大，只训练最后一层可能不够。这时可能要进一步微调更深层参数。

### 问题 5：不检查到底哪些参数在训练

冻结相关 bug 很隐蔽，最直接的办法永远是打印 `named_parameters()` 检查。

---

## 十五、本篇小结

这一篇最重要的认知是：

- 迁移学习适合中小数据集和资源有限场景
- 真正的迁移学习价值来自预训练权重，而不只是模型结构
- 常见做法是替换最后分类头，让模型适配你的类别数
- 冻结主干、只训练分类头，是一个很实用的入门策略
- `weights=None` 适合离线结构验证，但不等于预训练迁移已经生效

如果你把这篇吃透，后面做真实图像分类项目时，速度和起点都会比从零训练轻松很多。

---

## 十六、练习题

### 练习 1：替换成 5 分类头
把 `resnet18` 的最后分类层改成输出 `5` 类，并打印新分类头结构。

### 练习 2：检查冻结是否生效
冻结主干后，打印所有 `requires_grad=True` 的参数名，确认是否只剩 `fc.weight` 和 `fc.bias`。

### 练习 3：分别尝试在线和离线模式
分别运行：

- `weights=ResNet18_Weights.DEFAULT`
- `weights=None`

思考它们在意义上有什么本质区别。

### 练习 4：优化器只接收可训练参数
修改优化器写法，只传入 `requires_grad=True` 的参数，并打印参数数量。

### 练习 5：思考题
为什么说迁移学习的本质不是“复制一个模型结构”，而是“复用已经学到的通用视觉知识”？

试着用你自己的话解释一下。

---

## 下一篇预告

下一篇我们会进入训练提速与资源利用这个非常现实的话题：**GPU、混合精度与训练加速**。

到那时你会开始把注意力从“能训练”转向：

- 如何更高效地利用显卡
- 为什么混合精度能带来速度和显存收益
- 什么情况下加速是有效的，什么情况下只是徒增复杂度
