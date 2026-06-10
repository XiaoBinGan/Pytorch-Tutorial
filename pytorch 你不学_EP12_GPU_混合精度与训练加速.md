# EP12｜GPU、混合精度与训练加速


<p align="center">
 <img src="https://pytorch.org/assets/images/pytorch-logo.png" width="400" alt="PyTorch">
</p>

<h1 align="center">PyTorch 你不学</h1>
<h2 align="center">EP12 GPU、混合精度与训练加速</h2>
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

这一篇承接迁移学习与图像训练实践，开始把关注点从“能训练”转向“如何更高效地训练”。学完这一篇，你应该能够：

- 理解为什么深度学习训练通常优先使用 GPU
- 学会在 PyTorch 中正确管理 `device`
- 知道为什么模型、输入、标签必须在同一设备上
- 初步理解混合精度训练（AMP）是什么、为什么能加速
- 学会使用 `autocast` 和 `GradScaler` 编写基础 AMP 训练代码
- 明白“加速训练”不只是换设备，还包括减少无效错误和合理组织训练代码
- 在没有 GPU 的本地环境下，也能正确理解并阅读相关代码结构

---

## 一、为什么训练深度学习通常要用 GPU

如果你前面已经在 CPU 上跑过一些基础示例，应该会有直觉：

- 小实验还可以
- 模型稍大一些、数据稍多一些，速度就明显慢下来

这不是 PyTorch 写得慢，而是因为深度学习训练本质上包含大量：

- 矩阵乘法
- 卷积运算
- 并行张量计算

而 GPU 在这类高并行数值计算上通常比 CPU 更有优势。

所以在真实项目里，你经常会看到：

- 数据准备在 CPU
- 训练核心计算放到 GPU

这也是为什么几乎所有深度学习教程后期都会讲 `cuda`、设备迁移和训练加速。

---

## 二、核心理论讲解

### 1. 什么是 `device`

在 PyTorch 里，`device` 表示 Tensor 或模型参数当前所在的计算设备。

常见值包括：

- `cpu`
- `cuda`
- `cuda:0`
- `cuda:1`

它决定了计算真正在哪块硬件上执行。

### 2. 为什么模型和数据必须在同一设备上

这是入门到进阶过程中最常见的错误之一。

如果：

- 模型参数在 GPU
- 输入数据还在 CPU

那么前向传播通常会直接报错。

因为一次计算中的参与者，通常必须在同一个设备空间里。

所以你会反复看到这种代码：

```python
model = model.to(device)
x = x.to(device)
y = y.to(device)
```

这不是形式主义，而是 PyTorch 训练代码能否跑通的基本前提。

### 3. 什么是混合精度训练（AMP）

混合精度训练可以先粗略理解成：

> **训练时不是所有计算都强制用同一种高精度数值格式，而是让一部分计算使用更低精度，以换取更快速度和更低显存占用。**

常见直觉是：

- 一部分运算用更低精度（例如 float16）会更快
- 但纯低精度训练有时容易数值不稳定
- 所以需要“混合”使用不同精度，并配套数值稳定手段

这就是 AMP（Automatic Mixed Precision）的核心出发点。

### 4. `autocast` 在做什么

`autocast` 可以理解成：

> **在某个上下文中，让 PyTorch 自动决定哪些运算适合用更低精度执行。**

你不需要手动把每一层都改成半精度，而是告诉系统：

- 这段前向计算请按自动混合精度策略执行

这样代码会比手动管理 dtype 简洁很多。

### 5. `GradScaler` 在做什么

低精度训练的一个风险是：

- 梯度值太小，可能在低精度表示中下溢
- 导致训练不稳定甚至梯度接近消失

`GradScaler` 的作用可以先直观理解成：

> **在反向传播前对 loss 做缩放，让梯度处于更安全的数值范围，再在更新时正确处理缩放。**

所以它不是“另一个优化器”，而是 AMP 训练中帮助稳定梯度数值的辅助工具。

### 6. 训练加速不等于“盲目上 GPU”

很多人以为训练加速就是一句：

- `device = 'cuda'`

其实远远不止。

训练加速至少还包括：

- 正确的设备管理，避免来回拷贝和报错
- 合理的 batch size
- 混合精度的使用
- 稳定的数据加载
- 让训练脚本结构清晰，减少反复 debug 的时间损耗

也就是说，**工程正确性本身就是一种加速**。

### 7. 什么时候该上 AMP，什么时候别急着上

AMP 不是万能加速按钮，它是有适用门槛的。这里给一个非常实用的判断框架：

**通常值得一试的场景：**

- 你已经在用 GPU（特别是较新的 NVIDIA 卡，如 RTX 20xx、30xx、40xx 系列）
- 模型不算太小，batch 不算太小
- 显存紧张，想通过 FP16 降低显存占用
- 训练时间很长，任何一点加速都有意义

**不建议急着上的场景：**

- 你还在 CPU 环境（AMP 依赖于 GPU 的 Tensor Core）
- 模型极小、batch 极小（加速收益微乎其微，还可能引入数值不稳定）
- 训练本身还经常因 shape、dtype、device 问题报错（先把基础跑稳再谈加速）
- 你的模型对数值精度特别敏感（少数特殊场景下 FP16 可能导致训练不稳定）

**一个很稳的工程原则：**

> 先把 CPU 兼容的训练脚本写对 → 再加 GPU → 再加 AMP。

顺序一旦反过来，你会在一团错误里分不清到底是逻辑问题还是精度问题。


---

## 三、先建立一个直觉理解

你可以把 CPU 和 GPU 理解成两种不同风格的工作方式。

- CPU 更像“通用型熟练工”，擅长处理复杂控制逻辑
- GPU 更像“超大规模流水线”，擅长大量相似计算同时展开

而深度学习里的矩阵乘法、卷积、批量张量运算，恰好特别适合这种“大量并行流水线”模式。

混合精度则像是在这条流水线上进一步做“轻量化处理”：

- 不是所有环节都追求最高精度
- 而是在尽量不牺牲训练质量的前提下，提高吞吐和效率

---

## 四、真实项目里怎么用

### 场景 1：本机有 GPU，想明显缩短训练时间

这是最典型场景。

你会做的第一件事通常是：

- 把模型和数据迁移到 CUDA

接着，如果硬件和环境支持，再尝试：

- 使用 AMP 混合精度训练

### 场景 2：模型稍大，显存开始吃紧

混合精度除了可能提速，还经常有一个现实好处：

- 降低显存占用

这会让你更容易：

- 跑更大的 batch
- 跑更复杂的模型
- 减少 OOM 风险

### 场景 3：你当前没有 GPU，但想先把训练代码结构写对

这也非常现实。

即使本机当前没有 GPU，你仍然应该先把训练脚本写成：

- 自动检测 device
- 自动兼容 CPU / GPU
- AMP 可开可关

这样以后环境一换，代码就能更平滑地迁移过去。

---

## 五、流程图 / 结构图（Mermaid）

下面这张图展示了使用 GPU 和 AMP 的训练流程：

```mermaid
flowchart LR
    A[选择 device CPU 或 CUDA] --> B[模型迁移到 device]
    B --> C[batch 数据迁移到 device]
    C --> D[autocast 自动混合精度前向传播]
    D --> E[计算 loss]
    E --> F[GradScaler 缩放 loss]
    F --> G[backward 反向传播]
    G --> H[scaler.step 更新优化器]
    H --> I[scaler.update 更新缩放状态]
```

这张图最关键的是告诉你：

- AMP 并不是凭空替代训练循环
- 它是嵌入到正常训练骨架中的一层加速与数值稳定机制

---

## 六、最基础的设备选择写法

任何想兼容 CPU / GPU 的脚本，几乎都应该先有这样一段：

```python
import torch

# 如果当前环境有 CUDA，就优先使用 GPU；否则自动退回 CPU
# 这样你的代码既能在高性能环境跑，也能在普通本机环境调试
device = "cuda" if torch.cuda.is_available() else "cpu"
print("当前设备:", device)
```

这段代码看似简单，但它实际上承担了整个训练脚本的设备入口控制。

---

## 七、模型和数据必须同设备

下面这个例子展示最基本也最常见的设备迁移写法。

```python
import torch
import torch.nn as nn


model = nn.Linear(10, 2)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 假设 loader 会不断返回一批输入和标签
for x, y in loader:
    # 把输入和标签迁移到和模型相同的设备上
    x = x.to(device)
    y = y.to(device)

    # 后续前向传播和 loss 计算才不会出现设备不一致问题
    logits = model(x)
```

这段代码的核心不是 API 本身，而是建立一个习惯：

- **device 统一管理**
- **模型和 batch 同步迁移**

---

## 八、基础 AMP 混合精度训练示例

下面给出一个最基础、可读性比较强的 AMP 训练模板。

```python
import torch

# 统一设备管理
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# GradScaler 主要在 CUDA 混合精度训练时起作用
# 如果当前没有 GPU，则 enabled=False，它会自动退化成普通训练逻辑
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

for x, y in loader:
    # 把当前 batch 放到正确设备上
    x = x.to(device)
    y = y.to(device)

    # 清空旧梯度
    optimizer.zero_grad()

    # autocast 上下文中，PyTorch 会自动选择合适的精度执行部分运算
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        logits = model(x)
        loss = criterion(logits, y)

    # 对 loss 进行缩放后再反向传播，帮助提高低精度训练的数值稳定性
    scaler.scale(loss).backward()

    # 使用 scaler 接管优化器更新步骤
    scaler.step(optimizer)

    # 更新 scaler 的内部状态
    scaler.update()
```

这段代码要你重点理解三件事：

1. `autocast` 主要包裹前向传播和 loss 计算
2. `scaler.scale(loss).backward()` 不是普通 `loss.backward()` 的简单替换，而是带数值稳定处理
3. 没有 GPU 时，这套结构依然可以保留，只是相关功能会自动退化

---

## 九、把 AMP 放进一个更完整的训练循环

下面给出一个更接近真实项目的训练函数模板。

```python
import torch


def train_one_epoch_amp(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if batch_idx < 2:
            print(
                f"batch={batch_idx+1}, x.device={x.device}, "
                f"logits.shape={logits.shape}, loss={loss.item():.4f}"
            )

    return total_loss / len(loader)
```

这个模板的意义在于：

- 结构清晰
- 支持 CPU / GPU 兼容
- 支持 AMP 自动启停
- 更适合你后面收进训练脚手架中复用

---

## 十、如果本机没有 GPU，该怎么理解这一节

如果你当前环境没有 GPU，不需要焦虑，也完全没必要“硬上”。

这节你最需要学会的是：

- `device` 的统一管理思想
- 模型 / 数据同设备的要求
- AMP 代码结构长什么样
- 未来一旦切到 GPU 环境，代码应该怎么迁移

也就是说：

> **没有 GPU，不妨碍你理解训练加速的工程写法。**

CPU 环境依然可以把：

- 训练主逻辑
- checkpoint
- 推理脚本
- 调试流程

这些核心能力全部学扎实。

---

## 十一、运行结果应该怎么看

### 1. 先看当前 device 是否符合预期

如果本机有 GPU，你应该看到：

- `当前设备: cuda`

如果没有 GPU，看到 CPU 是正常的。

### 2. 检查模型和输入是否真的在同一设备上

你可以临时打印：

```python
print(x.device)
print(next(model.parameters()).device)
```

如果二者不一致，优先先修这个问题。

### 3. AMP 训练是否能正常完成一个 batch

先不要急着追求提速效果，先确保：

- 不报错
- loss 能正常算出来
- backward 和 optimizer step 能走完

### 4. 如果使用 GPU，观察显存和训练速度变化

这一步才是进阶优化。

先跑通，再看：

- batch size 能不能适当增大
- 速度有没有改善
- 显存压力是否下降

---

## 十二、常见错误与排查

### 问题 1：模型和数据不在同一设备

这是最经典错误。

常见报错类似：

- `Expected all tensors to be on the same device`

解决方法通常不是改模型结构，而是统一设备管理。

### 问题 2：只迁移了输入，忘了迁移标签

尤其在分类任务里，loss 计算时标签也经常需要和 logits 位于同一设备。

### 问题 3：没有 GPU 却强行写死 CUDA

例如直接写：

```python
device = "cuda"
```

这样在无 GPU 环境下就会直接失败。更稳妥的做法是始终用自动检测。

### 问题 4：把 AMP 当成“万能提速按钮”

AMP 很有价值，但它不是在所有环境里都一定神奇生效。是否收益明显，还和：

- 硬件
- 模型类型
- batch size
- 数据加载瓶颈

都有关系。

### 问题 5：还没跑通基础训练，就急着叠加太多优化

很现实的一点是：

- 先跑通普通训练
- 再加 GPU
- 再加 AMP

通常比一上来把所有东西堆满更稳。

---

## 十三、本篇小结

这一篇最重要的认知是：

- GPU 是深度学习训练常见的主力设备
- `device` 管理是训练脚本的基础工程能力
- 模型、输入、标签必须在同一设备上
- 混合精度训练通过 `autocast` 和 `GradScaler` 在速度、显存和稳定性之间做平衡
- 即使当前没有 GPU，也应该先把兼容 CPU / GPU 的训练结构写对

如果你真正理解了这一篇，后面你就不再只是“把代码搬到 GPU”，而是开始真正理解训练加速的工程组织方式。

---

## 十四、练习题

### 练习 1：打印设备信息
写一段代码打印：

- `torch.cuda.is_available()`
- 当前 `device`
- 模型参数所在设备

### 练习 2：把前面的训练脚本改成 CPU / GPU 兼容版
确保：

- 模型迁移到 `device`
- `x` 和 `y` 都迁移到 `device`
- 无 GPU 时也能正常运行

### 练习 3：给训练循环加上 AMP 结构
在你现有训练代码中加入：

- `autocast`
- `GradScaler`

先保证逻辑能跑通。

### 练习 4：故意制造一次设备错误再修复
尝试故意不把标签迁移到 `device`，观察会出现什么问题，然后再修复。

### 练习 5：思考题
为什么说“训练加速”不只是换成 GPU，而是设备管理、数值稳定、训练结构组织一起构成的工程能力？

---

## 下一篇预告

下一篇我们会把“训练好的模型”真正往可使用方向推进，进入 **推理脚本与本地部署**。

你会看到：

- 为什么训练完成不等于项目就能落地
- 推理脚本和训练脚本在目标与结构上有什么本质差别
- 一个模型怎样从“会训练”走到“可重复调用、可本地使用”
