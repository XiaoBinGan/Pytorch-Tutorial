# EP13｜推理脚本与本地部署


<p align="center">
 <img src="https://pytorch.org/assets/images/pytorch-logo.png" width="400" alt="PyTorch">
</p>

<h1 align="center">PyTorch 你不学</h1>
<h2 align="center">EP13 推理脚本与本地部署</h2>
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

这一篇承接前面的训练、保存与加速内容，开始把模型真正推进到“能被使用”的阶段。学完这一篇，你应该能够：

- 理解“训练”和“推理”在目标与代码结构上的区别
- 学会把训练好的模型整理成一个真正可复用的本地推理脚本
- 知道“本地部署”不等于一定上云，而是先把预测能力稳定封装出来
- 学会对单张图片做预处理、加载模型权重并输出预测结果
- 学会把推理代码拆成：模型加载、图片预处理、预测执行 三个清晰模块
- 为后续做命令行工具、本地 Web 服务、局域网服务打下基础

---

## 一、为什么训练完成后还要单独写推理脚本

很多初学者在训练完模型后，容易停在“模型已经训练好了”这个阶段。但真实项目真正能落地，还差关键一步：

> **把训练好的模型整理成可重复调用的推理能力。**

训练脚本通常关注的是：

- 读训练集
- 计算 loss
- backward
- 优化器更新
- 保存 checkpoint

而推理脚本关注的是：

- 加载训练好的权重
- 接收一个真实输入样本
- 做和训练时一致的预处理
- 输出预测结果

这两类脚本目标不同，结构也不同。

所以“本地部署”的第一步，往往不是上云、上服务集群，而是：

> **先把一个干净、稳定、可重复执行的推理脚本写好。**

---

## 二、核心理论讲解

### 1. 什么是推理（Inference）

推理可以简单理解成：

> **使用训练好的模型，对新输入样本做预测。**

它和训练的核心区别在于：

- 不更新参数
- 不需要 backward
- 不需要 optimizer.step()
- 更关注输入处理和输出解释

### 2. 为什么推理时要 `model.eval()`

推理时通常要写：

```python
model.eval()
```

这是因为有些层在训练和推理阶段行为不一样，例如：

- Dropout
- BatchNorm

即使你当前的小模型里没有这些层，也应该养成习惯。因为后面做更真实模型时，这一步非常关键。

### 3. 为什么推理时要 `torch.no_grad()`

推理阶段通常不需要计算梯度，所以应该关闭梯度跟踪：

```python
with torch.no_grad():
    ...
```

这样做的好处是：

- 更省内存
- 更省计算
- 逻辑更符合“只做预测”的目标

### 4. 为什么预处理必须和训练阶段保持一致

这是推理脚本最容易被忽略、但非常关键的一点。

如果训练时图片经过了：

- 灰度化
- resize 到 `28 × 28`
- `ToTensor()`

那推理时也必须做相同或兼容的处理。

否则模型看到的数据分布就和训练时不一样，预测结果很可能不可靠。

### 5. 为什么“部署”不等于一定上云

很多人一听“部署”就想到：

- 云服务
- Kubernetes
- Docker 集群
- 大型推理服务框架

这些当然属于部署，但在当前这套教程阶段，更现实的“本地部署”通常包括：

- 一个可以直接运行的 Python 推理脚本
- 一个命令行工具
- 一个只在本机或局域网里运行的本地 API 服务

所以部署的第一层目标是：

> **让模型预测能力脱离训练脚本，成为一个可独立调用的能力模块。**

---

## 三、先建立一个直觉理解

你可以把训练和推理想成两个不同岗位。

- **训练脚本**像老师：看大量带答案的数据、不断纠错、调整模型参数
- **推理脚本**像考官：拿一份新样本进来，直接给出判断结果

老师和考官都依赖同一个模型，但工作内容并不一样。

所以推理脚本不是把训练代码随便删几行，而是要重新整理成更适合“接收输入并给出结果”的结构。

---

## 四、真实项目里怎么用

### 场景 1：本地快速测一张图片

这是最常见的起点。

你训练好一个模型后，希望直接验证：

- 某张本地图片模型会预测成什么

这时一个简单的 `infer.py` 就非常有用。

### 场景 2：把推理能力封装成命令行工具

如果你后面会频繁测试文件，那么比起每次手改路径，一个支持：

```powershell
python infer.py sample.png
```

的脚本会高效得多。

### 场景 3：为后续本地 API 服务做准备

本地 Flask / FastAPI 服务，其本质也不过是：

- 接收输入
- 调用推理逻辑
- 返回预测结果

所以先把单机推理脚本整理干净，后面再包成接口服务会容易很多。

---

## 五、流程图 / 结构图（Mermaid）

下面这张图展示了本地推理脚本的基本流程：

```mermaid
flowchart LR
    A[输入图片路径] --> B[读取图片]
    B --> C[按训练时方式做预处理]
    C --> D[加载模型权重]
    D --> E[model.eval + no_grad 推理]
    E --> F[得到 logits]
    F --> G[argmax 取预测类别]
    G --> H[输出预测结果]
```

这张图提醒你：

- 推理不是只有 `model(x)` 一步
- 输入预处理和模型加载同样关键

---

## 六、一个最小可跑的推理脚本

下面我们先写一个完整、最小可运行的推理示例。

这个版本的目标是：

- 加载已经训练好的 MNIST 模型权重
- 读取一张本地图片
- 做预处理
- 输出预测类别

```python
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class MNISTNet(nn.Module):
    """
    和训练阶段保持一致的模型结构。
    如果训练时你改过层数或隐藏维度，这里也必须同步修改。
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


# 1. 创建模型实例
model = MNISTNet()

# 2. 加载训练好的权重
# map_location='cpu' 让脚本在没有 GPU 的机器上也能正常运行
state_dict = torch.load("checkpoints/mnist_model.pt", map_location="cpu")
model.load_state_dict(state_dict)

# 3. 切换到评估模式
model.eval()

# 4. 定义与训练阶段兼容的图片预处理流程
transform = transforms.Compose([
    transforms.Grayscale(),      # 转成单通道灰度图
    transforms.Resize((28, 28)), # 调整到训练时模型要求的尺寸
    transforms.ToTensor(),       # 转成 Tensor，并把像素归一化到 [0, 1]
])

# 5. 读取图片并做预处理
img = Image.open("sample.png")
x = transform(img).unsqueeze(0)

# unsqueeze(0) 的作用：
# 把单张图片从 (1, 28, 28) 变成 (1, 1, 28, 28)
# 其中最前面的 1 表示 batch size = 1

# 6. 推理阶段关闭梯度计算
with torch.no_grad():
    logits = model(x)
    pred = torch.argmax(logits, dim=1).item()

print("预测结果:", pred)
```

---

## 七、为什么这段推理代码能工作

把上面代码拆开理解，会更清楚。

### 1. 模型结构必须一致

推理脚本里重新定义 `MNISTNet`，不是多余，而是必须。

因为你加载的是参数字典，不是自动“长出来”的模型结构。

### 2. 图片要变成 batch 格式

训练时模型见到的通常不是单张样本，而是 batch。

即使现在只推理 1 张图片，输入格式仍然最好保持 batch 风格：

- `(1, 1, 28, 28)`

这就是 `unsqueeze(0)` 的意义。

### 3. 输出的通常是 logits，不是最终类别

`model(x)` 返回的通常是各类别的原始分数。

所以真正输出类别时，常用：

```python
torch.argmax(logits, dim=1)
```

取分数最高的类别。

---

## 八、把它整理成真正能复用的 `infer.py`

如果你后面会频繁测图片，建议把推理逻辑整理成可复用函数，并支持命令行参数。

这样它才更像一个真正能拿来用的小工具。

```python
import sys
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


def load_model(weight_path="checkpoints/mnist_model.pt"):
    """
    加载训练好的模型权重，并返回 eval 模式下的模型。
    """
    model = MNISTNet()
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_image(image_path):
    """
    读取图片并做与训练阶段一致的预处理。
    返回 shape 为 (1, 1, 28, 28) 的输入张量。
    """
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path)
    return transform(image).unsqueeze(0)


def predict(model, x):
    """
    对输入张量执行推理，并返回预测类别 id。
    """
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
    return pred


if __name__ == "__main__":
    # 从命令行接收图片路径，例如：python infer.py sample.png
    image_path = sys.argv[1]

    model = load_model()
    x = load_image(image_path)
    pred = predict(model, x)

    print("预测结果:", pred)
```

执行方式：

```powershell
python infer.py sample.png
```

这段代码的价值在于：

- 结构清楚
- 后续容易维护
- 方便继续扩展为更多输入方式或服务接口

---

## 九、本地目录建议

建议目录可以整理成：

```text
G:\PyTorch-教程\
├─ checkpoints\
│  └─ mnist_model.pt
├─ samples\
│  └─ sample.png
└─ infer.py
```

这种结构的好处是：

- 权重文件单独放
- 样本文件单独放
- 推理脚本单独放
- 后面要升级成完整项目时更容易管理

---

## 十、如果你想继续往“本地部署”走一步

在当前教程阶段，本地部署最现实的扩展方向通常有三种：

### 1. 命令行工具化

这是最轻量也最推荐的第一步。

例如：

```powershell
python infer.py samples\sample.png
```

### 2. 批量推理脚本

如果你要一次预测多个文件，可以把 `infer.py` 扩展成：

- 传入目录路径
- 批量遍历图片
- 输出每张结果

### 3. 本地 Web API

例如基于 Flask / FastAPI：

- 接收图片上传
- 调用模型推理
- 返回 JSON 结果

但建议顺序是：

> **先把单文件推理脚本跑稳，再上本地接口服务。**

---

## 十一、运行结果应该怎么看

### 1. 检查权重是否成功加载

如果模型加载失败，优先排查：

- 权重路径对不对
- 模型结构是否和训练时一致

### 2. 检查输入 shape 是否正确

推理前可以临时打印：

```python
print(x.shape)
```

你通常希望看到：

- `(1, 1, 28, 28)`

### 3. 检查输出是否是单个类别 id

最终 `pred` 应该是一个整数类别，例如：

- `0`
- `7`
- `9`

### 4. 如果结果明显不合理，先怀疑预处理

推理结果不对，不一定是模型训练失败。很多时候更可能是：

- 灰度化没做对
- resize 不一致
- 输入尺寸不匹配
- 图片内容分布和训练集差异太大

---

## 十二、常见错误与排查

### 问题 1：训练时结构和推理时结构不一致

这是最常见的坑之一。

如果你训练时改过：

- 隐藏层宽度
- 层数
- 输出类别数

那推理脚本里的模型类也必须同步修改。

### 问题 2：图片尺寸不对

MNIST 示例默认需要：

- 单通道灰度图
- `28 × 28`

如果你跳过了这些预处理，结果很可能不可靠。

### 问题 3：没加 batch 维度

单张图片经过 `ToTensor()` 后通常 shape 是：

- `(1, 28, 28)`

但模型往往期望 batch 格式输入，所以还要：

- `unsqueeze(0)`

### 问题 4：忘记 `model.eval()` 或 `torch.no_grad()`

小模型里有时不容易立刻看出问题，但这仍然是不规范的推理写法。

### 问题 5：把“本地部署”想得过于复杂

这一步最重要的不是上多大架构，而是：

- 先让模型预测能力稳定可复用
- 先把输入、模型加载、输出这条链路跑顺

---

## 十三、本篇小结

这一篇最重要的认知是：

- 训练和推理是两个不同目标的脚本阶段
- 推理脚本的重点是：加载模型、统一预处理、执行预测、输出结果
- `model.eval()` 和 `torch.no_grad()` 是推理阶段的标准写法
- 本地部署的第一步，往往不是上云，而是写一个稳定可复用的 `infer.py`
- 预处理必须尽量和训练阶段保持一致，否则模型效果可能会失真

如果你把这一篇真正掌握了，你的模型就开始从“会训练”走向“能被真正使用”。

---

## 十四、练习题

### 练习 1：把推理脚本改成接收权重路径
除了图片路径，再让脚本支持命令行传入模型权重路径。

### 练习 2：打印中间 shape
在推理脚本里打印：

- 预处理后图片 shape
- 模型输出 logits.shape

确认整个推理链路符合预期。

### 练习 3：批量预测一个目录
把 `infer.py` 扩展成：

- 输入一个目录
- 遍历其中所有图片
- 逐张打印预测结果

### 练习 4：增加更友好的输出
除了输出类别 id，再尝试输出：

- 文件名
- 预测类别
- 原始 logits（可选）

### 练习 5：思考题
为什么说“部署”的第一步往往不是上云，而是先把推理脚本整理干净、可复用、可稳定执行？

---

## 下一篇预告

下一篇我们会把整套教程里最容易让人卡住、但也最能拉开工程能力差距的一环单独拎出来：**常见报错排查与调试技巧**。

你会开始建立：

- 遇到报错先分类再定位的习惯
- 用 shape、dtype、device、grad 做快速排查的意识
- 从“看报错就慌”走向“能系统缩小问题范围”的能力
