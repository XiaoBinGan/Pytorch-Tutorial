# EP04｜用 nn.Module 搭建第一个网络

![PyTorch 系列教程](https://pytorch.org/assets/images/pytorch-logo.png)

作者：吴佳浩

撰稿时间：2026-06-03

## 本篇你会学到什么

这一篇承接自动求导机制，开始回答“梯度到底作用在谁身上”这个问题。学完这一篇，你应该能够：

- 理解 `nn.Module` 为什么是 PyTorch 搭建模型的核心基类
- 分清 `__init__()` 和 `forward()` 各自负责什么
- 独立写出一个最基础的全连接神经网络
- 理解模型输入输出 shape 是如何在层之间流动的
- 学会查看模型结构、参数量和可训练参数
- 为后面的训练循环、分类任务、CNN 和迁移学习打下模型定义基础

---

## 一、为什么要学 `nn.Module`

前面你已经学了：

- Tensor 是 PyTorch 的基础数据结构
- autograd 负责自动求导
- Dataset / DataLoader 负责组织数据

但到目前为止，还缺少一个非常关键的环节：

> **模型本身该怎么定义？**

在真实训练里，模型不是一串临时拼接的数学操作，而是一个可以被重复调用、能保存参数、能迁移到 GPU、能参与训练和推理的结构化对象。

这正是 `nn.Module` 的价值。

你可以先把它理解成：

> **PyTorch 里所有神经网络模块的统一父类。**

无论是：

- 一个最简单的线性层网络
- 卷积神经网络 CNN
- Transformer
- 一个由很多子模块组成的大模型

本质上都建立在 `nn.Module` 这套机制之上。

所以这一篇不是在学“某个语法点”，而是在学：

> **PyTorch 中模型究竟是如何被组织起来的。**

---

## 二、核心理论讲解

### 1. 什么是 `nn.Module`

`nn.Module` 是 PyTorch 用来表示“可学习模块”的基类。

它帮你做了很多关键工作：

- 管理模型中的子层
- 自动登记参数
- 支持 `model.parameters()` 遍历参数
- 支持 `model.to(device)` 统一迁移设备
- 支持 `model.train()` / `model.eval()` 切换模式
- 支持保存与加载模型权重

也就是说，如果你只是随手写几个 Tensor 运算，虽然也能得到结果，但它们不会自动拥有这些深度学习训练所需的工程能力。

### 2. `__init__()` 负责什么

在自定义模型里，`__init__()` 的核心职责是：

> **定义模型有哪些层和子模块。**

例如：

- 线性层 `nn.Linear`
- 激活函数 `nn.ReLU`
- 卷积层 `nn.Conv2d`
- Dropout、BatchNorm 等

你可以把 `__init__()` 理解成“把零件装到模型身上”。

### 3. `forward()` 负责什么

`forward()` 的核心职责是：

> **定义输入数据如何流过这些层，最终得到输出。**

也就是在说明：

- 先经过哪一层
- 再经过什么激活函数
- 再进入哪一层
- 最终输出什么结果

你可以把 `forward()` 理解成“规定数据在模型里的行进路线”。

### 4. 为什么平时写 `model(x)`，而不是 `model.forward(x)`

这是一个常见疑问。

虽然你自己定义了 `forward()`，但 PyTorch 推荐你这样调用：

```python
out = model(x)
```

而不是：

```python
out = model.forward(x)
```

原因是 `model(x)` 并不只是简单调用 `forward()`，它还会经过 `nn.Module` 内部的一层封装逻辑，确保：

- hook 机制正常工作
- 某些框架级特性能够挂接
- 模型调用行为保持统一

所以实践里请养成习惯：

- 定义时写 `forward()`
- 调用时写 `model(x)`

### 5. 什么是模型参数自动登记

当你在 `__init__()` 里这样写：

```python
self.fc1 = nn.Linear(10, 32)
```

PyTorch 会自动把这层里的权重和偏置登记到模型参数系统中。

于是你后面就可以直接：

```python
model.parameters()
```

拿到所有可训练参数。

这也是为什么模型、优化器、保存权重这些机制能彼此协同工作的关键。

### 6. 输入输出 shape 为什么这么重要

模型能不能跑通，最常见的不是数学推导错，而是：

- 输入特征维度和线性层不匹配
- batch 维度没处理对
- 输出类别数和任务标签数不一致

所以定义模型时一定要有一个很强的意识：

> **每一层的输入 shape 和输出 shape 到底是什么。**

如果这一点不清楚，后面训练阶段很容易报错。

---

## 三、先建立一个直觉理解

你可以把 `nn.Module` 模型理解成一个“装配好的可学习流水线”。

- `__init__()` 负责把流水线上的设备安装好
- `forward()` 负责规定原料怎么依次流过这些设备
- 参数则是这些设备内部可以被训练调节的旋钮

比如一个两层全连接网络，本质上就是：

1. 输入先经过第一层线性变换
2. 再经过一个非线性激活函数
3. 再经过第二层线性变换
4. 得到最终类别分数

所以模型不是一团神秘黑箱，而是一个有明确数据流路径的计算结构。

---

## 四、真实项目里怎么用

### 场景 1：表格数据分类

假设你有每条样本 10 个特征，想判断样本属于 2 个类别中的哪一个。

这时一个最基础的 `nn.Module` 模型通常会写成：

- 输入层：10 维
- 隐藏层：32 维
- 输出层：2 维

这就是最典型的多层感知机（MLP）入门结构。

### 场景 2：后续换成 CNN 或更复杂模型

虽然后面你会学 CNN、迁移学习甚至更复杂结构，但组织方式不会变：

- 都是继承 `nn.Module`
- 都是在 `__init__()` 里定义层
- 都是在 `forward()` 里定义数据流

也就是说，学会 `nn.Module` 之后，你其实就掌握了 PyTorch 模型构建的统一语法骨架。

### 场景 3：工程里查看模型结构和参数量

真实项目里，你经常会做这些事情：

- 打印模型结构确认层数和顺序
- 统计总参数量
- 统计可训练参数量
- 确认模型是否迁移到 GPU

这些都建立在 `nn.Module` 的标准化封装能力上。

---

## 五、流程图 / 结构图（Mermaid）

下面这张图展示了一个最基础全连接网络的数据流：

```mermaid
flowchart LR
    A[输入 Tensor
shape=(batch, 10)] --> B[Linear 10->32]
    B --> C[ReLU 激活]
    C --> D[Linear 32->2]
    D --> E[输出 logits
shape=(batch, 2)]
```

这张图要你记住两个重点：

1. 每一层都在改变数据表示方式
2. 最终输出的 shape 必须和任务需求对得上

例如二分类里，如果你使用两个输出神经元表示两个类别，那么输出常见 shape 是：

- `(batch_size, 2)`

---

## 六、从零写一个最小可运行示例

下面我们写一个完整但足够简单的 `nn.Module` 模型。

这个模型用来处理：

- 输入特征维度为 10
- 输出类别数为 2

适合理解全连接网络的最基础写法。

```python
import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    """
    一个最简单的两层全连接分类网络。

    输入：
        shape = (batch_size, 10)

    输出：
        shape = (batch_size, 2)
        这里的 2 表示两个类别的原始得分（logits）
    """
    def __init__(self):
        super().__init__()

        # 第一层线性层：把 10 维输入映射到 32 维隐藏表示
        # 可以理解成：先把原始特征投影到一个更高维的表达空间
        self.fc1 = nn.Linear(10, 32)

        # ReLU 激活函数：给模型引入非线性能力
        # 如果没有激活函数，多层线性层堆起来本质上仍然接近线性变换
        self.relu = nn.ReLU()

        # 第二层线性层：把 32 维隐藏表示映射到 2 维输出
        # 输出的 2 表示两个类别的打分
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        # x 的预期 shape 是 (batch_size, 10)
        # 先通过第一层线性层
        x = self.fc1(x)

        # 再经过非线性激活函数
        x = self.relu(x)

        # 最后映射到输出层，得到每个类别的原始分数
        x = self.fc2(x)

        # 返回 logits，而不是最终类别 id
        return x


# 实例化模型
model = SimpleNet()

# 打印模型结构，直观查看层级关系
print(model)

# 构造一批假的输入数据
# batch_size = 8，每条样本有 10 个特征
x = torch.randn(8, 10)

# 推荐使用 model(x) 的方式调用，而不是直接调 forward()
out = model(x)

print("输入 x.shape =", x.shape)
print("输出 out.shape =", out.shape)
```

---

## 七、进一步看模型参数和参数量

定义好模型之后，一个很常见的工程动作就是查看参数量。

这能帮助你：

- 判断模型规模大不大
- 粗略估计训练成本
- 确认哪些参数可训练

下面这段代码演示如何统计总参数量和可训练参数量：

```python
import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = SimpleNet()

# 统计模型总参数量
# numel() 表示张量中元素总数
total_params = sum(p.numel() for p in model.parameters())

# 统计可训练参数量
# requires_grad=True 的参数才会在训练中被优化器更新
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("总参数量 =", total_params)
print("可训练参数量 =", trainable_params)

# 逐个打印参数名称和 shape，帮助理解模型内部结构
for name, param in model.named_parameters():
    print(f"参数名: {name}, shape: {param.shape}, requires_grad: {param.requires_grad}")
```

---

## 八、拆解代码执行过程

### 1. `super().__init__()` 不能忘

这是继承 `nn.Module` 时的标准写法。

它的作用是先把父类的初始化逻辑跑起来，让当前模型具备 `nn.Module` 的基础能力，比如：

- 参数登记
- 子模块管理
- train / eval 切换
- state_dict 管理

如果忘了写，模型行为通常会不正常。

### 2. `self.fc1 = nn.Linear(...)` 不只是普通赋值

表面上看，这像是把一个对象挂到实例属性上；但对 `nn.Module` 来说，这一步还有额外意义：

- 它会自动把 `fc1` 注册成子模块
- `fc1` 里的权重和偏置也会被纳入模型参数系统

所以你后面才能通过 `model.parameters()` 拿到它们。

### 3. `forward()` 描述的是数据流，而不是训练逻辑

这是一个容易混淆的点。

`forward()` 只负责：

- 输入进来后怎么经过各层
- 最终得到什么输出

它通常不负责：

- 计算损失
- 调优化器
- backward
- step

这些训练逻辑应该放在训练循环里，而不是塞进模型定义里。

### 4. 输出 logits，不等于最终类别

`forward()` 返回的通常是 logits，也就是每个类别的原始分数。

如果你想得到最终类别，一般是后续再做：

```python
preds = torch.argmax(out, dim=1)
```

而不是在模型内部直接返回类别 id。

这是因为：

- 训练时损失函数通常更需要 logits
- 推理时你才更关心最终类别

---

## 九、运行结果应该怎么看

运行上面的代码后，重点看以下几个地方。

### 1. 模型结构是否和预期一致

打印 `model` 后，应能看到类似：

- 第一层 `Linear(10 -> 32)`
- 一个 `ReLU`
- 第二层 `Linear(32 -> 2)`

如果层顺序和你想的不一样，先不要急着训练，先把结构理顺。

### 2. 输入输出 shape 是否正确

重点看：

- `x.shape == (8, 10)`
- `out.shape == (8, 2)`

这说明：

- 输入批次大小为 8
- 每条输入有 10 个特征
- 输出对应 2 个类别得分

### 3. 参数量是否合理

如果模型很小，但你一统计参数量却特别大，通常说明：

- 某层维度设得太大
- 输入特征展平方式有问题
- 模型结构比你以为的复杂

参数量统计是非常实用的第一道工程 sanity check。

---

## 十、常见错误与排查

### 问题 1：输入维度和线性层不匹配

典型报错：

```python
RuntimeError: mat1 and mat2 shapes cannot be multiplied
```

这通常说明：

- 输入 `x.shape[1]` 和 `nn.Linear(in_features, ...)` 的 `in_features` 对不上

排查建议：

- 打印 `x.shape`
- 确认 `fc1` 的输入维度定义

### 问题 2：在 `forward()` 里混入不该有的训练逻辑

比如有人把：

- loss 计算
- optimizer.step()
- backward

都写进 `forward()` 里。

这样会让模型定义和训练逻辑高度耦合，非常不利于维护。

正确分工应该是：

- `forward()`：只管前向传播
- 训练循环：负责 loss、backward、step

### 问题 3：直接写 `model.forward(x)`

虽然某些情况下也能工作，但不推荐。

请统一写：

```python
out = model(x)
```

### 问题 4：误把输出当概率

模型输出的 logits 不是概率。它只是原始得分。

如果你需要概率，通常要在合适场景下再做 softmax；如果是训练分类模型，很多损失函数会内部处理相关逻辑，不要提前乱加。

### 问题 5：不知道哪些参数在训练

这时就打印：

```python
for name, param in model.named_parameters():
    print(name, param.shape, param.requires_grad)
```

这是最直接的排查方式。

---

## 十一、本篇小结

这一篇最重要的认知是：

- `nn.Module` 是 PyTorch 构建模型的统一基类
- `__init__()` 负责定义层和子模块
- `forward()` 负责定义数据流路径
- 模型输出通常是 logits，而不是最终类别
- 查看模型结构、shape 和参数量，是训练前必须做的基础检查

你只要把 `nn.Module` 这套骨架吃透，后面无论是写更深的 MLP、CNN，还是做迁移学习，本质上都只是“在同一套框架里换层和换数据流”。

---

## 十二、练习题

### 练习 1：修改隐藏层大小
把隐藏层从 `32` 改成 `64`，重新打印模型结构和参数量，观察变化。

### 练习 2：修改输入维度
把模型第一层改成接收 `20` 维输入，然后构造匹配的假数据，确认模型仍能正常前向传播。

### 练习 3：修改输出类别数
把输出类别从 `2` 改成 `5`，并观察输出 `out.shape` 发生了什么变化。

### 练习 4：打印模型所有参数
使用 `named_parameters()` 打印：

- 参数名
- 参数 shape
- 是否可训练

试着对应到模型中的每一层。

### 练习 5：思考题
为什么说：

- `__init__()` 是“安装零件”
- `forward()` 是“规定数据流路线”

如果把两者职责混在一起，会给模型维护带来什么问题？

---

## 下一篇预告

下一篇我们会继续把模型真正接上“可训练的数据流”：**Dataset 与 DataLoader 数据管道**。

你会开始看到：

- 为什么真实训练不能手搓几个 Tensor 就长期维持
- 单条样本和 batch 数据到底该怎么分层组织
- 数据读取、批量组装和训练循环之间应该如何解耦
