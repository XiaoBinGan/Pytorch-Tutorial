# EP05｜Dataset 与 DataLoader 数据管道

![PyTorch 系列教程](https://pytorch.org/assets/images/pytorch-logo.png)

作者：吴佳浩

撰稿时间：2026-06-03

## 本篇你会学到什么

这一篇承接模型定义，开始把“模型”真正接到“数据流”上。学完这一篇，你应该能够：

- 理解 `Dataset` 和 `DataLoader` 的分工
- 知道为什么真实训练不能靠手搓 Tensor 一把塞进模型
- 会写一个自定义 `Dataset`
- 会用 `DataLoader` 做批量读取、打乱数据和迭代训练
- 明白数据管道为什么是训练系统稳定性的关键一环
- 能排查批大小、返回值格式、Windows 多进程加载等常见问题

---

## 一、为什么要学数据管道

刚入门时，你可能经常这么写：

```python
x = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))
```

这样当然能做小实验，但一旦进入真实项目，很快就不够用了。

因为真实数据往往来自：

- 图片文件夹
- CSV 表格
- JSON / TXT 文本
- 音频文件
- 数据库存储
- 多个目录和标签文件的组合

这时候你需要解决的问题就不再只是“有数据”，而是：

- 如何按索引稳定拿到一条样本
- 如何把样本和标签对应起来
- 如何在训练时一批一批读取
- 如何打乱顺序避免训练偏差
- 如何让训练循环拿到统一格式的数据

这就是 `Dataset` 和 `DataLoader` 存在的原因。

可以先记一句最核心的话：

> **Dataset 负责定义“单条样本怎么取”，DataLoader 负责定义“多条样本怎么送”。**

---

## 二、核心理论讲解

### 1. 什么是 Dataset

`Dataset` 可以理解成：

> **一个按索引访问样本的数据集合接口。**

它最核心的两个方法是：

- `__len__()`：告诉你这个数据集有多大
- `__getitem__(idx)`：告诉你怎么取出第 `idx` 条样本

也就是说，`Dataset` 负责回答：

- 一共有多少条数据？
- 第 17 条数据长什么样？
- 第 86 条数据的标签是什么？

它更像是“数据定义层”。

### 2. 什么是 DataLoader

`DataLoader` 可以理解成：

> **一个把 Dataset 包装成批量迭代器的工具。**

它会在 `Dataset` 的基础上，帮你做这些事：

- 按 batch 组织数据
- 是否打乱顺序
- 每轮训练怎么遍历全量样本
- （可选）多进程并行加载

所以 `DataLoader` 更像是“数据调度层”。

### 3. Dataset 和 DataLoader 的关系

这两者经常一起出现，但职责完全不同：

- `Dataset` 决定“单个样本是什么”
- `DataLoader` 决定“这些样本如何成批流动起来”

如果把训练看成工厂流水线：

- Dataset 是仓库管理员，知道每个零件放哪
- DataLoader 是运输系统，负责每次拉一车零件到生产线

### 4. 为什么训练一定要用 batch

理论上你也可以一次把所有样本都送进模型，但在真实训练里通常并不合适，原因包括：

- 内存 / 显存占用太大
- 参数更新频率太低
- 不利于训练效率和泛化表现

所以绝大多数训练任务都采用 mini-batch 方式：

- 每次取一小批数据
- 做一次前向传播
- 计算一次损失
- 更新一次参数

这也是 DataLoader 最重要的用途之一。

### 5. 为什么要 shuffle

如果数据总是按固定顺序输入模型，可能会带来顺序偏差，尤其是在：

- 类别分布不均匀
- 数据按标签排序保存
- 相似样本连续出现

的情况下。

训练阶段通常会设置：

```python
shuffle=True
```

让每个 epoch 的样本顺序被打乱，从而减少顺序带来的偏差。

---

## 三、先建立一个直觉理解

你可以把整个数据管道想成“餐厅后厨的上菜流程”。

- `Dataset` 像菜单数据库：它知道每一道菜是什么、材料是什么、编号是什么
- `DataLoader` 像传菜系统：它决定每次送几道菜、送的顺序怎么安排
- 训练循环像厨房主线：每拿到一批菜，就开始加工处理

所以在训练里，并不是模型主动去磁盘里翻文件，而是：

1. `Dataset` 先定义怎么取样本
2. `DataLoader` 负责按 batch 调度这些样本
3. 训练循环每次从 `DataLoader` 拿到一批数据继续往下跑

这个理解非常重要，因为后面无论是表格数据、图像数据还是文本数据，你都会反复用到这个结构。

---

## 四、真实项目里怎么用

### 场景 1：CSV 表格分类任务

假设你有一个 `train.csv`，每一行是一条用户样本：

- 前 10 列是特征
- 最后一列是标签

这时你通常会在 `Dataset.__getitem__()` 里做：

- 根据索引取某一行
- 分出特征和标签
- 转成 Tensor
- 返回 `(x, y)`

然后用 `DataLoader` 按 batch 迭代。

### 场景 2：图片分类

假设你的目录结构是：

```text
images/
├─ cat/
├─ dog/
```

你在 `Dataset` 中通常会做：

- 根据索引找到某张图片路径
- 读取图片
- 做 resize / normalize / tensor 化
- 返回图像 Tensor 和类别标签

DataLoader 则负责：

- 每次拼成一个 batch
- 是否打乱顺序
- 后台是否并行读取

### 场景 3：本地 Windows 环境调试

在你的本地 Windows 环境里，DataLoader 的 `num_workers` 是一个常见坑点。

很多教程默认从 Linux 经验出发，直接把 `num_workers` 设得很大。但在 Windows 下，建议更稳妥地：

- 先从 `num_workers=0` 跑通
- 再逐步增加

因为 Windows 多进程加载机制和 Linux 有差异，报错时不容易第一时间定位。

---

## 五、流程图 / 结构图（Mermaid）

下面这张图展示了 Dataset、DataLoader 和训练循环之间的关系：

```mermaid
flowchart LR
    A[原始数据源
CSV 图片 文本] --> B[Dataset 按索引取单条样本]
    B --> C[返回 单个样本 x y]
    C --> D[DataLoader 组装成 batch]
    D --> E[训练循环读取 batch]
    E --> F[模型前向传播]
    F --> G[损失计算与反向传播]
```

记住这个顺序后，你以后就不容易把职责搞混：

- 原始数据不直接喂模型
- 必须先经过 Dataset 和 DataLoader 这层标准化组织

---

## 六、从零写一个最小可运行示例

下面先写一个最简单的自定义数据集。

这个例子依然用随机数据模拟真实训练任务，但结构和真实工程非常接近。重点是让你真正看清楚：

- `__len__()` 是干什么的
- `__getitem__()` 是干什么的
- `DataLoader` 最后返回的 batch 长什么样

```python
import torch
from torch.utils.data import Dataset, DataLoader


class ToyDataset(Dataset):
    """
    一个最小可运行的自定义数据集。

    这里我们用随机生成的数据模拟一个二分类任务：
    - 每条样本有 10 个特征
    - 标签是 0 或 1

    真实项目里，这里可以替换成：
    - 读取 CSV
    - 读取图片路径列表
    - 读取文本和标签文件
    """
    def __init__(self, num_samples=100, num_features=10):
        # 生成输入特征，shape 为 (样本数, 特征数)
        # 使用 float32，便于后续直接送入神经网络
        self.x = torch.randn(num_samples, num_features, dtype=torch.float32)

        # 生成分类标签，shape 为 (样本数,)
        # 对于分类任务，标签常见类型是 long / int64
        self.y = torch.randint(0, 2, (num_samples,), dtype=torch.long)

    def __len__(self):
        # 返回数据集大小
        # DataLoader 会依赖它来决定一轮 epoch 里有多少批数据
        return len(self.x)

    def __getitem__(self, idx):
        # 根据索引返回一条样本
        # 返回格式通常是 (输入, 标签)
        # 真实项目里，也常会在这里做：
        # - 文件读取
        # - 数据清洗
        # - transform 处理
        return self.x[idx], self.y[idx]


# 实例化数据集
dataset = ToyDataset(num_samples=100, num_features=10)

# 看看数据集大小
print("数据集大小:", len(dataset))

# 看看第 0 条样本长什么样
sample_x, sample_y = dataset[0]
print("单条样本特征 shape:", sample_x.shape)
print("单条样本标签:", sample_y)

# 使用 DataLoader 把单条样本组织成 batch
loader = DataLoader(
    dataset,
    batch_size=16,   # 每个 batch 放 16 条样本
    shuffle=True,    # 训练时通常打乱顺序
    num_workers=0    # Windows 本地环境建议先从 0 开始
)

# 从 DataLoader 中取出一个 batch 看看
for batch_x, batch_y in loader:
    print("batch_x.shape =", batch_x.shape)
    print("batch_y.shape =", batch_y.shape)
    print("batch_y.dtype =", batch_y.dtype)
    break
```

---

## 七、进一步看一个“更接近真实项目”的例子

上面的例子里数据是直接随机生成的。下面这个例子演示一种更接近真实工程的写法：

- 我们先把“样本元信息”放到一个列表里
- `Dataset` 再根据索引读取并转换

这和现实中“先有路径列表 / 表格索引，再按需取样本”的模式很接近。

```python
import torch
from torch.utils.data import Dataset


class UserFeatureDataset(Dataset):
    """
    模拟一个表格任务的数据集。

    records 中每一条记录都包含：
    - features: 特征列表
    - label: 类别标签
    """
    def __init__(self, records):
        # 保存原始记录
        self.records = records

    def __len__(self):
        # 数据集大小就是记录条数
        return len(self.records)

    def __getitem__(self, idx):
        # 取出第 idx 条记录
        record = self.records[idx]

        # 把 Python 列表转成 float32 Tensor
        # 这一步很常见，因为原始数据很多时候不是 Tensor 格式
        features = torch.tensor(record["features"], dtype=torch.float32)

        # 标签转成 long，便于后续分类损失函数使用
        label = torch.tensor(record["label"], dtype=torch.long)

        return features, label


records = [
    {"features": [0.2, 1.5, -0.3, 2.1], "label": 0},
    {"features": [1.2, 0.7, 0.5, -1.3], "label": 1},
    {"features": [0.9, -0.4, 1.1, 0.3], "label": 0},
]

dataset = UserFeatureDataset(records)

for i in range(len(dataset)):
    x, y = dataset[i]
    print(f"第 {i} 条样本: x.shape={x.shape}, y={y}")
```

这个例子的意义在于让你意识到：

- `Dataset` 并不要求你一开始就把所有东西变成大 Tensor
- 它更关心的是“给我一个索引，我能稳定返回一条样本”

这正是它和普通“直接把所有数据写死在内存里”的区别。

---

## 八、拆解代码执行过程

### 1. `__len__()` 决定数据集规模

训练时，框架需要知道：

- 一共有多少条样本
- 当前 epoch 会迭代多少次

这就是 `__len__()` 的作用。

如果这里写错，常见后果包括：

- DataLoader 遍历不完整
- 索引越界
- 训练轮数和预期不一致

### 2. `__getitem__()` 决定单条样本格式

这是 `Dataset` 的核心。

训练循环后面通常会写成：

```python
for x, y in loader:
    ...
```

那就意味着你的 `__getitem__()` 返回值要足够稳定，最好始终是统一结构，比如：

- `(features, label)`
- `(image_tensor, label)`
- `{"input_ids": ..., "label": ...}`

不要今天返回二元组，明天返回三元组，否则训练代码会很难维护。

### 3. DataLoader 会把单条样本自动“堆叠”成 batch

这是很多初学者第一次看到时会觉得神奇的地方。

比如 `Dataset` 每次返回：

- 一个 shape 为 `(10,)` 的特征向量
- 一个标量标签

DataLoader 在 batch_size=16 时，就会自动组织成：

- `batch_x.shape == (16, 10)`
- `batch_y.shape == (16,)`

这就是你后面模型能直接处理 batch 的原因。

### 4. 为什么数据管道是“训练系统”的地基

模型写得再漂亮，如果数据管道不稳定，训练还是会出问题。

比如：

- 某些样本返回 float，某些返回 int
- 有些图片尺寸不一致，没做统一 transform
- 标签编码不统一
- Dataset 返回结构不固定

这些问题往往不是模型层面的 bug，但一样会让训练彻底跑不起来。

---

## 九、运行结果应该怎么看

运行上面的代码后，重点看下面几件事。

### 1. 单条样本 shape 是否符合预期

例如：

- `sample_x.shape` 是否是 `(10,)`
- `sample_y` 是否是单个标签值

如果单条样本都不对，后面 batch 更不可能对。

### 2. batch shape 是否符合训练要求

重点看：

- `batch_x.shape == (16, 10)`
- `batch_y.shape == (16,)`

这两项几乎是最基础的数据管道健康检查。

### 3. dtype 是否适合后续任务

例如：

- 特征通常应是 `float32`
- 分类标签通常应是 `long`

你可以用：

```python
print(batch_x.dtype)
print(batch_y.dtype)
```

做进一步检查。

### 4. batch 是否真的被打乱

如果你把 `shuffle=True` 改成 `False`，可以多次运行观察样本顺序是否变化。这能帮助你更直观理解 DataLoader 的作用。

---

## 十、常见错误与排查

### 问题 1：`__getitem__()` 返回格式不统一

比如有时返回 `(x, y)`，有时返回 `(x,)`，这会让 DataLoader 在拼 batch 时出问题。

解决思路：

- 保持每条样本返回结构一致
- 必要时先单独打印若干个 `dataset[i]` 看格式

### 问题 2：标签类型不合适

分类任务里，标签很多时候需要 `long`。

如果你后面要接 `CrossEntropyLoss`，但标签是 `float32`，可能就会报错或行为不符合预期。

### 问题 3：batch 维度和模型输入不匹配

例如模型期待输入 `(batch_size, 10)`，但你返回的数据 shape 却不是这样。

排查建议：

- 打印 `batch_x.shape`
- 在模型前向传播前再打印一次输入 shape

### 问题 4：Windows 下 `num_workers` 引发问题

在 Windows 本地环境里，`DataLoader(num_workers>0)` 有时会导致一些新手不容易定位的问题。

更稳妥的建议是：

1. 先用 `num_workers=0` 跑通
2. 再根据需要逐步增加
3. 如果报错，先回退到 0 确认是不是多进程加载引起的

### 问题 5：把所有逻辑都堆在训练循环里

有些人会在训练循环里直接写文件读取、标签解析、数据清洗。短期看似能跑，长期会非常混乱。

更合理的做法是：

- 样本读取逻辑放进 `Dataset`
- batch 调度交给 `DataLoader`
- 训练循环只管训练流程本身

---

## 十一、本篇小结

这一篇最重要的不是背 API，而是建立下面这个分工意识：

- `Dataset` 负责定义“单条数据如何取出”
- `DataLoader` 负责定义“单条数据如何组成 batch 并被迭代”
- 训练循环只负责消费 batch，而不应该亲自管理底层读取细节

你一旦把这三层分工建立起来，后面切换到：

- 表格数据
- 图像数据
- 文本数据
- 多模态数据

都会轻松很多，因为骨架并没有变。

---

## 十二、练习题

### 练习 1：修改样本规模
把 `ToyDataset` 的样本数改成 `256`，然后设置：

- `batch_size=32`

观察一个 epoch 大约会有多少个 batch。

### 练习 2：修改特征维度
把每条样本的特征数从 `10` 改成 `20`，确认：

- 单条样本 shape
- batch shape

是否都随之变化。

### 练习 3：观察 shuffle 行为
分别设置：

- `shuffle=True`
- `shuffle=False`

多运行几次，对比 batch 中样本顺序是否发生变化。

### 练习 4：故意制造一个 dtype 问题
尝试把标签改成 `float32`，再思考如果后面接 `CrossEntropyLoss` 会发生什么问题。

### 练习 5：思考题
为什么说：

- `Dataset` 是“定义单条样本”
- `DataLoader` 是“组织批量数据流”

如果把这两者的职责混在一起，会给训练代码维护带来什么问题？

---

## 下一篇预告

下一篇我们会把前面的内容真正串起来，进入 **完整训练循环实战**。

到那时你会把这些零散部件第一次完整接上：

- 数据从 DataLoader 进来
- 模型做前向传播
- loss 触发反向传播
- 优化器根据梯度更新参数
