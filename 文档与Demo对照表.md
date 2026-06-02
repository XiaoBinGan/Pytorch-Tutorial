# 文档和 demo 的对应关系

这个文件主要是给我自己和后面看仓库的人做个索引，免得只看到一堆 md，不知道 demo 对应在哪里。

## 先说整体

这套内容现在分三部分：

1. 15 篇教程文档
2. 两个集中验证脚本
3. 一个最小项目模板 `project_demo/`

两个验证脚本分别是：

- `demo_validate_ep01_ep05.py`
- `demo_validate_ep06_ep15.py`

前者负责前 5 篇偏基础的内容，后者负责后面那批更偏训练、保存、推理、模板的内容。

---

## 对应关系

### EP01 ~ EP05
这几篇没有再额外拆很多独立 demo 文件，主要是放在：

- `demo_validate_ep01_ep05.py`

覆盖内容大概是：
- EP01：环境自检
- EP02：Tensor 基本操作
- EP03：自动求导和梯度
- EP04：`nn.Module` 最小网络
- EP05：`Dataset` / `DataLoader`

备注：EP01 之前有个激活命令写坏了，已经修过。

### EP06
对应：
- `demo_validate_ep06_ep15.py`

这里主要验证最小训练循环能不能跑通。

### EP07
这篇是 MNIST，不在单独 demo 文件里拆出来，但实际测过：
- 用 `torchvision.datasets.MNIST(..., download=True)` 下载并读取过

这篇如果以后要再拆，也可以单独补个 `demo_ep07_mnist.py`，但当前还没单独建。

### EP08
对应：
- `demo_validate_ep06_ep15.py`

测的是 `SmallCNN` 结构和输出 shape。

### EP09
对应：
- `demo_validate_ep06_ep15.py`

主要是验证准确率计算那段逻辑。

### EP10
对应：
- `demo_validate_ep06_ep15.py`

测过的内容：
- `state_dict` 保存
- 权重加载
- checkpoint 恢复

### EP11
这篇我后来补了两种写法：

1. 离线结构验证
   - `weights=None`
2. 在线预训练权重
   - `weights=ResNet18_Weights.DEFAULT`

这两种都实际测过。

### EP12
对应：
- `demo_validate_ep06_ep15.py`

这里主要是 device、autocast、GradScaler 这些结构有没有写崩。

备注：当前验证环境是 CPU，不是 GPU 机器，所以 GPU 路径属于结构检查，不是硬件实测。

### EP13
对应：
- `demo_validate_ep06_ep15.py`

这里测了单图推理链路。
之前测试时生成过 `samples/sample.png`，不过为了仓库干净已经清掉了。

### EP14
这篇比较特殊，不是一个单独 demo 能概括的。
它更像“前面所有 demo 真跑时会遇到的报错经验总结”，所以算是被前面的验证过程间接覆盖了。

### EP15
这篇对应：
- `project_demo/`
- `demo_validate_ep06_ep15.py`

现在 `project_demo/` 里保留的是最小骨架：
- `config.py`
- `models.py`
- `infer.py`
- `README.md`

它现在还不是完整训练项目，更像一个干净的起点。



---

## 最后说一句

这个文件不是正式文档，更像仓库维护说明。
作用就一个：以后回来看时，能快速知道“哪篇文档对应哪个 demo，哪些东西是测过的，哪些只是说明文字”。
