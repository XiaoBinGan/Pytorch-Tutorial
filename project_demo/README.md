# project_demo

这是 `PyTorch 系列教程_EP15_项目实战_整理一个可复用训练模板.md` 对应的**真实可运行最小模板工程**。

它的目标不是一上来就做复杂框架，而是先给你一套在本地 Windows 环境下可以跑通的 PyTorch 项目闭环：

- 训练
- 验证
- 测试评估
- 推理
- 权重保存
- 训练历史与曲线输出

---

## 当前目录结构

```text
project_demo/
├─ checkpoints/
├─ data/
├─ outputs/
├─ config.py
├─ datasets.py
├─ models.py
├─ train.py
├─ eval.py
├─ infer.py
└─ README.md
```

---

## 当前已经支持的能力

### 1. 训练集 / 验证集 / 测试集分离

模板会把原始训练集自动拆成：

- `train_loader`
- `val_loader`

并保留独立的：

- `test_loader`

这样训练时就能区分：

- 训练误差
- 验证集表现
- 测试集结果

### 2. early stopping

训练时会根据验证集 `val_loss` 判断是否继续训练。

如果连续若干轮验证集没有改善，就会提前停止，避免无意义地继续训练。

### 3. 训练历史保存

训练结束后会在 `outputs/` 下生成：

- `history.json`
- `training_curves.png`

这样你就不只是看控制台输出，还能看到训练过程曲线。

---

## 每个文件负责什么

- `config.py`
  - 集中管理路径、设备、超参数、验证集比例、early stopping 参数
- `datasets.py`
  - 构建 MNIST 数据集，并拆分训练 / 验证 / 测试 DataLoader
- `models.py`
  - 统一模型定义，避免训练和推理结构不一致
- `train.py`
  - 训练模型、验证模型、保存最优权重、保存训练历史和曲线
- `eval.py`
  - 独立加载最佳模型并在测试集上评估
- `infer.py`
  - 对单张本地图片做推理

---

## 使用步骤

### 1. 进入目录

```powershell
cd G:\openclaw\docs\PyTorch-教程\project_demo
```

### 2. 训练模型

```powershell
python train.py
```

训练完成后，你应该能看到：

- `checkpoints\best_model.pt`
- `checkpoints\last_checkpoint.pt`
- `outputs\history.json`
- `outputs\training_curves.png`

### 3. 独立评估模型

```powershell
python eval.py
```

### 4. 对单张图片推理

```powershell
python infer.py sample.png
```

注意：
- 当前模板默认按 MNIST 的输入方式处理图片
- 所以输入图片最好是接近手写数字风格的单目标图片
- 预处理会自动执行灰度化、缩放到 `28x28` 和 `ToTensor()`

---

## 你以后最常改哪些地方

如果你要把这个模板迁移到自己的任务，通常优先改：

- 换数据集：改 `datasets.py`
- 改学习率、epoch、验证集比例：改 `config.py`
- 换模型结构：改 `models.py`

而训练、评估、推理的整体入口逻辑，可以尽量继续复用。

---

## 后续还能怎么升级

如果你还想继续把它做得更像真实项目，可以再加：

- 命令行参数支持
- 更详细的日志系统
- confusion matrix
- 更复杂的模型切换机制
- 迁移学习版本模板
- 自动恢复 checkpoint 续训

---

## 一句话总结

这是一个**已经支持训练—验证—测试评估—推理—曲线输出—early stopping** 的本地 PyTorch 最小模板工程，已经足够作为你后续分类项目的起点。
