<p align="center">
  <img src="https://pytorch.org/assets/images/pytorch-logo.png" width="400" alt="PyTorch">
</p>

<h1 align="center">PyTorch 从入门到项目闭环</h1>
<h3 align="center">15 篇 · 本地 Windows 实战 · RTX 5090 实测</h3>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/GPU-RTX%205090-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="NVIDIA">
  <img src="https://img.shields.io/badge/OS-Windows%2011-0078D6?style=for-the-badge&logo=windows&logoColor=white" alt="Windows">
  <img src="https://img.shields.io/badge/Chapters-15-FF6B35?style=for-the-badge&logo=bookstack&logoColor=white" alt="15 Chapters">
</p>

<p align="center">
  <b>作者：</b>吴佳浩 &nbsp;|&nbsp; <b>撰稿：</b>2026-05-25 &nbsp;|&nbsp; <b>实测：</b>RTX 5090 + 96GB + Windows 11
</p>

---

<p align="center">
  <b>不是"API 大全"，不是"Hello World 堆砌"，不是"看完还是不会做项目"</b><br>
  <b>这是一套让你从 <code>import torch</code> 走到独立完成一个完整项目闭环的实战教程</b>
</p>

---

## 为什么是 PyTorch？

### PyTorch 是什么

PyTorch 是由 Meta（原 Facebook）AI 研究院（FAIR）开发并开源的深度学习框架，核心基于 Torch 库，于 2016 年发布第一个公开版本。经过近十年的迭代，PyTorch 已经从单纯的"科研工具"发展为覆盖研究、开发、部署全链路的工业级框架。

PyTorch 的底层是一个高效的张量计算引擎，支持 GPU 加速和多节点分布式训练。它的上层提供了 `torch.nn`、`torch.optim`、`torch.utils.data` 等模块，将深度学习中最常见的构建、训练、数据加载环节封装得极度灵活且直观。

### 为什么 PyTorch 统治了深度学习

2017 年前后，深度学习框架之争的核心玩家是 TensorFlow 和 PyTorch。TensorFlow 采用了**静态计算图**——你需要先定义整个计算图，再提交执行。PyTorch 则选择了**动态计算图（Define-by-Run）**——在哪里写代码就在哪里构建图，运行到哪行就执行到哪行。

这个设计差异带来了三个决定性优势：

1. **调试体验**：PyTorch 的代码可以直接用 `print`、`pdb`、IDE 断点调试，因为每一行都在你写的时候立即执行。TensorFlow 1.x 需要先构建图、再 `Session.run()`，中间出了错很难定位。
2. **控制流自由**：PyTorch 中 `if`、`for`、`while` 等 Python 原生的控制流天然融入计算图，不需要学习特殊的控制流 API。这意味着你可以根据数据动态改变网络结构，这在 NLP 和强化学习中尤其重要。
3. **Pythonic 设计**：PyTorch 的 API 风格极度贴近 Python 和 NumPy 的使用习惯。`torch.Tensor` 的行为和 `numpy.ndarray` 几乎一致，学习曲线非常平滑。

这些优势让 PyTorch 在学术界迅速占据统治地位。根据 Papers with Code 的统计，从 2019 年开始，顶级会议中超过 75% 的论文使用 PyTorch 实现，到 2024 年这个比例已经超过 90%。工业界同样快速跟进——OpenAI、Tesla、Microsoft、NVIDIA 都将 PyTorch 作为主力框架。

### PyTorch 的生态系统

PyTorch 不是一个孤立的核心库，它背后有一个庞大的官方和社区生态：

| 组件 | 定位 | 一句话说明 |
|:--|:--|:--|
| `torch` | 核心库 | 张量计算 + 自动求导 + GPU 加速 |
| `torch.nn` | 神经网络模块 | 层、激活函数、损失函数、参数管理 |
| `torch.optim` | 优化器 | SGD、Adam、AdamW 等主流优化算法 |
| `torch.utils.data` | 数据加载 | Dataset + DataLoader，支持多进程、分布式采样 |
| `torchvision` | 计算机视觉 | 预训练模型（ResNet、ViT 等）、数据集、图像变换 |
| `torchaudio` | 音频处理 | 音频 I/O、特征提取、预训练模型 |
| `torchtext` | 自然语言处理 | 文本数据集、词表、分词工具 |
| `torch.cuda.amp` | 混合精度训练 | 用 FP16/FP32 混合精度加速训练，几乎不损失精度 |
| `torch.compile` | 图编译加速 | PyTorch 2.0 引入，用 `torch.compile(model)` 一行代码提速 |
| `torch.distributed` | 分布式训练 | DDP、FSDP，支持多 GPU 和多节点训练 |
| `TorchServe` | 模型服务 | 训练好的模型一键部署为 REST API |
| `PyTorch Lightning` | 训练框架（社区） | 把训练样板代码抽象出来，专注模型逻辑 |
| `HuggingFace Transformers` | 预训练模型（社区） | 几行代码加载 BERT、GPT、LLaMA 等大模型 |

这套生态意味着：你学完基础 PyTorch 之后，无论是做图像分类、目标检测、语义分割，还是 NLP、语音识别、强化学习、大模型微调，都有成熟的库和预训练模型可以直接使用。你不需要从零造轮子，但你也需要理解轮子是怎么转的——这正是本教程的目标。

### TensorFlow 2.x 的追赶与结局

TensorFlow 在 2019 年发布了 2.0 版本，引入了 Eager Execution 来模拟动态图，Keras 成为官方高级 API，API 友好度大幅提升。但此时 PyTorch 已经在研究社区建立了深厚的护城河，加上 Meta、NVIDIA 等公司在生态上的持续投入，TensorFlow 已难以逆转局面。到 2025 年，PyTorch 已经成为深度学习领域事实上的标准框架，绝大多数开源项目、论文复现、竞赛方案都基于 PyTorch。

**结论很简单：如果你在 2026 年要系统学深度学习，PyTorch 是唯一值得从零投入的框架。**

---

## 前置准备

本教程假定你已经掌握了 Python 基础语法和环境配置。如果你对以下内容还不太熟悉：

- `pip` / `venv` 等包管理和虚拟环境工具
- `list`、`dict`、`tuple`、`set` 等内置数据结构
- 函数定义、`class`、模块导入
- `numpy` 基础（`ndarray` 创建、索引、切片、广播）
- 列表推导式、`lambda`、文件读写

**请先移步我的 Python 入门指南，把基础语法和环境配置掌握后再回来：**

- 掘金专栏：[Python 入门指南](https://juejin.cn/column/7294907512445206591)
- CSDN 博客：[Python 入门指南](https://blog.csdn.net/weixin_43712047/category_13098818.html)

> Python 基础不稳，学 PyTorch 会频繁卡在语法细节上，本末倒置。花一周打好基础，后面 15 篇效率翻倍。

---

## 学习路线图

```mermaid
flowchart TB
    subgraph P1["第一阶段：打基础"]
        direction LR
        EP01["EP01 环境安装"] --> EP02["EP02 Tensor"]
        EP02 --> EP03["EP03 自动求导"]
        EP03 --> EP04["EP04 nn.Module"]
        EP04 --> EP05["EP05 DataLoader"]
        EP05 --> EP06["EP06 训练循环"]
    end

    subgraph P2["第二阶段：第一个像样任务"]
        direction LR
        EP06 --> EP07["EP07 MNIST 分类"]
        EP07 --> EP08["EP08 CNN 入门"]
        EP08 --> EP09["EP09 验证与过拟合"]
    end

    subgraph P3["第三阶段：工程实战"]
        direction LR
        EP09 --> EP10["EP10 保存与续训"]
        EP10 --> EP11["EP11 迁移学习"]
        EP11 --> EP12["EP12 GPU 与 AMP"]
        EP12 --> EP13["EP13 推理部署"]
        EP13 --> EP14["EP14 报错排查"]
        EP14 --> EP15["EP15 项目模板"]
    end

    P1 --> P2 --> P3

    style P1 fill:#1a1a2e,stroke:#e94560,color:#eee
    style P2 fill:#1a1a2e,stroke:#f5a623,color:#eee
    style P3 fill:#1a1a2e,stroke:#4ecca3,color:#eee
    style EP15 fill:#4ecca3,stroke:#fff,stroke-width:2px,color:#000
    style EP01 fill:#e94560,stroke:#fff,stroke-width:2px,color:#000
```

---

## 你是不是也有这些症状？

- 看了无数 PyTorch 教程，API 都认识，但一到自己动手就懵
- 能跑通别人的 Notebook，但换个数据集就报一片红
- `shape`、`dtype`、`device` 报错翻来覆去搞不定
- 训练完了不知道怎么保存、怎么部署、怎么给别人用
- 每次做新任务都从零拼脚本，没有一套可复用的骨架

**如果中了两条以上 —— 这套教程就是为你写的。**

---

## 学完你能做到什么

```mermaid
mindmap
  root((PyTorch<br/>能力地图))
    基础内功
      Tensor 操作
      自动求导机制
      nn.Module 搭建
      数据管道设计
    训练能力
      完整训练循环
      分类任务实战
      CNN 网络设计
      过拟合诊断
    工程素养
      模型保存恢复
      断点续训
      迁移学习
      混合精度训练
    生产就绪
      推理脚本
      系统化排错
      可复用模板
```

---

## 这套教程和别的有什么不一样？

| 常见教程 | 这套教程 |
|:--|:--|
| 堆 API 列表，看完还是不会写 | 每个概念都讲 **为什么**、**什么时候用**、**不用会怎样** |
| 代码没注释，改一行就炸 | **逐行中文注释**，每行都让你看懂 |
| 没有图，全靠脑补 | 关键流程全配 **Mermaid 流程图 / 架构图** |
| 只说"怎么做"，不说"踩什么坑" | 每篇都附 **常见报错 + 排查思路** |
| 学完不知道下一步 | 每章都有 **练习题 + 章节衔接指引** |
| 示例依赖云平台 | **本地 Windows 直接跑**，有 GPU 用 GPU，没 GPU 走 CPU |

---

## 训练管线全景图

```mermaid
graph LR
    subgraph Data["数据流"]
        A[原始数据] --> B[Dataset]
        B --> C[DataLoader]
    end

    subgraph Model["模型流"]
        D[nn.Module] --> E[forward]
        C -.->|batch| E
    end

    subgraph Train["训练循环"]
        E --> F[前向传播]
        F --> G[Loss 计算]
        G --> H[反向传播]
        H --> I[Optimizer 更新]
        I -.->|下一轮| F
    end

    subgraph Eval["评估与部署"]
        I --> J[验证集评估]
        J --> K[Checkpoint 保存]
        K --> L[推理脚本部署]
    end

    style Data fill:#16213e,stroke:#e94560,color:#fff
    style Model fill:#16213e,stroke:#f5a623,color:#fff
    style Train fill:#16213e,stroke:#4ecca3,color:#fff
    style Eval fill:#16213e,stroke:#00d2ff,color:#fff
```

---

## 章节总览

### 第一阶段：夯实基础

> 没有地基，后面全是空中楼阁。先把 Tensor / autograd / Module / DataLoader / 训练循环吃透。

| EP | 标题 | 核心收获 |
|:--:|:--|:--|
| 01 | 环境准备与安装验证 | 本地 PyTorch 环境零报错跑通 |
| 02 | Tensor 与张量基本操作 | 搞懂 `shape` / `dtype` / `device` 三件套 |
| 03 | 自动求导与反向传播 | 理解梯度怎么自动算、反向传播在干什么 |
| 04 | 用 nn.Module 搭建第一个网络 | 学会用 Module 组织任意网络结构 |
| 05 | Dataset 与 DataLoader 数据管道 | 把任意数据变成可训练的数据管道 |
| 06 | 完整训练循环实战 | 第一次把模型 + 数据 + loss + 优化器串成闭环 |

### 第二阶段：做出第一个像样任务

> 从"会写代码片段"进化到"会看模型表现"。

| EP | 标题 | 核心收获 |
|:--:|:--|:--|
| 07 | 分类任务：手写数字识别 | 完成第一个标准分类项目（MNIST） |
| 08 | 卷积神经网络 CNN 入门 | 理解为什么 CNN 比 Flatten 更适合图像 |
| 09 | 验证集、指标与过拟合处理 | 区分"训练好看"和"真正泛化好" |

### 第三阶段：工程实战

> 不再只是训练——开始具备真正做项目的骨架能力。

| EP | 标题 | 核心收获 |
|:--:|:--|:--|
| 10 | 模型保存、加载与断点续训 | 保存、恢复、续训——训练现场永不失联 |
| 11 | 迁移学习：基于 torchvision 预训练模型 | 站在巨人肩膀上，用预训练模型加速收敛 |
| 12 | GPU、混合精度与训练加速 | 榨干显卡性能，训练快 2~5 倍 |
| 13 | 推理脚本与本地部署 | 训练好的模型真正"用起来" |
| 14 | 常见报错排查与调试技巧 | 建立系统化排错思维——不再被红字吓到 |
| 15 | 项目实战：可复用训练模板 | 把所有能力收进一个模板，下次直接复用 |

---

## 快速导航

```mermaid
flowchart TD
    START((你想学什么?)) --> Q1{有 Python 基础吗?}
    Q1 -->|有，刚接触 PyTorch| A1["从头开始<br/>EP01 → EP15 顺序走"]
    Q1 -->|有，只想快速跑训练| A2["速通路线<br/>EP02 → 03 → 04 → 05 → 06"]
    Q1 -->|有，关心分类项目| A3["分类路线<br/>EP07 → 08 → 09 → 10 → 13"]
    Q1 -->|有，要工程化能力| A4["工程路线<br/>EP10 → 12 → 13 → 14 → 15"]
    Q1 -->|基础薄弱| A0["先补 Python<br/>👉 见上方「前置准备」"]

    style START fill:#e94560,color:#fff,stroke:#fff
    style A1 fill:#4ecca3,color:#000
    style A2 fill:#f5a623,color:#000
    style A3 fill:#00d2ff,color:#000
    style A4 fill:#a855f7,color:#000
```

---

## 环境搭建（3 分钟搞定）

```powershell
# 1. 进入工作目录
cd G:\openclaw\docs\PyTorch-教程

# 2. 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\activate

# 3. 安装依赖
python -m pip install --upgrade pip
pip install torch torchvision torchaudio matplotlib pandas scikit-learn jupyter

# 4. 验证安装
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

> 有 GPU 自动走 GPU，没 GPU 走 CPU——所有代码兼容两种模式。

---

## 项目骨架

```text
PyTorch-教程/
├── README.md                         ← 你在这里
├── 教程重写规范.md                    ← 统一编写标准
├── 文档与Demo对照表.md                ← 每篇对应的可运行代码
├── DEMO验证结果.md                    ← 实测通过的验证记录
├── demo_validate_ep01_ep05.py        ← 前半段自动化验证
├── demo_validate_ep06_ep15.py        ← 后半段自动化验证
├── project_demo/                     ← 可复用的项目模板（EP15 产出）
├── data/                             ← 数据集存放
├── outputs/                          ← 训练输出
├── runs/                             ← TensorBoard 日志
├── checkpoints/                      ← 模型检查点
└── pytorch 你不学_EP*.md             ← 15 篇核心教程
```

---

## 怎么学效果最好？

```mermaid
graph TD
    A[读理论] --> B[亲手跑代码]
    B --> C[改参数 / 换数据]
    C --> D[故意制造报错]
    D --> E{报错类型?}
    E -->|shape| F[检查维度]
    E -->|dtype| G[检查类型]
    E -->|device| H[检查设备]
    F & G & H --> I[修复]
    I --> J[理解加深]
    J --> K[做练习]
    K --> L[真正掌握]

    style A fill:#16213e,stroke:#f5a623,color:#fff
    style D fill:#e94560,color:#fff
    style L fill:#4ecca3,color:#000,stroke-width:3px
```

**只看不改 = 眼熟而已。** 真正掌握的唯一路径是：跑通 → 改动 → 排错 → 复用。

---

## 核心理念

<p align="center">
  <b>不追求"高级模型"或"复杂框架"</b><br><br>
  <b>先把这几件事学扎实：</b>
</p>

<p align="center">
  <code>Tensor · autograd · Module</code><br>
  <code>DataLoader · 训练循环</code><br>
  <code>验证与过拟合 · 模型保存</code><br>
  <code>推理脚本 · 可复用模板</code>
</p>

<p align="center">
  <b>这些稳了，后面 CNN、Transformer、RL、大模型微调……全部轻松接住。</b>
</p>

---

<p align="center">
  <br>
  <b>现在就打开 <a href="./pytorch%20你不学_EP01_环境准备与安装验证.md">EP01</a>，开始你的 PyTorch 实战之旅！</b>
  <br><br>
  <sub>Made with ❤️ by 吴佳浩 · 2026 · 本地 Windows 实测</sub>
</p>
