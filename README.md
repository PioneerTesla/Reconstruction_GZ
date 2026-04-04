# Reconstruction_GZ：基于扩散模型的雷达 PRI 序列重建

本项目提出并实现了 **DiffSeqPRI**——一种基于扩散模型的序列到序列（Seq2Seq）框架，用于从含噪/缺失/虚假脉冲的雷达脉冲重复间隔（PRI）序列中重建干净的 PRI 序列，并在下游工作模式识别任务中验证重建质量。

---

## 目录

- [背景](#背景)
- [项目结构](#项目结构)
- [方法概述](#方法概述)
- [环境依赖](#环境依赖)
- [数据集准备](#数据集准备)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [评估指标](#评估指标)
- [下游任务：工作模式识别](#下游任务工作模式识别)
- [结果输出](#结果输出)

---

## 背景

雷达信号在实际采集过程中常受到以下三类干扰，导致 PRI 序列失真：

| 场景 | 描述 |
|------|------|
| **Miss（漏脉冲）** | 部分脉冲未被探测，序列出现跳跃 |
| **Spurious（虚假脉冲）** | 噪声产生额外的虚假脉冲，序列被污染 |
| **Mix（混合）** | 漏脉冲与虚假脉冲同时存在 |

本项目将 PRI 重建建模为 Seq2Seq 问题：以含噪观测序列为输入，预测干净的 Ground Truth 序列。通过离散扩散模型在嵌入空间中逐步去噪，实现高精度重建。

---

## 项目结构

```
Reconstruction_GZ/
├── config.json                  # 训练超参数配置
├── model.py                     # DiffSeqPRI 模型（Transformer + 扩散）
├── train_pri.py                 # 主训练脚本
├── run_file.py                  # 批量训练启动器
├── evaluation.py                # 评估指标（精度、MAE、多样性、多数投票）
├── visualization.py             # 训练曲线、重建对比、混淆矩阵等可视化
├── pri_tokenizer.py             # PRI 值均匀量化器（PRIQuantizer）
├── pri_dataset.py               # DiffuSeq 风格 Dataset / Collator
├── utlis.py                     # 通用工具函数（数据范围读取、调度器等）
├── rounding.py                  # 嵌入空间最近邻取整
├── smoke_validate.py            # 快速冒烟测试
├── test.py                      # 独立测试脚本
├── dataset/                     # 数据集（定长，序列长度 ~100）
│   ├── Ground_Truth/            # 干净 PRI .pt 文件
│   ├── Miss/                    # 漏脉冲观测序列
│   ├── Spurious/                # 虚假脉冲观测序列
│   ├── Mix/                     # 混合噪声观测序列
│   ├── process.py               # 数据生成脚本
│   ├── Radar_words.py           # 雷达工作模式 PRI 仿真
│   ├── Radar_phases.py          # 相位调制模型
│   └── Modulation_types.py      # 调制类型定义
├── dataset_random_len/          # 变长数据集（序列长度 60~80）
├── downstream_recognition/      # 下游工作模式识别
│   ├── generate_data.py         # 生成分类用 PRI 样本
│   ├── model.py                 # PRIModeClassifier（LSTM 分类器）
│   └── train.py                 # 工作模式识别训练与评估
├── CheckPoint/                  # 训练检查点
└── generated_outputs/           # 可视化输出（图片、日志）
```

---

## 方法概述

### 模型：DiffSeqPRI

DiffSeqPRI 将扩散模型与 Transformer 结合，完成 Seq2Seq PRI 重建：

```
观测序列 (Source)
      │
      ▼
[PRI 量化器] ──→ 离散 Token ID
      │
      ▼
[词嵌入层]  ──→ 连续嵌入向量
      │
      ▼
[Transformer 编码器]  (6层，8头，含时间步嵌入)
      │
      ▼
[Gaussian Diffusion 去噪]  (正向加噪 / 反向迭代去噪)
  ├── DDPM 采样（完整 T 步）
  └── DDIM 采样（加速，仅 5 步）
      │
      ▼
[最近邻取整 (Rounding)]  ──→ 预测 Token ID
      │
      ▼
[PRI 量化器解码]  ──→ 重建 PRI 序列
```

**核心组件：**

- **`PRIQuantizer`**：将连续 PRI 值均匀量化为 1500 个离散 Token（范围 [1, 1500] µs），支持 START / END / SEP / PAD 等特殊 Token。
- **`DenoiseTransformer`**：6 层 Transformer 编码器，融合时间步嵌入与位置编码，对含噪嵌入序列进行去噪。
- **`GaussianDiffusion1D`**：1D 高斯扩散过程，支持余弦 / 线性 Beta Schedule，以及 DDPM / DDIM 两种采样策略。
- **`EMAModel`**：指数移动平均参数平滑，提升推理稳定性。

**训练损失：**

$$\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda \cdot \mathcal{L}_{\text{CE}}$$

其中 MSE 在嵌入空间计算扩散重建误差，CE 对最终预测 Logits 计算交叉熵，$\lambda$（`ce_weight`）默认为 0.1。

---

## 环境依赖

```bash
pip install torch numpy matplotlib
```

| 依赖 | 说明 |
|------|------|
| PyTorch ≥ 1.12 | 模型训练与推理 |
| NumPy | 数值计算 |
| Matplotlib | 可视化 |

---

## 数据集准备

### 方式一：使用已有数据集

将 `.pt` 格式的 PRI 序列文件放入对应目录，每个文件需包含键 `'seq'`（1D float tensor）：

```
dataset/
  Ground_Truth/  ← 干净 PRI
  Miss/          ← 漏脉冲版本
  Spurious/      ← 虚假脉冲版本
  Mix/           ← 混合噪声版本
```

### 方式二：仿真生成

修改 `dataset/process.py` 中的配置后运行：

```bash
cd dataset
python process.py
```

可在 `dataset.json` 中调整：

```json
{
  "num_seqs": 10000,
  "ground_truth_seq_length": 100,
  "miss_process_ratio": [0.1, 0.15, 0.20, 0.25, 0.30],
  "suprious_process_ratio_min": [0.05, 0.10, 0.15]
}
```

支持的雷达工作模式（18 种）：

| 类别 | Word 编号 | 调制类型 |
|------|-----------|----------|
| sliding | 12–15 | 滑变 PRI |
| stagger | 16–21 | 交错 PRI |
| dwell_switch | 22–28 | 驻留切换 |
| complex_dwell | 29 | 复杂驻留 |

---

## 快速开始

### 单次训练

```bash
python train_pri.py
```

训练参数从 `config.json` 读取。

### 批量训练（多场景/多数据集）

```bash
# Miss + Spurious + Mix，两种数据集
python run_file.py --root dataset dataset_random_len --scene Miss Spurious Mix

# 仅 Miss 场景，自定义参数
python run_file.py --root dataset --scene Miss --epochs 500 --learning_rate 3e-4

# 预览配置而不实际训练
python run_file.py --root dataset --scene Miss Mix --dry_run
```

### 冒烟测试（快速验证环境）

```bash
python smoke_validate.py
```

---

## 配置说明

所有超参数集中在 `config.json`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `root` | `dataset_random_len` | 数据集根目录 |
| `scene` | `Miss` | 噪声场景（Miss / Spurious / Mix） |
| `epochs` | `1000` | 训练轮数 |
| `batch_size` | `128` | 批大小 |
| `learning_rate` | `1e-4` | 初始学习率 |
| `diff_steps` | `40` | 扩散总步数 T |
| `ddim_steps` | `5` | DDIM 加速步数 |
| `ddim_eta` | `0.0` | DDIM 随机性（0 = 确定性） |
| `noise_schedule` | `cosine` | Beta 调度（cosine / linear） |
| `sampling_method` | `ddpm` | 采样方法（ddpm / ddim） |
| `quantizer_bins` | `1500` | 量化 Bin 数 |
| `quantizer_min` | `1` | PRI 最小值（µs） |
| `quantizer_max` | `1500` | PRI 最大值（µs） |
| `spliced_seq_length` | `220` | 拼接序列最大长度 |
| `ce_weight` | `0.1` | 交叉熵损失权重 λ |
| `loss_mode` | `mse+ce` | 损失模式（mse+ce / mse_only） |
| `seed` | `42` | 随机种子 |

---

## 评估指标

训练结束后自动在测试集上计算以下指标（`evaluation.py`）：

| 指标 | 说明 |
|------|------|
| **Exact Accuracy** | 预测 Token 与 Ground Truth 完全匹配的比例 |
| **Tolerance Accuracy** | 预测 Token 在 ±1 Bin 内的比例 |
| **MAE** | 预测与真实 PRI 值的平均绝对误差（µs） |
| **Length Match Rate** | 预测序列长度与真实序列长度一致的样本比例 |
| **Diversity** | 多次独立采样的预测多样性（扩散随机性分析） |
| **Majority Vote** | 多次采样后取众数的重建精度（自洽性提升） |

---

## 下游任务：工作模式识别

通过评估重建序列在雷达工作模式分类任务上的表现，验证重建质量：

```bash
cd downstream_recognition
python train.py                        # 默认参数
python train.py --epochs 80 --lr 1e-3  # 自定义
```

**分类器**（`PRIModeClassifier`）：基于 LSTM 的 4 类工作模式识别器，输入为 PRI 序列，输出为：

- `0` sliding（滑变）
- `1` stagger（交错）
- `2` dwell_switch（驻留切换）
- `3` complex_dwell（复杂驻留）

输出结果包括训练曲线、per-class Precision/Recall/F1 及混淆矩阵。

---

## 结果输出

训练过程中自动保存以下内容到 `generated_outputs/<model_name>/`：

| 输出文件 | 说明 |
|----------|------|
| `training_curves.png` | 训练/验证 Loss 曲线 |
| `reconstruction_comparison.png` | 原始序列 vs 重建序列对比图 |
| `codebook_visualization.png` | 量化码本可视化（PCA 投影） |
| `diffusion_denoise.png` | 扩散去噪过程可视化 |
| `confusion_matrix.png` | Token 级预测混淆矩阵 |
| `diversity_visualization.png` | 多次采样多样性分析 |
| `per_sample_accuracy.png` | 每样本精度分布直方图 |
| `train_log.txt` | 完整训练日志 |
| `checkpoint_best.pt` | 最优模型检查点 |
