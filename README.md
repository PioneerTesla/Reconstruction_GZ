# Reconstruction_GZ

基于 **DiffuSeq**（扩散式 seq2seq）的 **雷达 PRI（脉冲重复间隔）序列重建** 研究仓库。
给定被噪声污染的 PRI 观测序列（丢失 Miss / 虚假 Spurious / 混合 Mix），训练一个扩散模型重建出干净的 PRI 序列，并在下游完成雷达工作模式识别。

配套文档：`DiffSeq.pdf`（见 Releases / 外部存储）。
对比方法：`BaseLine/`（DenoisingAE、RNNPrediction、SemanticCoding）。

---

## 目录结构

```
.
├── train_pri.py               # 主训练入口（精简后的 main，~140 行）
├── test.py                    # 推理脚本（加载 checkpoint、导出重建结果）
├── run_file.py                # 批量任务调度器（多 scene × 多数据集）
├── trainer.py                 # Trainer 类：epoch 循环、双 best checkpoint、LR schedule、early stop
├── reporting.py               # 训练后评估与出图流程（confusion / EMA / 多样性 / 多数投票 / 重建样例）
├── data_loader.py             # 量化器构建 + 90/10 划分 + DataLoader
├── model.py                   # PRIDiffuSeq 模型 + EMA
├── pri_tokenizer.py           # PRI 量化器（prototype / uniform）
├── pri_dataset.py             # DiffuSeq 风格 seq2seq Dataset / Collator
├── rounding.py                # 连续 hidden → 最近 token id
├── evaluation.py              # 测试指标、多样性、多数投票、混淆矩阵
├── visualization.py           # 论文风格绘图
├── utils.py                   # 日志 tee / 随机种子 / argparse / PRI 取值范围 / DDIM trace 步
├── config.json                # 基础超参
├── jobs.json                  # 批量任务定义（可覆盖 config.json）
├── requirements.txt
├── LICENSE
├── dataset/                   # 数据集相关 Python 模块（.py 进仓，.pt 数据不进仓）
│   ├── Modulation_types.py
│   ├── Radar_phases.py
│   ├── Radar_words.py
│   ├── process.py
│   └── dataset.json
├── downstream_recognition/    # 下游雷达工作模式识别
│   ├── generate_data.py
│   ├── model.py
│   └── train.py
└── BaseLine/                  # 对比方法
    ├── DenoisingAE/{model.py,run.py}
    ├── RNNPrediction/{model.py,run.py}
    └── SemanticCoding/{algorithm.py,run.py}
```

运行时会生成（均被 `.gitignore` 屏蔽）：

```
CheckPoint/   # 训练产出的权重、日志、可视化原始位置
logs/         # 每次运行的文本日志
generated_outputs/, test_results/
```

用于论文 / 算法分析的结果图会从 `CheckPoint/<exp>/visuals/` 拷贝到版本化目录
**`docs/figures/`**，见 [`docs/figures/README.md`](docs/figures/README.md)。

---

## 环境

- Python **3.10+**
- PyTorch **2.0+**（CUDA 11.8 / 12.x 或 CPU）
- 见 `requirements.txt`

```bash
# 先按 https://pytorch.org/get-started/locally/ 安装匹配 CUDA 的 PyTorch，然后：
pip install -r requirements.txt
```

---

## 数据集准备

数据集体量较大（~160MB/份），**未入库**，请自行准备以下目录结构：

```
dataset/
├── Ground_Truth/   # 干净 PRI 序列（*.pt，每个文件含 {"seq": tensor} 字段）
├── Miss/           # 丢失场景的观测序列
├── Spurious/       # 虚假场景的观测序列
└── Mix/            # 混合场景的观测序列
```

以及可选的变长数据集 `dataset_random_len/`，结构同上。

每个 `.pt` 文件命名形如 `word1_12_1016.pt`，前缀（`word1_12`）表示雷达工作模式类别，被用于下游识别任务。

如需数据生成脚本，请参考 `dataset/process.py`、`dataset/Radar_words.py`、`dataset/Radar_phases.py`，配合 `dataset/dataset.json` 使用。

数据与预训练权重建议通过 **GitHub Releases**、网盘或对象存储发布，而非直接入库。

---

## 常用命令

### 单次训练

```bash
# 使用 config.json 中的默认参数
python train_pri.py

# 覆盖命令行参数（argparse）
python train_pri.py --scene Miss --root dataset
```

### 批量训练

参数可写在 `jobs.json`，缺省字段回退到 `config.json`：

```bash
python run_file.py --jobs jobs.json

# 或者通过命令行生成多场景组合
python run_file.py --root dataset dataset_random_len --scene Miss Spurious Mix

# 预览任务但不执行
python run_file.py --jobs jobs.json --dry_run

# 从中断处继续
python run_file.py --jobs jobs.json --resume
```

### 推理 / 测试

```bash
python test.py --model_path CheckPoint/DiffSeqPRI_Miss_dataset/best_model.pth \
               --root dataset --scene Miss --output_dir test_results
```

每个测试样本会落盘为一个 `.pt`，包含 `clean_pri` / `pred_pri` / `src_pri` / `exact_acc` / `mae` 等字段。

### 下游工作模式识别

```bash
python downstream_recognition/train.py
```

### 对比方法

```bash
python BaseLine/DenoisingAE/run.py
python BaseLine/RNNPrediction/run.py
python BaseLine/SemanticCoding/run.py
```

---

## 参数优先级

当前同时使用 `config.json` + `jobs.json` + 命令行 argparse，**优先级从高到低**：

1. 命令行参数（`--epochs 100` 等）
2. `jobs.json` 中该任务的字段（仅 `run_file.py` 批量模式下）
3. `config.json` 中的全局默认

---

## 输出结构

默认训练会在 `Checkpoint/<model_name>_<scene>_<root>/` 下产出：

```
best_model.pth            / best_model_ema.pth             # 按 test_loss 选的最优权重
best_model_by_metric.pth  / best_model_by_metric_ema.pth   # 按重建指标选的最优权重 (exact_acc + 0.5·len_match_rate)
latest_model.pth
log.txt                                 # 训练日志（stdout/stderr 镜像）
visuals/                                # 码本、去噪过程、混淆矩阵、训练曲线、每字类柱状图等
```

最终 `visuals/` 与终端 final report 默认使用 `best_model_by_metric.pth`；若想退回原先行为，在 `config.json` 里设 `"final_eval_use_best_metric": false`。

---

## 代码模块划分

| 模块 | 职责 |
| --- | --- |
| `train_pri.py` | 组装：读 config → 建数据 → 建模型/EMA/optimizer → `Trainer(...).fit()` → `reporting.run_full_report(...)` |
| `data_loader.py` | `build_quantizer` / `build_demo_loader` / `extract_word_type` —— 90/10 划分 + 量化器 |
| `trainer.py` | `Trainer`, `TrainerPaths` —— 训练循环、双 best-checkpoint、LR schedule (warmup + Plateau / Cosine)、early stop |
| `reporting.py` | `run_full_report` 及其子函数 —— confusion matrix / test-mode & EMA 评估 / per-sample & per-word-type 图 / 多样性 / 多数投票 / 重建样例 / terminal demo |
| `utils.py` | 通用工具：`DualOutput`, `set_seed`, `resolve_device`, `create_argparser`, `choose_trace_steps`, `get_pri_range` |
| `evaluation.py` | 核心评估指标与混淆矩阵数据收集（被 `trainer.py` 和 `reporting.py` 调用） |
| `visualization.py` | 纯绘图函数（无 torch/训练依赖） |
| `model.py` / `pri_dataset.py` / `pri_tokenizer.py` / `rounding.py` | 模型与数据层 |

---

## 大体积资产的托管原则

不要把以下内容入库（已被 `.gitignore` 屏蔽）：

- **Checkpoint / 权重**（`CheckPoint/`、`*.pth`、`*.ckpt`）→ 推荐 GitHub Releases 或 Git LFS。
- **数据集**（`dataset/{Ground_Truth,Miss,Spurious,Mix}/`、`dataset_random_len/`）→ 外部存储 + 下载脚本。
- **论文 / 实验报告**（`*.pdf`、`*.xlsx`）→ Release 或网盘。
- **日志 / 可视化**（`logs/`、`paper_figures/`、`generated_outputs/`）→ 通常不版本化。

---

## 许可证

本项目采用 [MIT License](LICENSE)。
