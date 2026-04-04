# PRI-DiffuSeq: 针对雷达侦查 PRI 序列重构的改版

这套代码是把 DiffuSeq 的“条件扩散 + 非自回归 seq2seq”思想改成 **PRI 序列重构** 任务专用版本。

## 任务映射

原始 DiffuSeq:
- source = 输入文本
- target = 输出文本
- 扩散时固定 source，只生成 target

PRI 重构版本:
- source = 观测到的非理想 PRI 序列
- target = 完整无误的 PRI 序列
- 扩散时固定 source，只重构 target

这和你的任务高度一致：
**把受噪声、缺失、抖动、伪脉冲影响的观测 PRI 序列，恢复为完整、正确的 PRI 序列。**

## 代码文件

- `pri_tokenizer.py`
  - 把连续 PRI 数值按最近原型离散化为 token
  - 原型由理想 PRI 先验或理想序列提取而来
  - 便于沿用 DiffuSeq 的 embedding + rounding 机制

- `pri_dataset.py`
  - 构造 DiffuSeq 风格的训练样本
  - `input_ids = [observed] + [SEP] + [clean]`
  - `input_mask = 0/1` 用于指定哪些位置固定、哪些位置要生成

- `model.py`
  - 核心模型
  - `DenoiseTransformer`: 非自回归 Transformer 去噪器
  - `GaussianDiffusion1D`: 1D 序列扩散采样
  - `PRIDiffuSeq`: 训练、验证、重构接口封装

- `rounding.py`
  - 将连续隐表示映射回最近的离散 token / PRI bin

- `example_pri_usage.py`
  - 演示如何训练和推理

## 你的 dataloader 需要满足什么

训练/验证 batch 至少包含：

```python
{
    'input_ids': LongTensor[B, L],
    'input_mask': LongTensor[B, L],
}
```

推荐直接使用 `PRIDiffuSeqDataset`，它会帮你构建好这个格式。

## 最关键的改动点

### 1. 文本 tokenizer -> PRI quantizer
原项目是文本 token。
这里改成 **PRI 原型编码**：
- 先根据理想 PRI 先验构建原型集合
- 把连续 PRI 值映射到最近原型
- 每个原型对应一个 token id
- 这样就能直接做 embedding、diffusion、rounding

### 2. 文本 source-target -> 观测 PRI 到干净 PRI
训练时：
- source: 非理想观测序列
- target: 对应的干净真值序列

推理时：
- 给定 source
- 采样生成 target

### 3. rounding 改为数值 token rounding
原项目 rounding 是把隐状态映射回最近词向量。
这里保留同样逻辑：
- 把隐状态映射到最近的 PRI token embedding
- 再反解码为对应原型 PRI 值

## 你自己的数据怎么接

### 方案 A：直接用这套 Dataset
你只需要把每条样本整理成：

```python
PRISample(
    observed_pri=[...],
    clean_pri=[...],
)
```

### 方案 B：你已经有自己的 dataloader
那你只要保证输出：

```python
batch = {
    'input_ids': ...,   # [observed] + [SEP] + [clean]
    'input_mask': ...,  # source/sep=0, target/pad=1
}
```

然后直接：

```python
stats = model.compute_loss(batch['input_ids'], batch['input_mask'])
out = model.reconstruct(batch['input_ids'], batch['input_mask'])
```

## 进一步建议

如果你的 PRI 序列存在这些情况，可以继续增强：

1. **缺失脉冲 / 虚假脉冲**
   - 在 observed 侧做更真实的数据增强

2. **多工作模式切换**
   - 把模式标签作为额外条件嵌入输入模型

3. **连续数值精度要求更高**
   - 增大 `num_bins`
   - 或在 token 预测之后加一个小的回归 refine head

4. **长度不一致**
   - 当前版本适合同长度重构
   - 若要做插补/删除，可把 target 长度放宽并在后处理中对齐
