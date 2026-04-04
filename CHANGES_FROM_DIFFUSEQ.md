# 针对 PRI 重构，相比原始 DiffuSeq 的主要修改

## 一、任务语义变化

原项目做的是文本 seq2seq 生成。
这里改成：
- 输入：观测到的非理想 PRI 序列
- 输出：对应的完整干净 PRI 序列

因此最核心的映射是：
- `src` -> observed PRI
- `trg` -> clean PRI

## 二、数据层修改

原项目：
- 从 jsonl 中读入 `src` / `trg` 文本
- 用 tokenizer 编码

现在：
- 输入直接是数值 PRI 序列
- 通过 `PRIQuantizer` 做离散化
- 保持 `src + SEP + trg` 的组织方式不变

## 三、模型层修改

原项目依赖 HuggingFace BERT Encoder。
这里为了便于你快速改任务和调试：
- 去掉了外部 BERT 依赖
- 改成纯 PyTorch `TransformerEncoder`
- 保留了“embedding + timestep + transformer + diffusion”的主结构

## 四、训练目标修改

训练时：
- source 段保持不变
- target 段加入噪声后去噪
- loss 主要计算 target 段

损失包括：
- epsilon 预测 MSE
- token 交叉熵约束

## 五、rounding 修改

原项目 rounding 用于文本 token 恢复。
这里改成：
- 隐状态 -> 最近 embedding token
- token -> PRI bin center

这相当于把连续去噪结果重新投影回可解释的 PRI 数值空间。

## 六、你后续最可能还要改的地方

### 1. quantizer 范围
你需要把：
- `min_value`
- `max_value`
- `num_bins`

改成你的雷达 PRI 量级。

### 2. 数据增强
建议把真实非理想因素注入 observed：
- 漏检
- 假脉冲
- 抖动
- 模式切换
- 脉冲错位

### 3. 输出后处理
如果你需要恢复成严格的 PRI 模式序列，可以在解码后增加：
- 平滑
- 模式约束
- 聚类/模板匹配
- 动态规划对齐
