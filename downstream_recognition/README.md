# Downstream: Radar Working Mode Recognition

基于重建后的PRI序列进行雷达工作模式识别的下游任务。

## 架构

**Multi-Scale 1D-CNN + Transformer Encoder**
- 多尺度1D卷积（kernel 3/5/7）提取局部时序特征
- Transformer Encoder（2层，4头自注意力）捕获全局依赖
- 全局平均池化 + MLP分类头

## 分类体系（4类）

| Label | 类型 | Word编号 | PRI调制方式 |
|-------|------|----------|-------------|
| 0 | sliding | 12–15 | 线性递增/递减PRI |
| 1 | stagger | 16–21 | 参差PRI（N值交替） |
| 2 | dwell_switch | 22–28 | 驻留切换PRI |
| 3 | complex_dwell | 29 | 固定PRI + 驻留切换RF/PW |

## 使用

```bash
cd downstream_recognition
python train.py --epochs 60 --n_per_class 2000 --device cuda
```

输出保存至 `results/`：
- `training_curves.png` — 训练曲线
- `confusion_matrix.png` — 混淆矩阵
- `per_class_metrics.png` — 各类精度/召回/F1
- `best_model.pth` — 最优模型权重

## 参考文献风格

模型设计参考近年AI顶会/IEEE期刊中雷达辐射源识别的主流方法：
- Multi-scale CNN + Self-attention (IEEE JSTSP 2023)
- 1D-CNN + Transformer for radar emitter recognition (Radar Conference 2024)
