# `docs/figures/`

本目录收录训练产出的**可视化结果图**（用于论文、算法分析、对比），**是版本化的**，以便复现与讨论。

原始产出位置是运行时的 `CheckPoint/<experiment>/visuals/` 及
`BaseLine/<name>/CheckPoint/<scene>/visuals/`。这些运行目录本身仍然被 `.gitignore` 忽略（包含
大体积 `.pth` 权重、日志、临时文件），因此我们把**只保留结果图**的一份拷贝放到这里。

## 目录结构

```
docs/figures/
├── <experiment_name>/              # 对应 CheckPoint/<experiment>/visuals/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── epoch_<NNNN>_codebook.png
│   ├── epoch_<NNNN>_diffusion_denoise.png
│   └── ...
└── baselines/
    ├── DenoisingAE/<scene>/Miss_samples.png
    ├── RNNPrediction/<scene>/Miss_samples.png
    └── SemanticCoding/Miss_samples.png
```

## 约定

- 只放 **结果图 / 论文图**（PNG、SVG、PDF 示意图）。
- **不要**把 `.pth` / `.ckpt` / 原始 `log.txt` / `result.json` 放进来——它们属于运行目录
  `CheckPoint/`，继续保持被 `.gitignore` 忽略。
- 新增实验时，请把 `CheckPoint/<experiment>/visuals/*.png` 拷贝到
  `docs/figures/<experiment>/` 并提交。
