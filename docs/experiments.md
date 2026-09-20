# Project 1 实验记录

> 实验协议 + 已完成的本地验证结果 + 待执行的 GPU 全量实验
> 更新：2026-09-11

## 1. 实验协议

**数据**：合成 MiC 基准（`datasets/synthetic.py`），默认配置：
- 训练 80 / 验证 16 个样本，每样本 12,000 点
- 6 类模块 × 6 类主任务循环采样；传感器噪声 σ=1mm、墙面起伏 ≤1.5mm、墙倾角 ≤0.3°、预埋件遮挡、随机丢失
- 解析真值（mm 级可评测）

**对比基线**：
| 模式 | 含义 |
|------|------|
| `geom` | 确定性几何 + **GT 掩码**（"分割→拟合"范式上界，对应 Shu 2023 / Wan 2025 类管线） |
| `geom-pred` | 确定性几何 + **模型预测掩码**（完整两阶段管线 = 文献主流范式） |
| `learned` | 端到端 TADH-MiC（软掩码池化 + 学习式回归） |

**指标**：分任务 MAE/RMSE（长度槽位 mm、垂直度 deg）、分割准确率、任务分类准确率、容差合规率。
**消融**：w/o 任务路由（替换为共享头）、w/o 联合训练（冻结分割头后仅训测量头）、w/o 2D 先验（默认即无，预留）。

## 2. 已完成的本地验证（CPU，冒烟规模）

### 2.1 几何基线（`eval.py --mode geom`，默认配置验证集 16 样本）

| 组 | 样本数 | MAE (mm) | RMSE (mm) |
|----|--------|----------|-----------|
| module（长/宽/高/平整度/垂直度） | 16 | 9.4 | 11.2 |
| door | 13 | 9.6 | 13.2 |
| window | 11 | 6.3 | 8.7 |
| hole | 13 | 2.2 | 5.6 |
| embedded | 14 | 3.5 | 3.7 |

> 说明：module 组误差含平整度（噪声下限 ~4mm）与倾斜引起的 ~15mm 系统偏差；垂直度误差 <0.01°。
> 该结果即"分割→拟合"范式的上界，也是 TADH-MiC 需要对比的对象。

### 2.2 冒烟训练（24 训练 / 12 验证，6,000 点，15 epoch，warmup 4，CPU）

| epoch | seg_acc | task_acc | meas_mae |
|-------|---------|----------|----------|
| 1 | 0.52 | 0.17 | 1002 |
| 5 | 0.68 | 0.17 | 287 |
| 10 | 0.82 | 0.25 | 191 |
| 15 | 0.81 | 0.17 | 177 |

- 分割头收敛良好（81%）；测量头随联合训练持续下降但未收敛（样本量 24 过小）
- 任务路由在 24 样本下接近随机（1/6=0.17），需全量数据
- `learned` / `geom-pred` 模式在冒烟规模下明显欠拟合（module 965 / 297 mm），**仅验证管线可用性，不作结论**

## 3. 待执行：GPU 全量实验

```bash
cd project1
# 全量训练（有 GPU 时默认自动用 cuda）
python train.py --config configs/default.yaml --epochs 60 --out out/full

# 三种模式评测
python eval.py --config configs/default.yaml --mode geom --out out/eval_geom.json
python eval.py --config configs/default.yaml --checkpoint out/full/best.pt --mode geom-pred --out out/eval_geompred.json
python eval.py --config configs/default.yaml --checkpoint out/full/best.pt --mode learned --out out/eval_learned.json

# 消融（训练时改 config 或加 --ablation 参数，见 framework_design.md）
```

**预期与判读**：
- `learned` 在分布内应逼近或超越 `geom-pred`（联合优化缓解分割误差传播）；`geom`（GT 掩码上界）是最强参照
- 若 `learned` 未超越 `geom-pred`：如实报告，分析原因（回归任务样本效率、遮挡下拟合失效），讨论"学习式 + 几何精修"混合方案（框架预留推断期精修接口）
- 报告 4.3 消融表与 4.4 噪声/遮挡稳健性实验在 GPU 全量数据上执行

## 4. 已修复的工程问题记录（供报告 Robustness 章节引用）

1. 遮挡仿真需同步过滤标签（曾导致点-标签错位）
2. 墙面面片采样原点须在角点（曾导致半房间偏移）
3. 倾斜模拟须绕水平切向轴（绕竖轴是"偏航"而非"倾斜"）
4. PCA 主轴在矩形宽高相近时对采样噪声敏感 → 门窗尺寸改用轴对齐包围盒
5. 多孔洞/多预埋件需簇级测量 + 生成端最小间距约束（圆不重叠）
6. 孔洞圆拟合需"三轴投影择优 + 内点率校验"（小孔深度主导方差时 PCA 选错投影面）
7. 预埋件凸出测量需按密度峰选墙面带（墙角处 RANSAC 会被垂直墙劫持）
