# TADH-MiC — Project 1 双头框架实现

MiC 模块与预制构件几何测量质量检测（CIVL 7023 Project 1）的**任务感知双头深度学习框架**代码。

## 目录结构

```
project1/
├── configs/default.yaml      # 全部超参（数据范围/模型/训练/容差）
├── datasets/synthetic.py     # 合成 MiC 模块点云生成器（含解析真值）
├── geometry/
│   ├── fitting.py            # 平面拟合/RANSAC/PCA包围盒/圆拟合/聚类
│   └── measures.py           # 测量槽位定义 + 确定性测量（几何基线）
├── models/
│   ├── backbone.py           # PointNet++ 风格层级编码器（可换 PTv3）
│   ├── heads.py              # TaskRouter + SegHead + MeasureHead + 损失
│   ├── vision_prior.py       # 可选 2D 基础模型先验（SAM 接口，默认关）
│   └── dualhead.py           # TADH-MiC 整体网络
├── train.py                  # 训练（课程学习：先分割后联合）
├── eval.py                   # 评测（geom / geom-pred / learned 三模式）
└── visualize.py              # 导出 PLY 可视化
```

## 快速开始

```bash
# 环境（本仓库已建好 .venv，numpy/torch 已装）
source ../.venv/bin/activate

# 冒烟测试：小数据快速跑通
python train.py --config configs/default.yaml --n-train 16 --n-val 8 \
    --n-points 4000 --epochs 3 --out out/smoke

# 评测三种模式
python eval.py --config configs/default.yaml --mode geom --out out/eval_geom.json
python eval.py --config configs/default.yaml --checkpoint out/smoke/best.pt --mode geom-pred
python eval.py --config configs/default.yaml --checkpoint out/smoke/best.pt --mode learned

# 可视化
python visualize.py --config configs/default.yaml --index 0 \
    --checkpoint out/smoke/best.pt --out out/vis
```

## 三种评测模式的含义

| 模式 | 含义 |
|------|------|
| `geom` | 确定性几何方法 + GT 分割掩码（"分割→拟合"范式的上界，对应 Shu 2023 / Wan 2025 类管线） |
| `geom-pred` | 确定性几何方法 + 模型预测掩码（完整两阶段管线，即文献主流范式） |
| `learned` | 端到端 TADH-MiC（软掩码池化 + 学习式回归，无几何后处理） |

## 设计要点

- **任务感知路由**：6 类任务（尺寸/平整度/垂直度/洞口/孔洞/预埋件）的软路由，FiLM 调制双头
- **双头**：Head 1 逐点语义分割 → Head 2 软掩码池化区域特征回归几何量（米制，评测转 mm）
- **端到端**：分割与测量联合训练，缓解两阶段误差传播（框架动机见 docs/framework_design.md）
- **可插拔 2D 基础模型**：`use_vision_prior: true` 预留 SAM 先验接口（需图像与相机模型，合成管线未启用）

## 已知限制

- 训练 batch=1（PointNet++ 分组实现按样本处理）；GPU 上可加梯度累积
- 几何基线假设房间近似轴对齐（合成数据成立；CI3LAB 真实数据需先 PCA 对齐）
- CI3LAB 官方数据集发布后：将 `datasets/` 增加对应加载器，槽位定义无需改动
