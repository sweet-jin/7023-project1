# Project 1 文献调研笔记

**主题**：MiC 模块与预制构件几何测量质量检测（Geometric Measurement of Module and Prefabricated Components for MiC Quality Inspection）

> 更新记录
> - Round 1（2026-09-03）：初步调研，覆盖三个方向的代表性论文（近三年为主）
> - Round 2（2026-09-10）：每个方向补充 5–10 篇，覆盖平整度/垂直度专项、门窗孔洞预埋件、3D 基础模型、开放词汇 3D 分割、数据集；更正 Teng→Deng 2026

---

## 调研覆盖的三个方向

1. 深度 2D 检测/分割（含视觉基础模型在土木工程中的应用）
2. 深度 3D 检测/分割（点云实例/语义分割，3D 基础模型）
3. 几何测量与预制构件质量检测（本项目核心应用领域）

---

## 方向一：深度 2D 检测/分割

### 1.1 基础模型（顶会）

| 论文 | 会议/期刊 | 年份 | 要点 |
|------|-----------|------|------|
| Kirillov, A., et al. "Segment Anything." | ICCV | 2023 | 第一个提示式分割基础模型，SA-1B 数据引擎，零样本泛化强 |
| Ravi, N., et al. "SAM 2: Segment Anything in Images and Videos." | ICLR | 2025 | 图像+视频统一分割，流式记忆模块，实时 |
| Liu, S., et al. "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection." | ECCV | 2024 | 开放词汇检测标杆，文本提示定位任意物体（Grounding DINO 1.5 Pro / 1.6 Pro 为后续升级版，2024–2025） |
| Xiao, B., et al. "Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks." | CVPR | 2024 | 统一视觉任务表示，支持检测/分割/描述 |
| Oquab, M., et al. "DINOv2: Learning Robust Visual Features without Supervision." | ICLR | 2024 | 自监督视觉特征，作为下游任务骨干/特征提取器 |
| Cheng, T., et al. "YOLO-World: Real-Time Open-Vocabulary Object Detection." | CVPR | 2024 | 开放词汇+实时，适合工厂检测部署 |
| Zhao, Y., et al. "DETRs Beat YOLOs on Real-time Object Detection (RT-DETR)." | CVPR | 2024 | 实时端到端检测器，无需 NMS |
| Jiang, Q., Li, F., Zeng, Z., Ren, T., Liu, S., & Zhang, L. "T-Rex2: Towards Generic Object Detection via Text-Visual Prompt Synergy." | ECCV | 2024 | 文本+视觉提示协同的开放集检测，对罕见类别（预埋件、孔洞）友好 |

### 1.2 土木工程应用（顶刊）

| 论文 | 期刊 | 年份 | 要点 |
|------|------|------|------|
| Ye, Z., Lovell, L., Faramarzi, A., & Ninić, J. "SAM-based instance segmentation models for the automation of structural damage detection." | Advanced Engineering Informatics, 62, 102826 | 2024 | 用 LoRA 等 PEFT 适配 SAM 做结构损伤实例分割 |
| Ge, K., et al. "Fine-tuning vision foundation model for crack segmentation in civil infrastructures." | Construction and Building Materials, 431, 136573 | 2024 | 微调视觉基础模型做裂缝分割 |
| Xu, Y., Zhang, C., & Li, H. "Transformer-based large vision model for universal structural damage segmentation." | Automation in Construction, 176, 106256 | 2025 | Transformer 大模型做通用结构损伤分割 |
| Chai, C., et al. "Corrosion SAM: Adapting Segment Anything Model with Parameter-Efficient Fine-Tuning for Structural Corrosion Inspection." | arXiv（2024） | 2024 | 参数高效微调 SAM 做锈蚀检测 |
| Owor, N. J., et al. "PaveSAM – Segment anything for pavement distress." | Road Materials and Pavement Design, 26(3) | 2025 | SAM 用于路面病害 |
| Deng, P., Yao, J., Li, C., Wang, S., Li, X., Ojha, V., & He, X. "3D Modeling and Automated Measurement of Concrete Cracks via Segment Anything Refinement and Visual Inertial LiDAR Fusion." | Computer-Aided Civil and Infrastructure Engineering, 45, 100019 | 2026 | **高度相关**：SAM 裂缝感知提示精化 + 视觉-惯性-LiDAR 融合，3D 重建后亚毫米级自动测量，零样本 IoU 提升 6%；开源代码 CrackSeg（github.com/XR-Lee/CrackSeg）。注：Round 1 中误记为 "Teng et al."，以此为准 |

---

## 方向二：深度 3D 检测/分割

### 2.1 3D 实例/语义分割（顶会）

| 论文 | 会议 | 年份 | 要点 |
|------|------|------|------|
| Schult, J., et al. "Mask3D: Mask Transformer for 3D Semantic Instance Segmentation." | ICRA | 2023 | Transformer 式 3D 实例分割，ScanNet200 上 SOTA，常作基线 |
| Wu, X., et al. "Point Transformer V3 (PTv3): Simpler, Faster, Stronger." | ICCV | 2023 | 点云骨干网络，稀疏卷积+Transformer，性能强 |
| Takmaz, A., et al. "OpenMask3D: Open-Vocabulary 3D Instance Segmentation." | NeurIPS | 2023 | 开放词汇 3D 实例分割，用 2D 特征蒸馏到 3D |
| Liu, Y., et al. "Segment Any Point Cloud Sequences by Distilling Vision Foundation Models (Seal)." | NeurIPS | 2023 | 蒸馏 SAM 等 2D 基础模型到点云序列分割 |
| Deng, J., et al. "QueryFormer: Query Refinement Transformer for 3D Instance Segmentation." | ICCV | 2023 | 查询精化 |
| Shin, S., et al. "Spherical Mask: Coarse-to-Fine 3D Point Cloud Instance Segmentation with Spherical Representation." | CVPR | 2024 | 球坐标表示，解决 AABB 过大与误差传播，ScanNetV2/S3DIS 领先 |
| Zhang, Z., et al. "FreePoint: Unsupervised Point Cloud Instance Segmentation." | CVPR | 2024 | 无监督点云实例分割 |
| Roh, et al. "Insightful Instance Features for 3D Instance Segmentation." | CVPR | 2025 | 实例特征设计 |
| Lu, J., & Deng, J. "Relation3D: Enhancing Relation Modeling for Point Cloud Instance Segmentation." | CVPR | 2025 | 关系建模 |
| An, Z., et al. "Generalized Few-shot 3D Point Cloud Segmentation with Vision-Language Model (GFS-VL)." | CVPR | 2025 | 视觉语言模型驱动的少样本 3D 分割，含 RegionPLC |
| Li, et al. "SAS: Segment Any 3D Scene with Integrated 2D Priors." | ICCV | 2025 | 融合 2D 先验做任意 3D 场景分割 |
| Zhou, Y., et al. "Point-SAM: Promptable 3D Segmentation Model for Point Clouds." | arXiv:2406.17741 | 2024 | 提示式 3D 分割基础模型，SAM 范式迁移到点云，2D 知识蒸馏 |
| Lee, J., et al. "Mosaic3D: Foundation Dataset and Model for Open-Vocabulary 3D Segmentation." | CVPR | 2025 | NVIDIA 开放词汇 3D 分割基础模型，5.6M mask-text 对，逐点语言对齐+掩码解码器 |
| Boudjoghra, M. E. A., et al. "Open-YOLO 3D: Towards Fast and Accurate Open-Vocabulary 3D Instance Segmentation." | ICLR | 2025 | 快速开放词汇 3D 实例分割 |
| Piekenbrinck, J., et al. "OpenSplat3D: Open-Vocabulary 3D Instance Segmentation using Gaussian Splatting." | CVPR | 2025 | 3DGS 表征的开放词汇实例分割 |
| Rusnak, A. M., & Kaplan, F. "HAECcity: Open-Vocabulary Scene Understanding of City-Scale Point Clouds with Superpoint Graph Clustering." | CVPR | 2025 | 超点图聚类的城市级开放词汇 |
| Su, H., et al. "S²AM3D: Scale-controllable Part Segmentation of 3D Point Clouds." | CVPR | 2026 | **部件级**分割 + 10 万级部件数据集，尺度可控提示解码器，对模块内构件（门/窗/预埋件）级分割极具参考价值 |

### 2.2 室内/建筑点云应用

| 论文 | 期刊 | 年份 | 要点 |
|------|------|------|------|
| Brunklaus, M., Kellner, M., & Reiterer, A. "Three-Dimensional Instance Segmentation of Rooms in Indoor Building Point Clouds Using Mask3D." | Remote Sensing, 17(7), 1124 | 2025 | Mask3D 用于建筑室内房间实例分割，验证其建筑场景迁移能力 |
| Perez-Perez, Y., Golparvar-Fard, M., & El-Rayes, K. "Scan2BIM-NET: Deep Learning Method for Segmentation of Point Clouds for Scan-to-BIM." | Journal of Construction Engineering and Management, 147(9), 04021107 | 2021 | 室内点云分割做 Scan-to-BIM 的代表性基线（稍早，领域经典） |
| 佚名综述 "Indoor scan-to-BIM workflows: Progress, challenges, and future directions (2014–2024)." | Automation in Construction | 2025 | 109 篇文献的室内 Scan-to-BIM 综述，门窗洞口检测背景 |

---

## 方向三：几何测量与预制构件质量检测（核心应用文献）

### 3.1 综述（必读）

| 论文 | 期刊 | 年份 | 要点 |
|------|------|------|------|
| Ma, Z., Liu, Y., & Li, J. "Review on automated quality inspection of precast concrete components." | Automation in Construction, 150, 104828 | 2023 | 预制构件自动质检全景综述，项目报告"相关工作"章节的核心参考 |
| 另有 2023 年 MiC 数字检测技术综述（PMC 开放获取，期刊待核实） | — | 2023 | 覆盖 MiC 行业检测技术（BIM、RFID、深度学习等） |

### 3.2 点云几何测量/尺寸质检（重点方法论文）

| 论文 | 期刊 | 年份 | 要点 |
|------|------|------|------|
| Shu, et al. "Point cloud-based dimensional quality assessment of precast concrete components using deep learning." | Journal of Building Engineering, 70, 106391 | 2023 | 点云+深度学习做预制构件尺寸质检，被引 65+，经典基线 |
| Ren, H., Fu, Z., Zhang, Z., Ji, B., & Wang, Z. "Geometric quality inspection of precast concrete components assisted by point cloud data." | Journal of Building Engineering, 108, 112927 | 2025 | 点云辅助的预制构件几何质检（桥梁预制场景） |
| Wan, H.-P., et al. "An Efficient Three-Dimensional Point Cloud Segmentation Method for the Dimensional Quality Assessment of Precast Concrete Components Utilizing Multiview Information Fusion." | Journal of Computing in Civil Engineering (ASCE), 39(3), 04025028 | 2025 | **高度相关**：多视角信息融合的 3D 点云分割做 PC 构件尺寸质检 |
| Zhao, W. J., Jiang, Y., Liu, Y. Y., & Shu, J. P. "Automated recognition and measurement based on three-dimensional point clouds to connect precast concrete components." | Automation in Construction, 133, 104000 | 2022 | 点云自动识别与测量连接预制构件（稍早但高被引） |
| Tan, Y., et al. "A terrestrial laser scanning-based method for indoor geometric quality measurement." | Remote Sensing, 16(1), 59 | 2024 | TLS 室内几何质量测量方法 |
| Xu, Y., et al. "Two-stage terrestrial laser scan planning framework for geometric measurement of civil infrastructures." | Measurement, 242, 115785 | 2025 | TLS 扫描规划（数据采集端优化） |
| Tan, Y., et al. "Automated geometric quality inspection for modular boxes using BIM and LiDAR." | Automation in Construction | 2024 | **与 MiC 场景几乎一致**：BIM+LiDAR 做模块箱体自动几何质检，必读 |
| Shu, J., et al. "Point cloud and machine learning-based automated recognition and measurement of corrugated pipes and rebars for large precast concrete beams." | Automation in Construction | 2024 | 点云+ML 识别测量预制梁的波纹管与钢筋（预埋件测量） |
| Li, F., et al. "Point cloud data-based edge detection of precast concrete components for dimensional quality assessment using self-attention mechanisms." | Journal of Building Engineering | 2022 | 自注意力点云边缘检测做尺寸质检 |
| Liu, J., et al. "Dimensional accuracy and structural performance assessment of spatial structure components using 3D laser scanning." | Advanced Engineering Informatics | 2025 | 3D 激光扫描评估空间结构构件尺寸精度 |
| Tran, H., et al. "A digital twin approach for geometric quality assessment of as-built prefabricated façades." | Automation in Construction（待核实） | 2024 | 数字孪生做预制外立面几何质量评估 |
| Ren, H., et al. "Geometric quality inspection of steel structures assisted by point cloud data." | Measurement, 250, 117160 | 2025 | 同第一作者的点云钢结构几何质检（与 JBE 2025 方法族同源） |

### 3.3 平整度/平面度测量

| 论文 | 期刊/会议 | 年份 | 要点 |
|------|-----------|------|------|
| Cao, Y., et al. "Towards automatic flatness quality assessment for building indoor acceptance via terrestrial laser scanning." | Measurement, 203, 111862 | 2022 | TLS 室内验收平整度自动评估（稍早，奠基性） |
| Bosché, F., & Guenet, E. "Scan-vs-BIM slab flatness control." | ISARC | 近年 | Scan-vs-BIM 做楼板平整度控制 |
| Li, F., & Kim, M.-K. "Surface flatness inspection with mirror-aided laser scanning." | ISARC | 近年 | 镜面辅助激光扫描改善远距离平整度测量精度 |
| Li, D., et al. "A deep learning-based indoor acceptance system for assessment on flatness and verticality quality of concrete surfaces." | Journal of Building Engineering, 51, 104284 | 2022 | **深度学习 + 平整度/垂直度双指标**室内验收，测量头设计参考 |
| Kim, M. K., et al. "A mirror-aided laser scanning system for geometric quality inspection of side surfaces of precast concrete elements." | Automation in Construction | 2021 | 镜面辅助扫描测量预制构件侧面 |
| Kim, M. K., Wang, Q., & Li, H. "Non-contact sensing based geometric quality assessment of buildings and civil structures: A review." | Automation in Construction, 100, 163–179 | 2019 | 非接触几何质量评估经典综述（方法谱系） |
| 佚名 "Geometric Inspection in 3D Concrete Manufacturing." | ISPRS Geospatial Week 2025 | 2025 | 3D 混凝土打印/制造后几何检测，覆盖 SLS/摄影测量 |

### 3.4 其他相关

| 论文 | 期刊/会议 | 年份 | 要点 |
|------|-----------|------|------|
| Liang, Y., & Xu, Z. "Intelligent inspection of appearance quality for precast concrete components based on improved YOLO model and multi-source data." | Engineering, Construction and Architectural Management, 32(3), 1691–1714 | 2025 | 改进 YOLO + 多源数据做预制构件外观质检 |
| Wang, Q., et al. "3D tensor-based point cloud and image fusion for robust detection and measurement of rail surface defects." | Automation in Construction, 161, 105342 | 2024 | 点云-图像融合的缺陷检测与测量（方法可借鉴） |
| Dong, Z., & Lu, W. "A Case Study on Applying Machine Learning on Blockchain (MLOB) for Modular Integrated Construction Inspections." | ISARC | 2025 | MiC 检测（HKU 团队，与课程背景直接相关） |

### 3.5 数据集与基准

| 数据集/基准 | 来源 | 年份 | 要点 |
|-------------|------|------|------|
| CI3LAB MiC 数据集 | github.com/CI3LAB | 待发布 | 课程 Project 1 官方数据集；截至 2026-09-10 尚未发布，现有仓库为 Text2MBL、MGF-PC（多模态检索）等；仓库 "CIVIL 7023 Course information" 需持续跟踪 |
| ConSite + SiteNet | Computer-Aided Civil and Infrastructure Engineering, DOI 10.1016/j.cacaie.2026.100050 | 2026 | 建筑工地点云语义分割数据集（13 类）+ SiteNet 模型 |
| STPLS3D | 公开（stpls3d.com） | 2022 | 合成+真实航拍建筑场景点云（16 km²、19 类），含实例标注，可作预训练/迁移 |
| CVPR 2024 Scan-to-BIM Challenge | CVPR 2024 Workshop | 2024 | 8 栋建筑 16 层室内点云（LAZ），门窗洞口检测基准 |
| ScanNetV2 / S3DIS | 公开 | 2017/2016 | 3D 分割通用基准（Mask3D/Spherical Mask 等评测用） |

---

## 待补充调研方向（后续 Round）

- [x] 平整度/垂直度/对角线测量的专项方法（Round 2 已补：Li 2022 JBE、Kim 2021、Cao 2022 等）
- [x] 门窗洞口、孔洞、预埋件（embedded parts）检测与测量（Round 2 已补：Shu 2024 AutoCon、Scan-to-BIM 综述）
- [ ] 单目/RGB-D 深度估计做几何测量（monocular measurement）——下轮
- [ ] 6D 位姿估计用于构件定位装配——下轮（已初步浏览 ISARC 2025 相关工作）
- [x] 3D 基础模型最新进展（Point-SAM、Mosaic3D、S²AM3D 等）
- [x] 开放词汇 3D 分割在建筑/MiC 场景的应用（Open-YOLO 3D、OpenSplat3D、HAECcity）
- [x] 可用数据集与基准（ConSite、STPLS3D、Scan-to-BIM Challenge、ScanNet/S3DIS）
- [x] CACAIE 等顶刊近三年相关论文（Deng 2026、ConSite 2026）

## 备注

- Deng et al. 2026（CACAIE）信息已核实：45 卷，100019，代码 CrackSeg。
- 部分条目（如 Tran et al. 2024 期刊名）为检索信息，写报告引用前需逐一在官方来源核实卷期页码。
- 核心文献三级清单（Tier 1/2/3，20–30 篇）见 docs/related_works.md。
