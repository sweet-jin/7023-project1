# Project 1 Report Draft

> CIVL 7023 Industrialized Construction with Automation and Robotics
> Format: Title ≤20 words | Abstract 150–250 words | ≤5 keywords | Body ≤10,000 words
> APA references | Times New Roman 12pt, single spacing, 2.5 cm margins
> Status: draft — Results/Discussion to be finalized after full experiments

## Title

Task-Aware Dual-Head Deep Learning Framework for Geometric Quality Inspection of Modular Integrated Construction Modules

## Abstract

*(draft, target 150–250 words)*

Modular Integrated Construction (MiC) delivers fully finished modules from factory to site, but its promise of high quality depends on rigorous geometric inspection of as-built modules and prefabricated components—dimensions, flatness, perpendicularity, openings, holes, and embedded parts. Existing automated approaches follow a split-then-fit paradigm: a segmentation model first labels regions, then a deterministic geometric fitting stage recovers measurements. This two-stage design propagates segmentation errors into measurements, lacks task awareness across heterogeneous inspection items, and under-uses the representational power of foundation models. This project proposes TADH-MiC, a task-aware dual-head network that jointly optimizes region segmentation and geometric regression in an end-to-end manner. A task-aware router predicts the primary inspection task and modulates both heads through feature-wise affine conditioning. Head 1 performs point-wise semantic segmentation; Head 2 pools region features through soft masks and regresses measurement slots (length, width, height, flatness, perpendicularity, opening sizes, hole diameters, and embedded-part protrusion). Because the official CI3LAB MiC dataset was not yet released at the time of writing, the framework is trained and evaluated on a procedurally generated synthetic benchmark of six MiC module types with analytic ground truth, sensor noise, wall waviness, tilt, and single-station occlusion. The learned dual-head framework achieves a mean absolute measurement error of X.X mm, outperforming the geometric baseline by Y%, with ablations confirming the contribution of task-aware routing and joint optimization. The framework will be re-validated on the CI3LAB dataset upon release.

## Keywords

Modular integrated construction; geometric quality inspection; point cloud deep learning; multi-task learning; automated measurement

---

# Main Body

## 1. Introduction

MiC is a game-changing construction approach in which free-standing integrated modules are manufactured off-site and assembled on-site, improving productivity, quality, safety, and sustainability (Pan & Hon, 2018). Hong Kong has mandated MiC adoption in public housing, making factory-side quality control a bottleneck: module tolerances of a few millimetres determine whether on-site assembly succeeds. Traditional inspection relies on tape measures, spirit levels, and manual checklists, which are slow, sparse, and error-prone (Kim, Wang, & Li, 2019).

Laser scanning and structure light sensors now capture as-built modules as dense point clouds, opening the door to automated geometric quality inspection. The dominant paradigm in the literature is split-then-fit: deep segmentation of components followed by deterministic geometric fitting (RANSAC planes, bounding boxes, circle fits) to recover measurements (Shu et al., 2023; Wan et al., 2025; Deng et al., 2026). This paradigm has three limitations:

1. **Error propagation**: segmentation errors flow directly into the fitting stage with no joint optimization;
2. **No task awareness**: heterogeneous inspection items (dimension, flatness, perpendicularity, openings, holes, embedded parts) are handled by one-size-fits-all pipelines;
3. **Under-used foundation models**: 2D foundation models such as SAM are largely limited to crack/damage segmentation in civil applications (Ye et al., 2024; Ge et al., 2024), with little integration into 3D geometric measurement.

This project addresses these gaps with a **task-aware dual-head framework (TADH-MiC)**: a point-cloud backbone shared by (i) a task-aware router that predicts the primary inspection task and modulates both heads, (ii) Head 1, a semantic segmentation head, and (iii) Head 2, a measurement head that pools region features via soft masks and directly regresses measurement slots. The framework is trained end-to-end, and the design is validated on a synthetic MiC benchmark with analytic ground truth while the CI3LAB dataset is pending release.

Contributions: (1) an end-to-end dual-head architecture that jointly optimizes segmentation and measurement, mitigating the split-then-fit error propagation; (2) a task-aware routing mechanism enabling a single network to serve six inspection tasks; (3) an extensible 2D foundation-model prior branch; (4) a reproducible synthetic MiC benchmark with mm-level ground truth.

## 2. Literature Review

*(draft — will be expanded to ~2,000 words with citations from docs/related_works.md)*

### 2.1 Deep 2D detection and segmentation

Foundation models have transformed segmentation: SAM (Kirillov et al., 2023) introduced promptable segmentation trained on the SA-1B data engine; SAM 2 (Ravi et al., 2025) extended it to streaming video. In open-vocabulary detection, Grounding DINO (Liu et al., 2024) and T-Rex2 (Jiang et al., 2024) enable text/visual-prompt-based localization of rare categories—attractive for embedded parts and openings. In civil engineering, SAM has been adapted via parameter-efficient fine-tuning for structural damage segmentation (Ye et al., 2024) and crack segmentation (Ge et al., 2024); Xu, Zhang, and Li (2025) proposed a transformer-based large vision model for universal damage segmentation. Deng et al. (2026) combined SAM-based crack refinement with visual-inertial-LiDAR fusion for 3D crack modeling and sub-millimetre measurement—the closest prior work to this project, but still split-then-fit and crack-specific.

### 2.2 Deep 3D detection and segmentation

Point Transformer V3 (Wu et al., 2023) is a strong sparse-conv backbone; Mask3D (Schult et al., 2023) and Spherical Mask (Shin et al., 2024) are leading 3D instance segmentation methods. Open-vocabulary 3D segmentation emerged with OpenMask3D (Takmaz et al., 2023), Open-YOLO 3D (Boudjoghra et al., 2025), and the Mosaic3D foundation model (Lee et al., 2025); S²AM3D (Su et al., 2026) targets part-level, scale-controllable segmentation, directly relevant to module-level components. In construction, Mask3D has been applied to indoor room segmentation (Brunklaus et al., 2025), and scan-to-BIM research addresses door/window detection in indoor scans (Perez-Perez et al., 2021; review in Automation in Construction, 2025).

### 2.3 Geometric measurement and precast quality inspection

Non-contact geometric quality assessment was surveyed by Kim, Wang, and Li (2019); Ma, Liu, and Li (2023) reviewed automated quality inspection of precast concrete components. Representative systems: point-cloud-based dimensional quality assessment using deep learning (Shu et al., 2023); multi-view fused 3D segmentation for dimensional assessment (Wan et al., 2025); BIM-and-LiDAR geometric inspection of modular boxes (Tan et al., 2024); deep-learning indoor acceptance for flatness and verticality (Li et al., 2022); recognition and measurement of corrugated pipes and rebars (Shu et al., 2024); and mirror-aided laser scanning for side surfaces (Kim et al., 2021). All follow segmentation-then-fitting pipelines and address single or few inspection items.

### 2.4 Research gap

*(points 1–3 from Introduction, supported by the cited pipeline structure; positioned against Deng et al. 2026 as nearest neighbor)*

## 3. Research Methodology

*(summary of docs/framework_design.md and docs/problem_definition.md; describe TADH-MiC architecture, task taxonomy T1–T6, measurement slot definitions, losses, training strategy)*

### 3.1 Problem formulation
### 3.2 TADH-MiC architecture
### 3.3 Synthetic MiC benchmark and data strategy
### 3.4 Baselines and evaluation metrics

## 4. Results and Discussion

*(to be filled from docs/experiments.md)*

### 4.1 Geometry baseline accuracy (split-then-fit upper bound)
### 4.2 TADH-MiC vs baselines
### 4.3 Ablations: task-aware routing, joint training, 2D prior
### 4.4 Robustness to noise and occlusion
### 4.5 Discussion and limitations

## 5. Conclusions

*(to be filled; summary + future work: CI3LAB validation, PTv3 backbone, open-vocabulary extension)*

## Acknowledgements

We thank Prof. LI Xiao and the teaching assistants of CIVL 7023 for organizing the course and project, and Dr. HONG Jie, Dr. LI Tingtian, and Dr. CHEN Zhe for preparing Project 1 materials.

## References

*(APA list — see docs/related_works.md Tier 1–3; final numbering to be reconciled with in-text citations)*

Kirillov, A., et al. (2023). Segment anything. *ICCV*, 4015–4026.
Ravi, N., et al. (2025). SAM 2: Segment anything in images and videos. *ICLR*.
Liu, S., et al. (2024). Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection. *ECCV*.
Jiang, Q., et al. (2024). T-Rex2: Towards generic object detection via text-visual prompt synergy. *ECCV*.
Ye, Z., Lovell, L., Faramarzi, A., & Ninić, J. (2024). SAM-based instance segmentation models for the automation of structural damage detection. *Advanced Engineering Informatics, 62*, 102826.
Ge, K., et al. (2024). Fine-tuning vision foundation model for crack segmentation in civil infrastructures. *Construction and Building Materials, 431*, 136573.
Xu, Y., Zhang, C., & Li, H. (2025). Transformer-based large vision model for universal structural damage segmentation. *Automation in Construction, 176*, 106256.
Deng, P., Yao, J., Li, C., Wang, S., Li, X., Ojha, V., & He, X. (2026). 3D modeling and automated measurement of concrete cracks via segment anything refinement and visual inertial LiDAR fusion. *Computer-Aided Civil and Infrastructure Engineering, 45*, 100019.
Wu, X., et al. (2023). Point Transformer V3: Simpler, faster, stronger. *ICCV*, 4840–4851.
Schult, J., et al. (2023). Mask3D: Mask transformer for 3D semantic instance segmentation. *ICRA*, 8216–8223.
Shin, S., et al. (2024). Spherical Mask. *CVPR*, 4060–4069.
Takmaz, A., et al. (2023). OpenMask3D. *NeurIPS, 36*.
Boudjoghra, M. E. A., et al. (2025). Open-YOLO 3D. *ICLR*.
Lee, J., et al. (2025). Mosaic3D: Foundation dataset and model for open-vocabulary 3D segmentation. *CVPR*.
Su, H., et al. (2026). S²AM3D: Scale-controllable part segmentation of 3D point clouds. *CVPR*.
Brunklaus, M., et al. (2025). Three-dimensional instance segmentation of rooms in indoor building point clouds using Mask3D. *Remote Sensing, 17*(7), 1124.
Perez-Perez, Y., Golparvar-Fard, M., & El-Rayes, K. (2021). Scan2BIM-NET. *Journal of Construction Engineering and Management, 147*(9), 04021107.
Kim, M. K., Wang, Q., & Li, H. (2019). Non-contact sensing based geometric quality assessment of buildings and civil structures: A review. *Automation in Construction, 100*, 163–179.
Ma, Z., Liu, Y., & Li, J. (2023). Review on automated quality inspection of precast concrete components. *Automation in Construction, 150*, 104828.
Shu, J., et al. (2023). Point cloud-based dimensional quality assessment of precast concrete components using deep learning. *Journal of Building Engineering, 70*, 106391.
Wan, H.-P., et al. (2025). An efficient three-dimensional point cloud segmentation method for the dimensional quality assessment of precast concrete components utilizing multiview information fusion. *Journal of Computing in Civil Engineering, 39*(3), 04025028.
Tan, Y., et al. (2024). Automated geometric quality inspection for modular boxes using BIM and LiDAR. *Automation in Construction*.
Li, D., et al. (2022). A deep learning-based indoor acceptance system for assessment on flatness and verticality quality of concrete surfaces. *Journal of Building Engineering, 51*, 104284.
Shu, J., et al. (2024). Point cloud and machine learning-based automated recognition and measurement of corrugated pipes and rebars for large precast concrete beams. *Automation in Construction*.
Kim, M. K., et al. (2021). A mirror-aided laser scanning system for geometric quality inspection of side surfaces of precast concrete elements. *Automation in Construction*.
Pan, W., & Hon, C. K. (2018). Briefing: Modular integrated construction for high-rise buildings. *Proceedings of the Institution of Civil Engineers—Municipal Engineer*.
