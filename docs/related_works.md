# Project 1 核心文献清单（Related Works 草稿）

> 用途：项目报告 Literature Review 章节的选文依据与引用清单
> 分级：Tier 1 = 双头框架直接支撑；Tier 2 = 重要基线/对比方法；Tier 3 = 背景与方法参考
> 更新：2026-09-10（Round 1 + Round 2）

## 阅读建议

- **必读精读（方法模仿/对比对象）**：Deng 2026、Tan 2024、Wan 2025、Shu 2023、Li D 2022、Mask3D、PTv3、SAM、Mosaic3D
- **写 Related Works 的主体结构**：2D 检测/分割基础模型 → 3D 检测/分割 → 几何测量与预制构件质检 → 研究空白（gap）

## Tier 1：双头框架直接支撑（15 篇）

### 2D 基础模型与提示式检测/分割
1. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollár, P., & Girshick, R. (2023). Segment anything. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 4015–4026.
2. Ravi, N., Gabeur, V., Hu, Y.-T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., Mintun, E., Pan, J., Alwala, K. V., Carion, N., Wu, C.-Y., Girshick, R., Dollár, P., & Feichtenhofer, C. (2025). SAM 2: Segment anything in images and videos. *The Thirteenth International Conference on Learning Representations (ICLR)*.
3. Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Li, C., Yang, J., Su, H., Zhu, J., & Zhang, L. (2024). Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection. *European Conference on Computer Vision (ECCV)*, 38–55.
4. Jiang, Q., Li, F., Zeng, Z., Ren, T., Liu, S., & Zhang, L. (2024). T-Rex2: Towards generic object detection via text-visual prompt synergy. *European Conference on Computer Vision (ECCV)*, 38–55.
5. Deng, P., Yao, J., Li, C., Wang, S., Li, X., Ojha, V., & He, X. (2026). 3D modeling and automated measurement of concrete cracks via segment anything refinement and visual inertial LiDAR fusion. *Computer-Aided Civil and Infrastructure Engineering, 45*, 100019. （代码：github.com/XR-Lee/CrackSeg）
6. Ye, Z., Lovell, L., Faramarzi, A., & Ninić, J. (2024). SAM-based instance segmentation models for the automation of structural damage detection. *Advanced Engineering Informatics, 62*, 102826.

### 3D 骨干与分割
7. Wu, X., Jiang, L., Wang, P.-S., Liu, Z., Liu, X., Qiao, Y., Ouyang, W., He, T., & Zhao, H. (2023). Point Transformer V3: Simpler, faster, stronger. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 4840–4851.
8. Schult, J., Engelmann, F., Hermans, A., Litany, O., Tang, S., & Leibe, B. (2023). Mask3D: Mask transformer for 3D semantic instance segmentation. *IEEE International Conference on Robotics and Automation (ICRA)*, 8216–8223.
9. Shin, S., Zhou, K., Vankadari, M., Markham, A., & Trigoni, N. (2024). Spherical Mask: Coarse-to-fine 3D point cloud instance segmentation with spherical representation. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 4060–4069.
10. Lee, J., Park, C., Choe, J., Wang, F., Kautz, J., Cho, M., & Choy, C. (2025). Mosaic3D: Foundation dataset and model for open-vocabulary 3D segmentation. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.
11. Su, H., Huang, T., Wan, Z., Wu, X., & Zuo, W. (2026). S²AM3D: Scale-controllable part segmentation of 3D point clouds. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

### 几何测量（本项目核心应用）
12. Wan, H.-P., et al. (2025). An efficient three-dimensional point cloud segmentation method for the dimensional quality assessment of precast concrete components utilizing multiview information fusion. *Journal of Computing in Civil Engineering, 39*(3), 04025028.
13. Shu, J., et al. (2023). Point cloud-based dimensional quality assessment of precast concrete components using deep learning. *Journal of Building Engineering, 70*, 106391.
14. Tan, Y., et al. (2024). Automated geometric quality inspection for modular boxes using BIM and LiDAR. *Automation in Construction*.
15. Li, D., et al. (2022). A deep learning-based indoor acceptance system for assessment on flatness and verticality quality of concrete surfaces. *Journal of Building Engineering, 51*, 104284.

## Tier 2：重要基线与对比方法（11 篇）

16. Ma, Z., Liu, Y., & Li, J. (2023). Review on automated quality inspection of precast concrete components. *Automation in Construction, 150*, 104828.
17. Kim, M. K., Wang, Q., & Li, H. (2019). Non-contact sensing based geometric quality assessment of buildings and civil structures: A review. *Automation in Construction, 100*, 163–179.
18. Ren, H., Fu, Z., Zhang, Z., Ji, B., & Wang, Z. (2025). Geometric quality inspection of precast concrete components assisted by point cloud data. *Journal of Building Engineering, 108*, 112927.
19. Shu, J., et al. (2024). Point cloud and machine learning-based automated recognition and measurement of corrugated pipes and rebars for large precast concrete beams. *Automation in Construction*.
20. Zhao, W. J., Jiang, Y., Liu, Y. Y., & Shu, J. P. (2022). Automated recognition and measurement based on three-dimensional point clouds to connect precast concrete components. *Automation in Construction, 133*, 104000.
21. Li, F., et al. (2022). Point cloud data-based edge detection of precast concrete components for dimensional quality assessment using self-attention mechanisms. *Journal of Building Engineering*.
22. Liu, J., et al. (2025). Dimensional accuracy and structural performance assessment of spatial structure components using 3D laser scanning. *Advanced Engineering Informatics*.
23. Takmaz, A., Fedele, E., Sumner, R. W., Pollefeys, M., Tombari, F., & Engelmann, F. (2023). OpenMask3D: Open-vocabulary 3D instance segmentation. *Advances in Neural Information Processing Systems (NeurIPS), 36*.
24. Boudjoghra, M. E. A., Dai, A., Lahoud, J., Cholakkal, H., Anwer, R. M., Khan, S., & Khan, F. S. (2025). Open-YOLO 3D: Towards fast and accurate open-vocabulary 3D instance segmentation. *The Thirteenth International Conference on Learning Representations (ICLR)*.
25. Xu, Y., Zhang, C., & Li, H. (2025). Transformer-based large vision model for universal structural damage segmentation. *Automation in Construction, 176*, 106256.
26. Ge, K., Wang, C., Guo, Y. T., Tang, Y. S., Hu, Z. Z., & Chen, H. B. (2024). Fine-tuning vision foundation model for crack segmentation in civil infrastructures. *Construction and Building Materials, 431*, 136573.

## Tier 3：背景与方法参考（10 篇）

27. Oquab, M., et al. (2024). DINOv2: Learning robust visual features without supervision. *Transactions on Machine Learning Research*（ICLR 2024 版亦可引用）.
28. Xiao, B., Wu, H., Xu, W., Dai, X., Hu, H., Lu, Y., Zeng, M., Liu, C., & Yuan, L. (2024). Florence-2: Advancing a unified representation for a variety of vision tasks. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 4818–4829.
29. Cheng, T., Song, L., Ge, Y., Liu, W., Wang, X., & Shan, Y. (2024). YOLO-World: Real-time open-vocabulary object detection. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 16901–16911.
30. Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y., & Chen, J. (2024). DETRs beat YOLOs on real-time object detection. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 16965–16974.
31. Zhou, Y., et al. (2024). Point-SAM: Promptable 3D segmentation model for point clouds. *arXiv:2406.17741*.
32. Piekenbrinck, J., Schmidt, C., Hermans, A., Vaskevicius, N., Linder, T., & Leibe, B. (2025). OpenSplat3D: Open-vocabulary 3D instance segmentation using Gaussian splatting. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 5246–5255.
33. Rusnak, A. M., & Kaplan, F. (2025). HAECcity: Open-vocabulary scene understanding of city-scale point clouds with superpoint graph clustering. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 5256–5265.
34. Cao, Y., et al. (2022). Towards automatic flatness quality assessment for building indoor acceptance via terrestrial laser scanning. *Measurement, 203*, 111862.
35. Kim, M. K., et al. (2021). A mirror-aided laser scanning system for geometric quality inspection of side surfaces of precast concrete elements. *Automation in Construction*.
36. Indoor scan-to-BIM workflows: Progress, challenges, and future directions (2014–2024). (2025). *Automation in Construction*.

## 研究空白（Gap，框架设计动机）

基于以上文献梳理，现有方法存在以下缺陷，构成项目框架设计动机：

1. **两阶段割裂**：主流范式为"分割/检测 → 几何拟合测量"串行两阶段（Deng 2026、Shu 2023、Wan 2025 均如此），分割误差直接传播到测量且不可端到端优化；
2. **缺乏任务感知**：不同检测项（尺寸、平整度、垂直度、孔洞、预埋件）共用同一测量管线，无法针对任务选择最优测量策略，而课程要求"task-aware dual-head"；
3. **2D 基础模型与 3D 测量未深度融合**：SAM 等 2D 基础模型在土木应用多止步于裂缝/损伤分割（Ye 2024、Ge 2024、Xu 2025），尚未与 3D 几何测量联合建模；
4. **开放词汇/部件级能力缺失**：现有 PC 质检方法依赖封闭类别集，面对 MiC 模块内多样部件（门窗、开关面板、各类预埋件）泛化不足——Mosaic3D、S²AM3D 的开放词汇与部件级能力尚未引入本领域。

> 注：所有引用在写入报告前需在官方来源逐一核实卷期页码（部分条目来自检索摘要）。
