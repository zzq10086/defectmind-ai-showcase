# DefectMind AI

**基于 AIGC 增强与多模态智能分析的工业表面缺陷检测平台**

常德学院 AIGC 创新创业大赛 · 赛道三「AIGC 赋能工业场景」

**项目展示页**：<https://zzq10086.github.io/defectmind-ai-showcase/>  
**在线质检平台（永久地址）**：<https://zzq10086.github.io/defectmind-ai-showcase/platform/>

项目以 NEU-DET 热轧钢材表面缺陷数据集为基础，用 YOLO26n 与 Stable Diffusion 1.5 Inpainting
建立四组严格对照，回答一个具体问题：**在小样本条件下，扩散增强到底有没有用？**

---

## 核心结论

> **在本实验条件下，扩散增强未带来可测的性能增益。**

增量图片对总体指标有普遍收益，但扩散方法未超越程序化增强：
`C−B = +0.0069`、`D−B = −0.0005`、`D−C = −0.0074`（龟裂 AP50-95）。

项目保留这一负结果，并给出可复核的机制解释，而不是把它包装成“有效”。

## 四组对照

测试集 180 张 / 434 框，全程冻结；conf=0.001，iou=0.7。

| 组 | 数据方案 | mAP50 | mAP50-95 | 龟裂 AP50-95 |
|---|---|---:|---:|---:|
| A | 100% 原始（1440 张） | 0.7626 | 0.4327 | 0.1393 |
| B | 20% 原始（288 张） | 0.6315 | 0.3219 | 0.1027 |
| C | 20% + 程序化增强 40 张 | 0.7034 | 0.3632 | 0.1096 |
| D | 20% + 扩散增强 40 张 | 0.6658 | 0.3477 | 0.1022 |

基线 A 组：P 0.7783 / R 0.6651 / mAP50 0.7626 / mAP50-95 0.4327；
龟裂 AP50 0.4318 / AP50-95 0.1393 / Recall 0.2344。

> C 组是 **程序化增强**，不是 AIGC。
> D 组是 **「程序化 mask + 扩散填充」的混合方案**，且定性为「低频弱增广」：
> 8 张 4× 放大抽检，仅 4 张可见连续暗线。CPU 单张扩散约 72 秒。

## 项目构成

- **数据治理 v2**：`datasets/neu_det_v2/`，train 1440 图 / 3320 框、val 180 / 433、test 180 / 434；
  每类 240 / 30 / 30。仅修正 train 内 2 行重复标签，val/test 冻结。
- **定向人工复核**：47 行（A16 / B1 / C14 / D16，人工 31 行 + AI 16 行）；
  结论是无图需删除，主因是低对比度而非错标。
- **四组实验**：A/B/C/D 各一份 `best.pt`，共 12 次评估，校验文件齐全。
- **质检平台**：公网版为四页质检闭环（质检工作台 / 质检记录 / 质检报告 / 模型实验），完成
  29/29 实机验收；免登录、无需服务器，图片只在本机浏览器推理，记录存入当前浏览器 IndexedDB。
  本地 Streamlit 版保留三页开发台；后处理同源冒烟 18/18 PASS，不等同于 ONNX 精度评测。
- **参赛材料**：`reports/materials/`，含 12 页正文 + 1 页附录的答辩 PPT、14 页项目说明书（DOCX + PDF）、
  一页式展示页、路演讲稿、20 题问答、视频脚本与录屏清单。

## 快速开始

```bash
# 公网检测平台（无需安装，直接打开）
# https://zzq10086.github.io/defectmind-ai-showcase/platform/

# 检测平台（Windows）
.venv\Scripts\activate
streamlit run app.py

# 单张 / 批量推理
python predict.py --source datasets/neu_det_v2/test/images/scratches_1.jpg
python predict.py --source datasets/neu_det_v2/test/images --limit 30

# 测试集评估（正式口径）
python test.py
```

## 目录

```text
NEU-DET/
├─ datasets/
│  ├─ neu_det_v2/              # 权威数据集（v2，已冻结）
│  └─ experiment/              # subset20、C_aug、D_aug、B/C/D_train
├─ runs/defectmind/
│  ├─ yolo26n_baseline/        # A 组基线
│  ├─ exp_{A,B,C,D}/           # 四组权重与训练日志
│  ├─ predictions/             # 批量推理与 JSON/CSV
│  └─ platform/records.db      # 平台历史记录
├─ models/sd15-inpainting/     # SD1.5 inpainting 权重
├─ reports/
│  ├─ experiment/              # 四组指标、QC、校验
│  ├─ crazing_manual_review/   # 47 行复核
│  └─ materials/
│     ├─ platform/             # 纯浏览器检测平台（ONNX + ORT Web）
│     └─ ...                   # 参赛材料包
├─ app.py  platform_core.py
├─ app_pages/{workbench,history,compare}.py
└─ scripts/                    # 生成、审计、截图、冒烟脚本
```

## 权威数据来源

- 四组指标：`reports/experiment/matrix_metrics.csv`、`per_class_metrics.csv`
- 实验结论：`reports/experiment/experiment_matrix.md`
- 单次评估 JSON：`reports/metrics/exp_*_metrics.json`
- 数据划分：`datasets/neu_det_v2/split_manifest.csv`、`dataset_stats.json`
- 增强 QC：`reports/experiment/C_aug_qc.md`、`D_aug_qc_v2.md`

## 诚实边界

- 单 seed、无重复训练与显著性检验，结论只具方向性。
- 20% 子集仅取单类纯净样本，与 A 组不是严格同分布比较。
- D 与 C 增强强度不等价；D 弱于 C 可能部分源于强度差异。
- A 组沿用第一阶段 v1 结果，非严格同数据基线。
- 320 分辨率是 CPU 条件的妥协；未做产线阈值标定与域外泛化。
- 平台是原型系统，不是已标定的产线质检系统。

## 材料与复检

- 提交材料索引：`reports/materials/submission/比赛材料自检清单.md`
- AI 协作日志：`reports/materials/submission/AI协作日志.md`（DOCX/PDF 同目录）
- PT/ONNX 一致性审计：`reports/materials/submission/PT-ONNX一致性审计日志.md`（DOCX/PDF 同目录）
- 部署说明：`reports/materials/submission/部署说明.md`
- 5/10/30 分钟演示答辩方案：`reports/materials/submission/`
- 完整源码包：`reports/materials/submission/DefectMind_AI_完整源码.zip`
- 演示数据包：`reports/materials/submission/DefectMind_AI_演示数据.zip`
- 交付清单：`reports/materials/00_inventory.md`
- 展示页：`reports/materials/showcase.html`（离线可开，含 PPT / DOCX / PDF 下载）
- 答辩 PPT：`reports/materials/DefectMind_AI_答辩PPT.pptx`（12 页正文 + 1 页附录）
- 项目说明书：`reports/materials/DefectMind_AI_项目说明书.docx`（14 页）
- 只读 PDF：`reports/materials/DefectMind_AI_项目说明书.pdf`
- 一键复检：`scripts/qa_materials.bat`（禁用词 / 数字 / 链接 / 页数）

## 约束

- 原始 `ANNOTATIONS/`、根目录 `train/`、`valid/` 只读，不覆盖。
- 测试集不参与训练、增强、调参或模型选择。
- 所有数字来自实际运行；负结果按原话保留，不改写、不美化。
