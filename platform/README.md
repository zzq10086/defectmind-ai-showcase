# DefectMind AI · 浏览器质检平台

永久地址：<https://zzq10086.github.io/defectmind-ai-showcase/platform/>

## 结构

- `index.html` / `styles.css` / `app.js`：质检工作台、质检记录、质检报告、模型实验四页。
- `model/defectmind_yolo26n.onnx`：YOLO26n 原始检测头，输出 `[1, 10, 2100]`。
- `vendor/`：ONNX Runtime Web 1.22.0（WASM 单线程）。
- `samples/`：18 张 NEU-DET 示例图。
- `data/experiment.js`：来自 `reports/experiment/` 的冻结实验数字。

## 质检闭环

- 工作台把风险结论放在结果图上方：风险等级、风险分、处置建议、主要缺陷、最大单框面积、总缺陷覆盖与可靠性提示。
- 判定依据可展开，显示类别权重、单框面积、总覆盖、缺陷数量、覆盖密度和计算式。
- 支持确认缺陷、标记误检、记录处置（放行 / 返修 / 报废 / 隔离）；标记误检不删除检测框。
- 质检记录保留复核人、处置结果、备注与更新时间，支持按待复核 / 已确认 / 误检 / 已处置筛选。
- 报告冻结生成时的检测结果与复核快照；单图报告显示复核信息，批次报告显示每张图状态与统计。

## 风险评分

风险分为 0–100 的确定性评分，阈值与权重集中在 `RISK_CONFIG`：

- 类别权重 42%：龟裂 / 夹杂高于点蚀、氧化皮压入、斑块、划痕。
- 最大单框面积 24%，35% 覆盖即记满分。
- 总缺陷覆盖 16%，60% 覆盖即记满分。
- 覆盖密度 18%：缺陷数量（8 个满分）与总覆盖（30% 满分）各占一半。
- 风险分 ≥65 为高风险，≥38 为中风险，其余有检出为低风险，无检出为未检出。
- 平均置信度只用于“可靠性偏低 / 一般 / 正常”提示，不参与质量严重度主评分。

## 数据与报告

- 记录与报告存储在浏览器 IndexedDB：`records`、`quality_reports`。
- 数据库版本 v3；v2 旧记录升级后保留，缺少 `review` 字段时按“待复核”显示，不迁移、不清空。
- 旧报告缺少复核字段时显示“未复核”，不因缺失字段报错。
- 报告支持打印 / 另存为 PDF；质检记录与报告 CSV 均包含复核字段。
- 报告归档只存在当前浏览器，不跨设备同步；清理浏览器数据会同时丢失历史记录与报告。

## 推理口径

- 输入：letterbox 到 320×320，RGB，`/255`。
- 输出：4 个框坐标（cx, cy, w, h）+ 6 类分数。
- 解码：置信度过滤 → cxcywh 转 xyxy → 按类别 NMS → 映射回原图。
- 默认参数：`conf=0.25`、`iou=0.45`、`imgsz=320`。
- 模型、类别映射、坐标换算、默认阈值与后处理未在本轮改动。

## 验收脚本

```powershell
$env:NODE_PATH='C:\Users\Chian zhang\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules'
node scripts/verify_platform_v23.js
node scripts/check_risk_distribution.js
```

脚本会用真实 Chrome 验证 1366×768、1440×1200、390px、复核闭环、报告、CSV、打印、旧 IndexedDB 兼容和控制台错误。

## 边界

- 浏览器 Canvas 的双线性缩放与 OpenCV 实现存在细微像素差异，坐标与置信度可能有约 1e-2 级浮动。
- 质检结论是图像级辅助判定，不替代人工终判，不代表真实缺陷深度或产品寿命。
- 首次访问需下载约 11 MB 模型；之后由浏览器缓存。GitHub Pages 国内访问速度取决于网络。
