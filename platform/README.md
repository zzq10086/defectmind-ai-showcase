# DefectMind AI · 浏览器检测平台

永久地址：<https://zzq10086.github.io/defectmind-ai-showcase/platform/>

## 结构

- `index.html` / `styles.css` / `app.js`：检测工作台、历史记录、实验对比三页。
- `model/defectmind_yolo26n.onnx`：YOLO26n 原始检测头，输出 `[1, 10, 2100]`。
- `vendor/`：ONNX Runtime Web 1.22.0（WASM 单线程）。
- `samples/`：18 张 NEU-DET 示例图。
- `data/experiment.js`：来自 `reports/experiment/` 的冻结实验数字。

## 推理口径

- 输入：letterbox 到 320×320，RGB，`/255`。
- 输出：4 个框坐标（cx, cy, w, h）+ 6 类分数。
- 解码：置信度过滤 → cxcywh 转 xyxy → 按类别 NMS → 映射回原图。
- 默认参数：`conf=0.25`、`iou=0.45`、`imgsz=320`。

## 边界

- 浏览器 Canvas 的双线性缩放与 OpenCV 实现存在细微像素差异，坐标与置信度可能有约 1e-2 级浮动。
- 历史记录存储在访问者自己的浏览器 IndexedDB，不跨设备同步，清理浏览器数据后会丢失。
- 首次访问需下载约 11 MB 模型；之后由浏览器缓存。GitHub Pages 国内访问速度取决于网络。
