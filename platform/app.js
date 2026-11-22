/* DefectMind AI - browser-only inspection platform.
 * The ONNX graph is the raw YOLO26n head: [1, 10, 2100] = 4 box + 6 class scores.
 * Decode and NMS are implemented locally to mirror predict.py.
 */

(() => {
  "use strict";

  const DATA = window.DEFECTMIND_DATA;
  const CLASS_NAMES = DATA.classNames;
  const CLASS_ZH = DATA.classZh;
  const ARMS = DATA.arms;
  const ARMS_BY_ID = Object.fromEntries(ARMS.map((arm) => [arm.id, arm]));
  const CLASS_COLORS = ["#e74c3c", "#3498db", "#e67e22", "#3cb956", "#9b59b6", "#f1c40f"];
  const MODEL_URL = "model/defectmind_yolo26n.onnx";
  const IMG_SIZE = 320;
  const PAD_COLOR = 114;
  const MAX_DETECTIONS = 300;
  const DEFAULT_CONF = 0.25;
  const DEFAULT_IOU = 0.45;
  const DB_NAME = "defectmind_platform";
  const DB_STORE = "records";
  const DB_VERSION = 1;

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
  const fmt = (value, digits = 4) => Number(value).toFixed(digits);
  const escapeHtml = (value) =>
    String(value).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));

  const state = {
    view: "workbench",
    model: null,
    modelReady: false,
    modelLoading: false,
    image: null,
    imageName: "",
    imageUrl: "",
    imageObjectUrl: null,
    detections: [],
    timing: null,
    conf: DEFAULT_CONF,
    iou: DEFAULT_IOU,
    lastRun: null,
    split: "test",
    history: [],
    historyQuery: "",
    busy: false,
    rerunTimer: null,
  };

  const els = {
    boot: $("#bootOverlay"),
    bootProgress: $("#bootProgress"),
    bootPercent: $("#bootPercent"),
    bootDetail: $("#bootDetail"),
    bootText: $("#bootText"),
    modelDot: $("#modelDot"),
    modelState: $("#modelState"),
    modelMeta: $("#modelMeta"),
    modelProgress: $("#modelProgress"),
    pageName: $("#pageName"),
    fileInput: $("#fileInput"),
    dropzone: $("#dropzone"),
    sampleGrid: $("#sampleGrid"),
    confRange: $("#confRange"),
    iouRange: $("#iouRange"),
    confValue: $("#confValue"),
    iouValue: $("#iouValue"),
    runButton: $("#runButton"),
    saveButton: $("#saveButton"),
    resetParams: $("#resetParams"),
    emptyViewer: $("#emptyViewer"),
    compareView: $("#compareView"),
    originalImage: $("#originalImage"),
    resultCanvas: $("#resultCanvas"),
    originalSize: $("#originalSize"),
    detectedCount: $("#detectedCount"),
    imageMeta: $("#imageMeta"),
    timingBadge: $("#timingBadge"),
    metricCount: $("#metricCount"),
    metricClasses: $("#metricClasses"),
    metricConfidence: $("#metricConfidence"),
    metricTiming: $("#metricTiming"),
    detectionRows: $("#detectionRows"),
    classDistribution: $("#classDistribution"),
    exportJson: $("#exportJson"),
    historyList: $("#historyList"),
    emptyHistory: $("#emptyHistory"),
    historyCount: $("#historyCount"),
    historySearch: $("#historySearch"),
    exportHistory: $("#exportHistory"),
    clearHistory: $("#clearHistory"),
    metricToggle: $("#metricToggle"),
    compareKpis: $("#compareKpis"),
    mapChart: $("#mapChart"),
    chartMeta: $("#chartMeta"),
    matrixRows: $("#matrixRows"),
    classChart: $("#classChart"),
    toast: $("#toast"),
    modal: $("#modal"),
    modalTitle: $("#modalTitle"),
    modalBody: $("#modalBody"),
    modalClose: $("#modalClose"),
  };

  let toastTimer = null;
  let dbPromise = null;

  function showToast(message, kind = "ok") {
    clearTimeout(toastTimer);
    els.toast.classList.toggle("error", kind === "error");
    $("span", els.toast).textContent = message;
    els.toast.classList.add("show");
    toastTimer = setTimeout(() => els.toast.classList.remove("show"), 3200);
  }

  function setView(view, updateHash = true) {
    const names = { workbench: "检测工作台", history: "历史记录", compare: "实验对比" };
    if (!names[view]) view = "workbench";
    state.view = view;
    $$(".nav-item").forEach((button) => button.classList.toggle("active", button.dataset.view === view));
    $$(".view").forEach((section) => section.classList.toggle("active", section.id === `view-${view}`));
    els.pageName.textContent = names[view];
    if (view === "history") renderHistory();
    if (view === "compare") renderCompare();
    if (updateHash) history.replaceState(null, "", `#${view}`);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function setModelStatus(kind, text, meta) {
    els.modelDot.className = `status-dot ${kind}`;
    els.modelState.textContent = text;
    if (meta) els.modelMeta.textContent = meta;
  }

  function updateBoot(percent, detail, text) {
    const pct = clamp(percent, 0, 100);
    els.bootProgress.style.width = `${pct}%`;
    els.bootPercent.textContent = `${Math.round(pct)}%`;
    if (detail) els.bootDetail.textContent = detail;
    if (text) els.bootText.textContent = text;
    els.modelProgress.style.width = `${pct}%`;
  }

  function hideBoot() {
    els.boot.classList.add("hidden");
    setTimeout(() => {
      if (els.boot.classList.contains("hidden")) els.boot.style.display = "none";
    }, 450);
  }

  async function fetchModelWithProgress(url) {
    const response = await fetch(url, { cache: "force-cache" });
    if (!response.ok) throw new Error(`模型下载失败：HTTP ${response.status}`);
    const total = Number(response.headers.get("content-length")) || 11200000;
    if (!response.body || !response.body.getReader) {
      const buffer = await response.arrayBuffer();
      updateBoot(62, "权重已下载", "正在初始化 WASM 推理引擎…");
      return buffer;
    }
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      const pct = 8 + (received / total) * 48;
      updateBoot(pct, `${(received / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} MB`);
    }
    const merged = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    updateBoot(62, "权重已下载", "正在初始化 WASM 推理引擎…");
    return merged.buffer;
  }

  async function initModel() {
    if (state.modelLoading || state.modelReady) return;
    state.modelLoading = true;
    try {
      if (!window.ort) throw new Error("ONNX Runtime 未加载");
      ort.env.wasm.wasmPaths = new URL("vendor/", window.location.href).href;
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      ort.env.logLevel = "error";
      updateBoot(5, "连接本地模型", "正在装载检测模型");
      setModelStatus("", "模型加载中", "YOLO26n · ONNX · 320px");

      const buffer = await fetchModelWithProgress(MODEL_URL);
      state.model = await ort.InferenceSession.create(buffer, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      updateBoot(78, "WASM 引擎就绪", "正在预热卷积核与检测头…");

      const warm = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
      const feeds = {};
      feeds[state.model.inputNames[0]] = new ort.Tensor("float32", warm, [1, 3, IMG_SIZE, IMG_SIZE]);
      await state.model.run(feeds);

      state.modelReady = true;
      state.modelLoading = false;
      updateBoot(100, "模型已就绪", "正在载入示例图像");
      setModelStatus("ready", "模型已就绪", "YOLO26n · ONNX · 320px");
      els.runButton.disabled = false;
      setTimeout(hideBoot, 300);
      loadSample("crazing_101.jpg", true);
    } catch (error) {
      state.modelLoading = false;
      console.error(error);
      updateBoot(100, "加载失败", "无法启动浏览器推理");
      setModelStatus("error", "模型加载失败", String(error.message || error));
      els.bootDetail.textContent = String(error.message || error);
      showToast(`模型加载失败：${error.message || error}`, "error");
    }
  }

  function buildSampleGrid() {
    const samples = [
      "crazing_101.jpg", "crazing_11.jpg", "crazing_110.jpg",
      "inclusion_117.jpg", "inclusion_121.jpg", "inclusion_125.jpg",
      "patches_134.jpg", "patches_136.jpg", "patches_162.jpg",
      "pitted_surface_123.jpg", "pitted_surface_126.jpg", "pitted_surface_129.jpg",
      "rolled-in_scale_102.jpg", "rolled-in_scale_132.jpg", "rolled-in_scale_135.jpg",
      "scratches_122.jpg", "scratches_126.jpg", "scratches_13.jpg",
    ];
    els.sampleGrid.innerHTML = samples
      .map((name) => {
        const cls = classFromFilename(name);
        return `<button class="sample-thumb" data-sample="${name}" data-class="${escapeHtml(CLASS_ZH[cls] || cls)}" title="${escapeHtml(CLASS_ZH[cls])} · ${name}">
          <img src="samples/${name}" alt="${escapeHtml(CLASS_ZH[cls])}示例" loading="lazy">
        </button>`;
      })
      .join("");
    $$(".sample-thumb", els.sampleGrid).forEach((button) => {
      button.addEventListener("click", () => loadSample(button.dataset.sample, true));
    });
  }

  function classFromFilename(name) {
    return CLASS_NAMES.find((cls) => name.startsWith(`${cls}_`)) || CLASS_NAMES[0];
  }

  function markActiveSample(name) {
    $$(".sample-thumb", els.sampleGrid).forEach((button) => {
      button.classList.toggle("active", button.dataset.sample === name);
    });
  }

  function releaseObjectUrl() {
    if (state.imageObjectUrl) {
      URL.revokeObjectURL(state.imageObjectUrl);
      state.imageObjectUrl = null;
    }
  }

  function loadSample(name, autoRun = false) {
    markActiveSample(name);
    loadImageFromUrl(`samples/${name}`, name, autoRun);
  }

  function loadImageFromUrl(url, name, autoRun = false, objectUrl = null) {
    releaseObjectUrl();
    if (objectUrl) state.imageObjectUrl = objectUrl;
    const image = new Image();
    image.decoding = "async";
    image.onload = () => {
      state.image = image;
      state.imageName = name || "image.jpg";
      state.imageUrl = image.src;
      state.detections = [];
      state.lastRun = null;
      state.timing = null;
      els.originalImage.src = image.src;
      els.originalSize.textContent = `${image.naturalWidth} × ${image.naturalHeight}`;
      els.imageMeta.textContent = state.imageName;
      els.emptyViewer.classList.add("hidden");
      els.compareView.classList.remove("hidden");
      els.saveButton.disabled = true;
      els.exportJson.disabled = true;
      resetResultDisplay();
      if (autoRun && state.modelReady) runInference();
    };
    image.onerror = () => {
      showToast("图像加载失败，请换一张图片", "error");
    };
    image.src = url;
  }

  function handleFile(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      showToast("请选择 JPG、PNG 或 WebP 图像", "error");
      return;
    }
    const url = URL.createObjectURL(file);
    markActiveSample("");
    loadImageFromUrl(url, file.name, true, url);
  }

  function resetResultDisplay() {
    els.detectedCount.textContent = "—";
    els.timingBadge.textContent = "—";
    els.metricCount.textContent = "0";
    els.metricClasses.textContent = "0";
    els.metricConfidence.textContent = "—";
    els.metricTiming.textContent = "—";
    els.detectionRows.innerHTML = '<tr class="placeholder-row"><td colspan="4">还没有检测结果</td></tr>';
    els.classDistribution.innerHTML = '<div class="empty-mini">检测后显示各类别数量</div>';
    const ctx = els.resultCanvas.getContext("2d");
    if (state.image) {
      els.resultCanvas.width = state.image.naturalWidth;
      els.resultCanvas.height = state.image.naturalHeight;
      ctx.drawImage(state.image, 0, 0);
    }
  }

  function preprocess(image) {
    const width = image.naturalWidth;
    const height = image.naturalHeight;
    const ratio = Math.min(IMG_SIZE / width, IMG_SIZE / height);
    const newWidth = Math.round(width * ratio);
    const newHeight = Math.round(height * ratio);
    const dw = (IMG_SIZE - newWidth) / 2;
    const dh = (IMG_SIZE - newHeight) / 2;
    const left = Math.round(dw - 0.1);
    const top = Math.round(dh - 0.1);

    const sourceCanvas = document.createElement("canvas");
    sourceCanvas.width = width;
    sourceCanvas.height = height;
    const sourceCtx = sourceCanvas.getContext("2d", { willReadFrequently: true });
    sourceCtx.drawImage(image, 0, 0);
    const pixels = sourceCtx.getImageData(0, 0, width, height).data;

    const plane = IMG_SIZE * IMG_SIZE;
    const tensor = new Float32Array(plane * 3);
    tensor.fill(PAD_COLOR / 255);

    // Center-aligned bilinear resize, matching OpenCV INTER_LINEAR.
    for (let y = 0; y < newHeight; y += 1) {
      const srcY = ((y + 0.5) * height) / newHeight - 0.5;
      const y0 = Math.floor(srcY);
      const fy = srcY - y0;
      const y0c = clamp(y0, 0, height - 1);
      const y1c = clamp(y0 + 1, 0, height - 1);
      const row0 = y0c * width;
      const row1 = y1c * width;
      const dstRow = (top + y) * IMG_SIZE + left;
      for (let x = 0; x < newWidth; x += 1) {
        const srcX = ((x + 0.5) * width) / newWidth - 0.5;
        const x0 = Math.floor(srcX);
        const fx = srcX - x0;
        const x0c = clamp(x0, 0, width - 1);
        const x1c = clamp(x0 + 1, 0, width - 1);
        const i00 = (row0 + x0c) * 4;
        const i01 = (row0 + x1c) * 4;
        const i10 = (row1 + x0c) * 4;
        const i11 = (row1 + x1c) * 4;
        const w00 = (1 - fx) * (1 - fy);
        const w01 = fx * (1 - fy);
        const w10 = (1 - fx) * fy;
        const w11 = fx * fy;
        const dst = dstRow + x;
        tensor[dst] = (pixels[i00] * w00 + pixels[i01] * w01 + pixels[i10] * w10 + pixels[i11] * w11) / 255;
        tensor[plane + dst] = (pixels[i00 + 1] * w00 + pixels[i01 + 1] * w01 + pixels[i10 + 1] * w10 + pixels[i11 + 1] * w11) / 255;
        tensor[plane * 2 + dst] = (pixels[i00 + 2] * w00 + pixels[i01 + 2] * w01 + pixels[i10 + 2] * w10 + pixels[i11 + 2] * w11) / 255;
      }
    }
    return { tensor, ratio, left, top };
  }

  function boxIoU(a, b) {
    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    if (!inter) return 0;
    const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
    const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
    return inter / (areaA + areaB - inter + 1e-9);
  }

  function decodeOutput(output, ratio, left, top, confThreshold, iouThreshold, width, height) {
    const dims = output.dims;
    if (!dims || dims.length !== 3 || dims[1] !== 4 + CLASS_NAMES.length) {
      throw new Error(`未识别的模型输出形状：${dims ? dims.join("×") : "unknown"}`);
    }
    const count = dims[2];
    const data = output.data;
    const candidates = [];
    for (let index = 0; index < count; index += 1) {
      let bestScore = -1;
      let bestClass = 0;
      for (let cls = 0; cls < CLASS_NAMES.length; cls += 1) {
        const score = data[(4 + cls) * count + index];
        if (score > bestScore) {
          bestScore = score;
          bestClass = cls;
        }
      }
      if (bestScore < confThreshold) continue;
      const cx = data[index];
      const cy = data[count + index];
      const bw = data[count * 2 + index];
      const bh = data[count * 3 + index];

      let x1 = (cx - bw / 2 - left) / ratio;
      let y1 = (cy - bh / 2 - top) / ratio;
      let x2 = (cx + bw / 2 - left) / ratio;
      let y2 = (cy + bh / 2 - top) / ratio;
      x1 = clamp(x1, 0, width);
      y1 = clamp(y1, 0, height);
      x2 = clamp(x2, 0, width);
      y2 = clamp(y2, 0, height);
      if (x2 - x1 < 1 || y2 - y1 < 1) continue;
      candidates.push({ x1, y1, x2, y2, conf: bestScore, cls: bestClass });
    }

    candidates.sort((a, b) => b.conf - a.conf);
    const kept = [];
    for (const candidate of candidates) {
      let keep = true;
      for (const chosen of kept) {
        if (chosen.cls !== candidate.cls) continue;
        if (boxIoU(candidate, chosen) > iouThreshold) {
          keep = false;
          break;
        }
      }
      if (keep) kept.push(candidate);
      if (kept.length >= MAX_DETECTIONS) break;
    }
    return kept;
  }

  function setBusy(busy) {
    state.busy = busy;
    els.runButton.disabled = busy || !state.modelReady;
    const label = $("span", els.runButton);
    if (label) label.textContent = busy ? "正在检测…" : "开始检测";
    els.runButton.classList.toggle("loading", busy);
  }

  async function runInference() {
    if (!state.image || !state.modelReady || state.busy) return;
    setBusy(true);
    try {
      const started = performance.now();
      const pre = preprocess(state.image);
      const feeds = {};
      feeds[state.model.inputNames[0]] = new ort.Tensor("float32", pre.tensor, [1, 3, IMG_SIZE, IMG_SIZE]);
      const outputMap = await state.model.run(feeds);
      const inferenceDone = performance.now();
      const output = outputMap[state.model.outputNames[0]];
      const detections = decodeOutput(
        output,
        pre.ratio,
        pre.left,
        pre.top,
        state.conf,
        state.iou,
        state.image.naturalWidth,
        state.image.naturalHeight
      );
      const finished = performance.now();
      state.detections = detections;
      state.timing = {
        total: finished - started,
        inference: inferenceDone - started,
        postprocess: finished - inferenceDone,
      };
      state.lastRun = {
        name: state.imageName,
        width: state.image.naturalWidth,
        height: state.image.naturalHeight,
        conf: state.conf,
        iou: state.iou,
        imgsz: IMG_SIZE,
        detections,
        timing: state.timing,
        createdAt: new Date().toISOString(),
      };
      renderResult();
      els.saveButton.disabled = false;
      els.exportJson.disabled = false;
    } catch (error) {
      console.error(error);
      showToast(`检测失败：${error.message || error}`, "error");
    } finally {
      setBusy(false);
    }
  }

  function renderResult() {
    const image = state.image;
    if (!image) return;
    const width = image.naturalWidth;
    const height = image.naturalHeight;
    const canvas = els.resultCanvas;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(image, 0, 0, width, height);
    const lineWidth = Math.max(2, Math.min(width, height) / 115);
    const fontSize = Math.max(10, Math.min(width, height) / 34);
    ctx.lineJoin = "round";
    for (const detection of state.detections) {
      const color = CLASS_COLORS[detection.cls];
      const x = detection.x1;
      const y = detection.y1;
      const w = detection.x2 - detection.x1;
      const h = detection.y2 - detection.y1;
      ctx.strokeStyle = color;
      ctx.fillStyle = `${color}20`;
      ctx.lineWidth = lineWidth;
      ctx.strokeRect(x, y, w, h);
      ctx.fillRect(x, y, w, h);
      const label = `${CLASS_ZH[CLASS_NAMES[detection.cls]]} ${detection.conf.toFixed(3)}`;
      ctx.font = `600 ${fontSize}px "Segoe UI", "Microsoft YaHei UI", sans-serif`;
      const paddingX = fontSize * 0.42;
      const textWidth = ctx.measureText(label).width;
      const labelHeight = fontSize * 1.48;
      const labelY = y - labelHeight - 1 < 0 ? y + 1 : y - labelHeight - 1;
      ctx.fillStyle = color;
      ctx.fillRect(x, labelY, textWidth + paddingX * 2, labelHeight);
      ctx.fillStyle = "#fff";
      ctx.textBaseline = "middle";
      ctx.fillText(label, x + paddingX, labelY + labelHeight / 2);
    }

    const classSet = new Set(state.detections.map((item) => item.cls));
    const avgConf = state.detections.length
      ? state.detections.reduce((sum, item) => sum + item.conf, 0) / state.detections.length
      : 0;
    els.detectedCount.textContent = `${state.detections.length} 个目标`;
    els.timingBadge.textContent = `${Math.round(state.timing.total)} ms`;
    els.metricCount.textContent = String(state.detections.length);
    els.metricClasses.textContent = String(classSet.size);
    els.metricConfidence.textContent = state.detections.length ? avgConf.toFixed(3) : "—";
    els.metricTiming.textContent = `${Math.round(state.timing.total)} ms`;

    if (!state.detections.length) {
      els.detectionRows.innerHTML = '<tr class="placeholder-row"><td colspan="4">当前阈值下未检出目标</td></tr>';
      els.classDistribution.innerHTML = '<div class="empty-mini">当前阈值下未检出目标</div>';
      return;
    }

    els.detectionRows.innerHTML = state.detections
      .map((item, index) => {
        const color = CLASS_COLORS[item.cls];
        const pct = Math.round(item.conf * 100);
        return `<tr>
          <td class="idx">${String(index + 1).padStart(2, "0")}</td>
          <td><span class="class-tag" style="--tag-color:${color}"><i></i>${escapeHtml(CLASS_ZH[CLASS_NAMES[item.cls]])}</span></td>
          <td><span class="confidence-bar" style="--tag-color:${color};--pct:${pct}%">${item.conf.toFixed(3)}</span></td>
          <td class="coords">${item.x1.toFixed(1)}, ${item.y1.toFixed(1)}, ${item.x2.toFixed(1)}, ${item.y2.toFixed(1)}</td>
        </tr>`;
      })
      .join("");

    const counts = new Map();
    state.detections.forEach((item) => counts.set(item.cls, (counts.get(item.cls) || 0) + 1));
    const maxCount = Math.max(...counts.values());
    els.classDistribution.innerHTML = CLASS_NAMES.map((name, cls) => {
      const count = counts.get(cls) || 0;
      if (!count) return "";
      return `<div class="dist-row">
        <span style="--tag-color:${CLASS_COLORS[cls]}"><i></i>${escapeHtml(CLASS_ZH[name])}</span>
        <div class="bar"><i style="width:${(count / maxCount) * 100}%;--tag-color:${CLASS_COLORS[cls]}"></i></div>
        <b>${count}</b>
      </div>`;
    }).join("");
  }

  function scheduleRerun() {
    if (!state.image || !state.modelReady) return;
    clearTimeout(state.rerunTimer);
    state.rerunTimer = setTimeout(runInference, 260);
  }

  function updateParamLabels() {
    els.confValue.textContent = state.conf.toFixed(2);
    els.iouValue.textContent = state.iou.toFixed(2);
  }

  function resetParams() {
    state.conf = DEFAULT_CONF;
    state.iou = DEFAULT_IOU;
    els.confRange.value = String(DEFAULT_CONF);
    els.iouRange.value = String(DEFAULT_IOU);
    updateParamLabels();
    scheduleRerun();
  }

  function imageToDataUrl(image, maxSize = 1200) {
    const ratio = Math.min(1, maxSize / Math.max(image.naturalWidth, image.naturalHeight));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(image.naturalWidth * ratio));
    canvas.height = Math.max(1, Math.round(image.naturalHeight * ratio));
    canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.84);
  }

  function openDatabase() {
    if (dbPromise) return dbPromise;
    dbPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(DB_STORE)) {
          const store = db.createObjectStore(DB_STORE, { keyPath: "id" });
          store.createIndex("createdAt", "createdAt");
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    return dbPromise;
  }

  async function dbGetAll() {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(DB_STORE, "readonly");
      const request = tx.objectStore(DB_STORE).getAll();
      request.onsuccess = () => resolve(request.result || []);
      request.onerror = () => reject(request.error);
    });
  }

  async function dbPut(record) {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(DB_STORE, "readwrite");
      tx.objectStore(DB_STORE).put(record);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async function dbDelete(id) {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(DB_STORE, "readwrite");
      tx.objectStore(DB_STORE).delete(id);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async function dbClear() {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(DB_STORE, "readwrite");
      tx.objectStore(DB_STORE).clear();
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async function saveCurrentRecord() {
    if (!state.lastRun || !state.image) return;
    els.saveButton.disabled = true;
    try {
      const record = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        createdAt: new Date().toISOString(),
        name: state.imageName,
        width: state.image.naturalWidth,
        height: state.image.naturalHeight,
        conf: state.conf,
        iou: state.iou,
        imgsz: IMG_SIZE,
        timing: state.timing,
        detections: state.detections,
        original: imageToDataUrl(state.image, 1100),
        annotated: els.resultCanvas.toDataURL("image/jpeg", 0.86),
      };
      await dbPut(record);
      await loadHistory();
      showToast("已保存到历史记录");
    } catch (error) {
      console.error(error);
      showToast("保存失败：浏览器可能禁用了本地存储", "error");
      els.saveButton.disabled = false;
    }
  }

  async function loadHistory() {
    try {
      state.history = (await dbGetAll()).sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)));
    } catch (error) {
      console.error(error);
      state.history = [];
    }
    renderHistory();
  }

  function historyMatches(record, query) {
    if (!query) return true;
    const q = query.toLowerCase();
    const classText = (record.detections || [])
      .map((item) => `${CLASS_NAMES[item.cls]} ${CLASS_ZH[CLASS_NAMES[item.cls]]}`)
      .join(" ");
    return `${record.name} ${classText}`.toLowerCase().includes(q);
  }

  function renderHistory() {
    if (!els.historyList) return;
    const filtered = state.history.filter((record) => historyMatches(record, state.historyQuery));
    els.historyCount.textContent = `${filtered.length} 条记录`;
    els.emptyHistory.classList.toggle("hidden", filtered.length > 0);
    els.historyList.classList.toggle("hidden", filtered.length === 0);
    els.exportHistory.disabled = state.history.length === 0;
    els.clearHistory.disabled = state.history.length === 0;
    if (!filtered.length) {
      els.historyList.innerHTML = "";
      return;
    }
    els.historyList.innerHTML = filtered
      .map((record) => {
        const date = new Date(record.createdAt);
        const dateText = Number.isNaN(date.getTime())
          ? record.createdAt
          : date.toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
        const classNames = [...new Set(record.detections.map((item) => CLASS_ZH[CLASS_NAMES[item.cls]]))];
        return `<article class="history-card" data-id="${escapeHtml(record.id)}">
          <div class="history-thumb"><img src="${record.annotated || record.original}" alt="" loading="lazy"></div>
          <div class="history-info">
            <div class="history-title">
              <strong>${escapeHtml(record.name)}</strong>
              <span class="history-date">${escapeHtml(dateText)}</span>
            </div>
            <div class="history-meta">
              <span class="mini-tag">${record.detections.length} 个目标</span>
              <span class="mini-tag">conf ${Number(record.conf).toFixed(2)}</span>
              <span class="mini-tag">iou ${Number(record.iou).toFixed(2)}</span>
              <span>${escapeHtml(classNames.slice(0, 3).join(" · "))}${classNames.length > 3 ? " …" : ""}</span>
            </div>
          </div>
          <div class="history-actions">
            <button class="icon-button view" title="查看详情"><svg><use href="#i-eye"/></svg></button>
            <button class="icon-button restore" title="重新载入"><svg><use href="#i-upload"/></svg></button>
            <button class="icon-button delete delete-action" title="删除"><svg><use href="#i-trash"/></svg></button>
          </div>
        </article>`;
      })
      .join("");
    $$(".history-card", els.historyList).forEach((card) => {
      const record = state.history.find((item) => item.id === card.dataset.id);
      $(".view", card).addEventListener("click", () => openRecordModal(record));
      $(".restore", card).addEventListener("click", () => restoreRecord(record));
      $(".delete-action", card).addEventListener("click", () => deleteRecord(record));
    });
  }

  function openRecordModal(record) {
    if (!record) return;
    els.modalTitle.textContent = record.name;
    const date = new Date(record.createdAt);
    const dateText = Number.isNaN(date.getTime()) ? record.createdAt : date.toLocaleString("zh-CN");
    els.modalBody.innerHTML = `
      <div class="record-summary">
        <span>${escapeHtml(dateText)}</span>
        <span>${record.width} × ${record.height}</span>
        <span>conf ${Number(record.conf).toFixed(2)}</span>
        <span>iou ${Number(record.iou).toFixed(2)}</span>
        <span>${record.detections.length} 个目标</span>
      </div>
      <div class="record-images">
        <figure><figcaption>原图</figcaption><img src="${record.original}" alt="原图"></figure>
        <figure><figcaption>检测结果</figcaption><img src="${record.annotated}" alt="检测结果"></figure>
      </div>
      <table class="record-table">
        <thead><tr><th>#</th><th>类别</th><th>置信度</th><th>坐标</th></tr></thead>
        <tbody>${record.detections.map((item, index) => `<tr>
          <td>${index + 1}</td>
          <td>${escapeHtml(CLASS_ZH[CLASS_NAMES[item.cls]])}</td>
          <td>${Number(item.conf).toFixed(3)}</td>
          <td>${Number(item.x1).toFixed(1)}, ${Number(item.y1).toFixed(1)}, ${Number(item.x2).toFixed(1)}, ${Number(item.y2).toFixed(1)}</td>
        </tr>`).join("")}</tbody>
      </table>`;
    els.modal.classList.add("open");
  }

  function closeModal() {
    els.modal.classList.remove("open");
  }

  async function deleteRecord(record) {
    if (!record) return;
    try {
      await dbDelete(record.id);
      await loadHistory();
      showToast("记录已删除");
    } catch (error) {
      showToast("删除失败", "error");
    }
  }

  async function restoreRecord(record) {
    if (!record) return;
    closeModal();
    markActiveSample("");
    state.conf = Number(record.conf) || DEFAULT_CONF;
    state.iou = Number(record.iou) || DEFAULT_IOU;
    els.confRange.value = String(state.conf);
    els.iouRange.value = String(state.iou);
    updateParamLabels();
    setView("workbench");
    loadImageFromUrl(record.original, record.name, true);
  }

  function downloadFile(filename, content, type = "application/json") {
    const blob = content instanceof Blob ? content : new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  function exportCurrentJson() {
    if (!state.lastRun) return;
    const payload = {
      platform: "DefectMind AI browser demo",
      model: "YOLO26n ONNX 320",
      createdAt: state.lastRun.createdAt,
      image: state.imageName,
      size: [state.lastRun.width, state.lastRun.height],
      params: { conf: state.conf, iou: state.iou, imgsz: IMG_SIZE },
      timing_ms: state.timing,
      detections: state.detections.map((item) => ({
        class_id: item.cls,
        class: CLASS_NAMES[item.cls],
        confidence: Number(item.conf.toFixed(6)),
        bbox_xyxy: [item.x1, item.y1, item.x2, item.y2].map((value) => Number(value.toFixed(2))),
      })),
    };
    downloadFile(`${state.imageName.replace(/\.[^.]+$/, "")}_defectmind.json`, JSON.stringify(payload, null, 2));
    showToast("JSON 已导出");
  }

  function exportHistoryCsv() {
    if (!state.history.length) return;
    const rows = [
      ["record_id", "created_at", "image", "width", "height", "conf", "iou", "detection_index", "class", "confidence", "x1", "y1", "x2", "y2"],
    ];
    for (const record of state.history) {
      if (!record.detections.length) {
        rows.push([record.id, record.createdAt, record.name, record.width, record.height, record.conf, record.iou, "", "", "", "", "", "", ""]);
        continue;
      }
      record.detections.forEach((item, index) => {
        rows.push([
          record.id, record.createdAt, record.name, record.width, record.height, record.conf, record.iou,
          index + 1, CLASS_NAMES[item.cls], Number(item.conf.toFixed(6)).toString(),
          Number(item.x1.toFixed(2)).toString(), Number(item.y1.toFixed(2)).toString(),
          Number(item.x2.toFixed(2)).toString(), Number(item.y2.toFixed(2)).toString(),
        ]);
      });
    }
    const csv = rows
      .map((row) => row.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(","))
      .join("\r\n");
    downloadFile("defectmind_history.csv", `\ufeff${csv}`, "text/csv;charset=utf-8");
    showToast("历史 CSV 已导出");
  }

  async function clearHistory() {
    if (!state.history.length) return;
    if (!window.confirm("确定清空当前浏览器中的全部检测记录吗？此操作不可撤销。")) return;
    try {
      await dbClear();
      await loadHistory();
      showToast("历史记录已清空");
    } catch (error) {
      showToast("清空失败", "error");
    }
  }

  function renderCompare() {
    const rows = DATA[state.split];
    const isDeploy = state.split === "deploy";
    els.chartMeta.textContent = isDeploy ? "conf 0.25 · iou 0.45" : "conf 0.001 · iou 0.7";
    els.compareKpis.innerHTML = ARMS.map((arm) => {
      const row = rows.find((item) => item.arm === arm.id);
      return `<article class="compare-kpi" style="--arm-color:${arm.color}">
        <div class="kpi-top"><span class="arm-badge">${arm.id}</span><span>${escapeHtml(arm.short)}</span></div>
        <strong>${fmt(row.map5095)}</strong>
        <small>mAP50-95 · mAP50 ${fmt(row.map50)}</small>
      </article>`;
    }).join("");

    els.mapChart.innerHTML = ARMS.map((arm) => {
      const row = rows.find((item) => item.arm === arm.id);
      return `<div class="bar-group" style="--arm-color:${arm.color}">
        <div class="bar-label"><i></i><b>${arm.id}</b> ${escapeHtml(arm.short)}</div>
        <div class="bar-stack">
          <div class="bar-line" style="--bar-color:#7c86e8">
            <div class="track"><i style="width:${(row.map50 / 0.85) * 100}%"></i></div>
            <b>${fmt(row.map50)}</b>
          </div>
          <div class="bar-line" style="--bar-color:#474c78">
            <div class="track"><i style="width:${(row.map5095 / 0.85) * 100}%"></i></div>
            <b>${fmt(row.map5095)}</b>
          </div>
        </div>
      </div>`;
    }).join("");

    const maxMap50 = Math.max(...rows.map((row) => row.map50));
    const maxMap5095 = Math.max(...rows.map((row) => row.map5095));
    els.matrixRows.innerHTML = ARMS.map((arm) => {
      const row = rows.find((item) => item.arm === arm.id);
      return `<tr>
        <td><span class="arm-cell" style="--arm-color:${arm.color}"><i>${arm.id}</i>${arm.id}</span></td>
        <td class="strategy">${escapeHtml(arm.label)}</td>
        <td>${arm.trainImages}</td>
        <td>${arm.syntheticImages}</td>
        <td>${fmt(row.precision)}</td>
        <td>${fmt(row.recall)}</td>
        <td class="${row.map50 === maxMap50 ? "best" : ""}">${fmt(row.map50)}</td>
        <td class="${row.map5095 === maxMap5095 ? "best" : ""}">${fmt(row.map5095)}</td>
        <td>${fmt(row.crazingAp50)}</td>
        <td>${fmt(row.crazingAp5095)}</td>
      </tr>`;
    }).join("");

    els.classChart.innerHTML = CLASS_NAMES.map((name, cls) => {
      const classRows = DATA.perClass.filter((row) => row.id === cls);
      return `<div class="class-row">
        <div class="class-name">${escapeHtml(CLASS_ZH[name])}<small>${escapeHtml(name)}</small></div>
        <div class="class-bars">${ARMS.map((arm) => {
          const item = classRows.find((row) => row.arm === arm.id);
          return `<div class="class-bar" style="--arm-color:${arm.color}">
            <div class="track"><i style="width:${(item.ap5095 / 0.65) * 100}%"></i></div>
            <b>${fmt(item.ap5095)}</b>
          </div>`;
        }).join("")}</div>
      </div>`;
    }).join("");
  }

  function bindEvents() {
    $$(".nav-item").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
    $$("[data-jump]").forEach((button) => button.addEventListener("click", () => setView(button.dataset.jump)));
    els.fileInput.addEventListener("change", (event) => {
      handleFile(event.target.files && event.target.files[0]);
      event.target.value = "";
    });
    ["dragenter", "dragover"].forEach((type) => {
      els.dropzone.addEventListener(type, (event) => {
        event.preventDefault();
        els.dropzone.classList.add("dragging");
      });
    });
    ["dragleave", "drop"].forEach((type) => {
      els.dropzone.addEventListener(type, (event) => {
        event.preventDefault();
        els.dropzone.classList.remove("dragging");
      });
    });
    els.dropzone.addEventListener("drop", (event) => {
      handleFile(event.dataTransfer.files && event.dataTransfer.files[0]);
    });
    els.confRange.addEventListener("input", () => {
      state.conf = Number(els.confRange.value);
      updateParamLabels();
      scheduleRerun();
    });
    els.iouRange.addEventListener("input", () => {
      state.iou = Number(els.iouRange.value);
      updateParamLabels();
      scheduleRerun();
    });
    els.runButton.addEventListener("click", runInference);
    els.saveButton.addEventListener("click", saveCurrentRecord);
    els.resetParams.addEventListener("click", resetParams);
    els.exportJson.addEventListener("click", exportCurrentJson);
    els.historySearch.addEventListener("input", () => {
      state.historyQuery = els.historySearch.value.trim();
      renderHistory();
    });
    els.exportHistory.addEventListener("click", exportHistoryCsv);
    els.clearHistory.addEventListener("click", clearHistory);
    els.metricToggle.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-split]");
      if (!button) return;
      state.split = button.dataset.split;
      $$("button", els.metricToggle).forEach((item) => item.classList.toggle("active", item === button));
      renderCompare();
    });
    els.modalClose.addEventListener("click", closeModal);
    els.modal.addEventListener("click", (event) => {
      if (event.target === els.modal) closeModal();
    });
    document.addEventListener("keydown", (event) => {
      const tag = (event.target && event.target.tagName) || "";
      if (["INPUT", "TEXTAREA", "SELECT"].includes(tag)) return;
      if (event.key === "1") setView("workbench");
      if (event.key === "2") setView("history");
      if (event.key === "3") setView("compare");
      if (event.key === "Enter" && state.view === "workbench") runInference();
      if (event.key === "Escape") closeModal();
    });
    window.addEventListener("hashchange", () => {
      const view = window.location.hash.replace("#", "");
      if (["workbench", "history", "compare"].includes(view)) setView(view, false);
    });
  }

  async function boot() {
    buildSampleGrid();
    updateParamLabels();
    bindEvents();
    const initialView = window.location.hash.replace("#", "");
    setView(["workbench", "history", "compare"].includes(initialView) ? initialView : "workbench", false);
    await loadHistory();
    renderCompare();
    initModel();
  }

  boot();
})();
