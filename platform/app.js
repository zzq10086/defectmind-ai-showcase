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
  const DB_REPORT_STORE = "quality_reports";
  const DB_VERSION = 3;
  const MAX_BATCH_RECORDS = 20;
  const MODEL_CACHE = "defectmind-model-v1";
  const BOOT_STALL_MS = 45000;

  const CLASS_META = {
    crazing: {
      priority: 3,
      appearance: "表面呈网状或龟甲状细裂纹，通常边界弥散，局部对比度偏低。",
      cause: "可能与冷却过程、轧制应力或材料状态有关，需结合现场工艺复核。",
      advice: "建议人工复核裂纹连续性；高风险时隔离该批次，并排查冷却与轧制工艺。",
    },
    inclusion: {
      priority: 3,
      appearance: "非金属夹杂或异物压入表面，形态不规则，常呈局部块状。",
      cause: "可能与来料洁净度、冶炼或轧制过程有关，需追溯材料与工序。",
      advice: "建议复核夹杂尺寸与深度，高风险时隔离该批次并追溯来料与冶炼工序。",
    },
    patches: {
      priority: 1,
      appearance: "局部颜色或纹理异常，边界相对清晰，常见斑片状区域。",
      cause: "可能与氧化、附着物或局部表面状态变化有关。",
      advice: "建议复检并确认是否可修磨；若为偶发，可先按常规频次抽检。",
    },
    pitted_surface: {
      priority: 2,
      appearance: "表面分布密集小凹坑，局部呈现点状或麻点状纹理。",
      cause: "可能与除鳞、轧制润滑或表面清洁状态有关。",
      advice: "建议复核凹坑密度与分布，并关注除鳞、润滑和表面清洁工序。",
    },
    "rolled-in_scale": {
      priority: 2,
      appearance: "氧化皮被压入表面，呈不规则嵌入状或条带状区域。",
      cause: "可能与除鳞不充分或轧制过程有关。",
      advice: "建议复核压入深度，并排查除鳞和轧制工序。",
    },
    scratches: {
      priority: 1,
      appearance: "表面呈线状机械擦伤，方向性通常较明显。",
      cause: "可能与辊道、吊装或转运过程中的接触摩擦有关。",
      advice: "建议确认划痕深度和范围，并排查辊道、吊装及转运接触点。",
    },
  };

  const RISK_META = {
    high: {
      label: "高风险",
      order: 3,
      tone: "danger",
      short: "需拦截复核",
      disposition: "建议拦截并人工复核",
      batchDisposition: "建议批次隔离 / 人工复检",
    },
    medium: {
      label: "中风险",
      order: 2,
      tone: "warn",
      short: "需人工复检",
      disposition: "建议人工复检",
      batchDisposition: "建议批次抽检 / 人工复核",
    },
    low: {
      label: "低风险",
      order: 1,
      tone: "",
      short: "可放行抽检",
      disposition: "建议放行，保留抽检",
      batchDisposition: "建议批次放行，保留抽检",
    },
    none: {
      label: "未检出",
      order: 0,
      tone: "",
      short: "常规抽检",
      disposition: "未检出明显缺陷，建议常规抽检",
      batchDisposition: "未检出明显缺陷，建议常规抽检",
    },
  };

  // Deterministic, explainable risk scoring. Every threshold lives here.
  const RISK_CONFIG = {
    weights: { class: 0.42, maxBox: 0.24, totalBox: 0.16, density: 0.18 },
    classScore: {
      crazing: 100,
      inclusion: 94,
      pitted_surface: 64,
      "rolled-in_scale": 60,
      patches: 42,
      scratches: 36,
    },
    maxBoxFull: 0.35,
    totalBoxFull: 0.6,
    densityCountFull: 8,
    densityCoverageFull: 0.3,
    thresholds: { high: 65, medium: 38 },
    confidence: { low: 0.4, medium: 0.55 },
  };

  const REVIEW_META = {
    pending: { label: "待复核", className: "pending" },
    confirmed: { label: "已确认", className: "confirmed" },
    false_positive: { label: "误检", className: "false" },
    closed: { label: "已处置", className: "closed" },
    unreviewed: { label: "未复核", className: "unreviewed" },
  };
  const REVIEW_STATUSES = ["pending", "confirmed", "false_positive", "closed"];
  const DISPOSITIONS = ["放行", "返修", "报废", "隔离"];

  const BASELINE_METRICS = Object.fromEntries(
    DATA.perClass.filter((row) => row.arm === "A").map((row) => [row.name, row])
  );

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
  const fmt = (value, digits = 4) => Number(value).toFixed(digits);
  const escapeHtml = (value) =>
    String(value).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));

  function reviewOf(record) {
    const source = record && record.review && typeof record.review === "object" ? record.review : {};
    const status = REVIEW_STATUSES.includes(source.status) ? source.status : "pending";
    return {
      status,
      disposition: status === "closed" && DISPOSITIONS.includes(source.disposition) ? source.disposition : "",
      reviewer: typeof source.reviewer === "string" ? source.reviewer.trim().slice(0, 40) : "",
      note: typeof source.note === "string" ? source.note.trim().slice(0, 300) : "",
      updatedAt: typeof source.updatedAt === "string" ? source.updatedAt : "",
    };
  }

  function snapshotReviewOf(record) {
    if (!record || !record.review || typeof record.review !== "object") return null;
    return reviewOf(record);
  }

  function normalizeReviewForSave(review) {
    const normalized = reviewOf({ review });
    if (!normalized.updatedAt && normalized.status !== "pending") {
      normalized.updatedAt = new Date().toISOString();
    }
    return normalized;
  }

  function reviewStats(records) {
    const counts = { pending: 0, confirmed: 0, false_positive: 0, closed: 0, unreviewed: 0 };
    (records || []).forEach((record) => {
      const review = snapshotReviewOf(record);
      if (!review) {
        counts.unreviewed += 1;
        return;
      }
      counts[review.status] += 1;
    });
    return counts;
  }

  function reviewBadge(status, label = "") {
    const meta = REVIEW_META[status] || REVIEW_META.pending;
    return `<span class="review-badge ${meta.className}">${escapeHtml(label || meta.label)}</span>`;
  }

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
    reports: [],
    reportMode: "single",
    singleReportSource: "current",
    batchSelection: new Set(),
    reportsLoaded: false,
    currentSaved: false,
    currentRecordId: null,
    currentRecordCreatedAt: null,
    currentReview: null,
    historyFilter: "all",
    busy: false,
    rerunTimer: null,
  };

  const els = {
    boot: $("#bootOverlay"),
    bootProgress: $("#bootProgress"),
    bootPercent: $("#bootPercent"),
    bootDetail: $("#bootDetail"),
    bootText: $("#bootText"),
    bootElapsed: $("#bootElapsed"),
    bootHint: $("#bootHint"),
    bootRetry: $("#bootRetry"),
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
    outcomePanel: $("#outcomePanel"),
    outcomeRisk: $("#outcomeRisk"),
    outcomeSummary: $("#outcomeSummary"),
    verdictScore: $("#verdictScore"),
    verdictMainDefect: $("#verdictMainDefect"),
    verdictMaxBox: $("#verdictMaxBox"),
    verdictCoverage: $("#verdictCoverage"),
    verdictConfidence: $("#verdictConfidence"),
    verdictReview: $("#verdictReview"),
    verdictBasisBody: $("#verdictBasisBody"),
    reviewConfirm: $("#reviewConfirm"),
    reviewFalse: $("#reviewFalse"),
    reviewDispose: $("#reviewDispose"),
    outcomeSave: $("#outcomeSave"),
    outcomeReport: $("#outcomeReport"),
    outcomeBatch: $("#outcomeBatch"),
    paramSummary: $("#paramSummary"),
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
    metricMain: $("#metricMain"),
    metricMainHint: $("#metricMainHint"),
    metricRisk: $("#metricRisk"),
    metricRiskHint: $("#metricRiskHint"),
    metricClasses: $("#metricClasses"),
    metricConfidence: $("#metricConfidence"),
    metricTiming: $("#metricTiming"),
    detailParams: $("#detailParams"),
    detectionRows: $("#detectionRows"),
    classDistribution: $("#classDistribution"),
    exportJson: $("#exportJson"),
    historyList: $("#historyList"),
    emptyHistory: $("#emptyHistory"),
    historyCount: $("#historyCount"),
    historySearch: $("#historySearch"),
    historyFilters: $("#historyFilters"),
    exportHistory: $("#exportHistory"),
    clearHistory: $("#clearHistory"),
    historyBatchReport: $("#historyBatchReport"),
    reportMode: $("#reportMode"),
    singleReportPane: $("#singleReportPane"),
    batchReportPane: $("#batchReportPane"),
    singleRecordSelect: $("#singleRecordSelect"),
    singleReportPreview: $("#singleReportPreview"),
    batchSelectionCount: $("#batchSelectionCount"),
    selectRecentRecords: $("#selectRecentRecords"),
    selectAllRecords: $("#selectAllRecords"),
    clearSelectedRecords: $("#clearSelectedRecords"),
    batchRecordList: $("#batchRecordList"),
    batchReportPreview: $("#batchReportPreview"),
    reportTitle: $("#reportTitle"),
    reportJobId: $("#reportJobId"),
    createReportButton: $("#createReportButton"),
    reportKpis: $("#reportKpis"),
    reportCount: $("#reportCount"),
    reportList: $("#reportList"),
    emptyReports: $("#emptyReports"),
    refreshReports: $("#refreshReports"),
    metricToggle: $("#metricToggle"),
    compareKpis: $("#compareKpis"),
    mapChart: $("#mapChart"),
    chartMeta: $("#chartMeta"),
    matrixRows: $("#matrixRows"),
    classChart: $("#classChart"),
    toast: $("#toast"),
    modal: $("#modal"),
    modalCard: $(".modal-card"),
    modalKicker: $("#modalKicker"),
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
    const names = {
      workbench: "质检工作台",
      history: "质检记录",
      compare: "模型实验",
      reports: "质检报告",
    };
    if (!names[view]) view = "workbench";
    state.view = view;
    $$(".nav-item").forEach((button) => button.classList.toggle("active", button.dataset.view === view));
    $$(".view").forEach((section) => section.classList.toggle("active", section.id === `view-${view}`));
    els.pageName.textContent = names[view];
    if (view === "history") renderHistory();
    if (view === "compare") renderCompare();
    if (view === "reports") renderReportPage();
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
    const absolute = new URL(url, window.location.href).href;
    const request = new Request(absolute);
    let cache = null;
    try {
      cache = "caches" in window ? await caches.open(MODEL_CACHE) : null;
      const cached = cache ? await cache.match(request) : null;
      if (cached) {
        updateBoot(60, "已从本地缓存读取", "正在装载检测模型");
        return cached.arrayBuffer();
      }
    } catch (error) {
      console.warn("模型缓存不可用，继续网络加载", error);
    }

    const controller = new AbortController();
    let stallTimer = null;
    const armStallTimer = () => {
      clearTimeout(stallTimer);
      stallTimer = setTimeout(() => controller.abort(), BOOT_STALL_MS);
    };
    try {
      armStallTimer();
      const response = await fetch(request, { cache: "force-cache", signal: controller.signal });
      if (!response.ok) throw new Error(`模型下载失败：HTTP ${response.status}`);
      const total = Number(response.headers.get("content-length")) || 11200000;
      if (!response.body || !response.body.getReader) {
        const buffer = await response.arrayBuffer();
        updateBoot(62, "权重已下载", "正在初始化 WASM 推理引擎…");
        if (cache) cache.put(request, new Response(buffer)).catch(() => {});
        return buffer;
      }
      const reader = response.body.getReader();
      const chunks = [];
      let received = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        armStallTimer();
        chunks.push(value);
        received += value.length;
        const pct = 8 + Math.min(1, received / total) * 48;
        updateBoot(pct, `${(received / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} MB`);
      }
      clearTimeout(stallTimer);
      const merged = new Uint8Array(received);
      let offset = 0;
      for (const chunk of chunks) {
        merged.set(chunk, offset);
        offset += chunk.length;
      }
      if (cache) {
        cache
          .put(request, new Response(merged, { headers: { "Content-Type": "application/octet-stream" } }))
          .catch((error) => console.warn("模型写入缓存失败", error));
      }
      updateBoot(62, "权重已下载", "正在初始化 WASM 推理引擎…");
      return merged.buffer;
    } catch (error) {
      if (error && error.name === "AbortError") {
        throw new Error("模型下载超时：网络长时间没有响应，请关闭重复标签页后重试");
      }
      throw error;
    } finally {
      clearTimeout(stallTimer);
    }
  }

  async function initModel() {
    if (state.modelLoading || state.modelReady) return;
    state.modelLoading = true;
    const startedAt = Date.now();
    const timer = setInterval(() => {
      const seconds = Math.round((Date.now() - startedAt) / 1000);
      els.bootElapsed.textContent = `${seconds}s`;
      if (seconds >= 20 && !state.modelReady) {
        els.bootHint.textContent = "网络较慢，仍在下载推理资源；可以先关闭重复打开的标签页，或稍后重试。";
      }
    }, 1000);
    try {
      if (!window.ort) throw new Error("ONNX Runtime 未加载");
      ort.env.wasm.wasmPaths = new URL("vendor/", window.location.href).href;
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      ort.env.logLevel = "error";
      els.bootRetry.classList.add("hidden");
      updateBoot(5, "连接模型资源", "正在装载检测模型");
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
      clearInterval(timer);
      els.bootElapsed.textContent = `${Math.round((Date.now() - startedAt) / 1000)}s`;
      setTimeout(hideBoot, 300);
      loadSample("crazing_101.jpg", true);
    } catch (error) {
      clearInterval(timer);
      state.modelLoading = false;
      console.error(error);
      updateBoot(100, "加载失败", "无法启动浏览器推理");
      setModelStatus("error", "模型加载失败", String(error.message || error));
      els.bootDetail.textContent = String(error.message || error);
      els.bootHint.textContent = "若多次重试仍失败，请检查网络后刷新；本地演示可运行 start_platform.bat。";
      els.bootRetry.classList.remove("hidden");
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
      state.currentRecordId = null;
      state.currentRecordCreatedAt = null;
      state.currentReview = null;
      state.currentSaved = false;
      els.originalImage.src = image.src;
      els.originalSize.textContent = `${image.naturalWidth} × ${image.naturalHeight}`;
      els.imageMeta.textContent = state.imageName;
      els.emptyViewer.classList.add("hidden");
      els.compareView.classList.remove("hidden");
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
    els.metricMain.textContent = "—";
    els.metricMainHint.textContent = "类别 / 数量";
    els.metricRisk.textContent = "—";
    els.metricRisk.dataset.risk = "none";
    els.metricRiskHint.textContent = "0–100 分";
    els.metricClasses.textContent = "0";
    els.metricConfidence.textContent = "—";
    els.metricTiming.textContent = "—";
    els.detailParams.textContent = `conf ${state.conf.toFixed(2)} · iou ${state.iou.toFixed(2)} · ${IMG_SIZE}px`;
    els.detectionRows.innerHTML = '<tr class="placeholder-row"><td colspan="5">还没有检测结果</td></tr>';
    els.classDistribution.innerHTML = '<div class="empty-mini">检测后显示各类别数量</div>';
    const ctx = els.resultCanvas.getContext("2d");
    if (state.image) {
      els.resultCanvas.width = state.image.naturalWidth;
      els.resultCanvas.height = state.image.naturalHeight;
      ctx.drawImage(state.image, 0, 0);
    }
    renderOutcomePanel();
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
    if (label) label.textContent = busy ? "正在质检…" : "开始质检";
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
      state.currentSaved = false;
      state.currentRecordId = null;
      state.currentRecordCreatedAt = null;
      state.currentReview = null;
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
      renderOutcomePanel();
      els.exportJson.disabled = false;
      if (state.view === "reports") renderReportPage();
    } catch (error) {
      console.error(error);
      showToast(`质检失败：${error.message || error}`, "error");
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
    const analysis = analyzeRecord({
      width,
      height,
      detections: state.detections,
      review: state.currentReview,
    });
    els.detectedCount.textContent = `${state.detections.length} 个目标`;
    els.timingBadge.textContent = `${Math.round(state.timing.total)} ms`;
    els.metricCount.textContent = String(state.detections.length);
    els.metricMain.textContent = analysis.classes.length ? analysis.classes[0].label : "未检出";
    els.metricMainHint.textContent = analysis.classes.length
      ? `${analysis.classes[0].count} 个 · 共 ${analysis.classes.length} 类`
      : "当前阈值下无目标";
    els.metricRisk.textContent = analysis.riskLabel;
    els.metricRisk.dataset.risk = analysis.riskLevel;
    els.metricRiskHint.textContent = `${analysis.score} 分 · ${analysis.riskMeaning}`;
    els.metricClasses.textContent = String(classSet.size);
    els.metricConfidence.textContent = state.detections.length ? avgConf.toFixed(3) : "—";
    els.metricTiming.textContent = `${Math.round(state.timing.total)} ms`;
    els.detailParams.textContent = `conf ${state.conf.toFixed(2)} · iou ${state.iou.toFixed(2)} · ${IMG_SIZE}px`;

    if (!state.detections.length) {
      els.detectionRows.innerHTML = '<tr class="placeholder-row"><td colspan="5">当前阈值下未检出目标</td></tr>';
      els.classDistribution.innerHTML = '<div class="empty-mini">当前阈值下未检出目标</div>';
      return;
    }

    els.detectionRows.innerHTML = state.detections
      .map((item, index) => {
        const color = CLASS_COLORS[item.cls];
        const pct = Math.round(item.conf * 100);
        const areaRatio =
          ((Number(item.x2) - Number(item.x1)) * (Number(item.y2) - Number(item.y1))) /
          Math.max(1, width * height);
        return `<tr>
          <td class="idx">${String(index + 1).padStart(2, "0")}</td>
          <td><span class="class-tag" style="--tag-color:${color}"><i></i>${escapeHtml(CLASS_ZH[CLASS_NAMES[item.cls]])}</span></td>
          <td><span class="confidence-bar" style="--tag-color:${color};--pct:${pct}%">${item.conf.toFixed(3)}</span></td>
          <td>${(areaRatio * 100).toFixed(1)}%</td>
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

  function renderOutcomePanel() {
    if (!els.outcomePanel) return;
    const ready = Boolean(state.lastRun && state.image);
    if (!ready) {
      els.outcomePanel.dataset.risk = "none";
      els.outcomeRisk.className = "risk-badge";
      els.outcomeRisk.textContent = "等待质检";
      els.outcomeSave.textContent = "保存记录";
      els.verdictScore.textContent = "—";
      els.verdictMainDefect.textContent = "—";
      els.verdictMaxBox.textContent = "—";
      els.verdictCoverage.textContent = "—";
      els.verdictConfidence.textContent = "—";
      els.verdictReview.className = "review-badge pending";
      els.verdictReview.textContent = "待复核";
      els.outcomeSummary.textContent =
        "选择或上传一张钢材表面图像，开始质检后这里会给出风险结论与处置建议。";
      els.verdictBasisBody.textContent =
        "完成质检后显示风险评分组成：类别权重、最大单框面积、总缺陷覆盖、缺陷数量与模型可靠性提示。";
      els.outcomeSave.disabled = true;
      els.reviewConfirm.disabled = true;
      els.reviewFalse.disabled = true;
      els.reviewDispose.disabled = true;
      els.outcomeReport.disabled = true;
      els.outcomeBatch.disabled = true;
      return;
    }

    const analysis = analyzeRecord({
      width: state.image.naturalWidth,
      height: state.image.naturalHeight,
      detections: state.detections,
      review: state.currentReview,
    });
    const risk = RISK_META[analysis.riskLevel] || RISK_META.none;
    const topClass = analysis.classes[0];
    els.outcomePanel.dataset.risk = analysis.riskLevel;
    els.outcomeRisk.className = `risk-badge ${analysis.riskLevel}`;
    els.outcomeRisk.textContent = risk.label;
    els.verdictScore.textContent = String(analysis.score);
    els.verdictMainDefect.textContent = topClass
      ? `${topClass.label} × ${topClass.count}`
      : "未检出";
    els.verdictMaxBox.textContent = analysis.count ? pct(analysis.maxBoxRatio) : "—";
    els.verdictCoverage.textContent = analysis.count ? pct(analysis.totalBoxRatio) : "—";
    els.verdictConfidence.textContent = analysis.count
      ? `${analysis.avgConf.toFixed(3)} · ${analysis.confidenceHint}`
      : "—";
    els.outcomeSummary.textContent = analysis.count
      ? `检出 ${analysis.count} 个目标，主要缺陷 ${topClass.label} × ${topClass.count}。${analysis.disposition}。`
      : `当前阈值下未检出目标。${analysis.disposition}。`;
    const review = reviewOf({ review: state.currentReview });
    const reviewMeta = REVIEW_META[review.status] || REVIEW_META.pending;
    els.verdictReview.className = `review-badge ${reviewMeta.className}`;
    els.verdictReview.textContent =
      review.status === "closed" && review.disposition
        ? `${reviewMeta.label} · ${review.disposition}`
        : reviewMeta.label;
    els.verdictBasisBody.innerHTML = `
      <ul class="basis-list">
        <li><span>类别风险</span><b>${topClass ? `${escapeHtml(topClass.label)} · ${analysis.classScore}/100` : "未检出 · 0/100"}</b></li>
        <li><span>最大单框面积</span><b>${analysis.count ? pct(analysis.maxBoxRatio) : "0.0%"}</b></li>
        <li><span>总缺陷覆盖</span><b>${analysis.count ? pct(analysis.totalBoxRatio) : "0.0%"}</b></li>
        <li><span>缺陷数量</span><b>${analysis.count} 个</b></li>
        <li><span>覆盖密度</span><b>${analysis.count ? `${analysis.densityScore}/100` : "0/100"}</b></li>
        <li class="basis-reliability"><span>模型可靠性</span><b>${escapeHtml(analysis.reliability)}</b></li>
      </ul>
      <p class="basis-formula">风险分 = 类别 ${Math.round(RISK_CONFIG.weights.class * 100)}% × ${analysis.classScore} + 最大单框 ${Math.round(RISK_CONFIG.weights.maxBox * 100)}% × ${analysis.maxBoxScore} + 总覆盖 ${Math.round(RISK_CONFIG.weights.totalBox * 100)}% × ${analysis.totalBoxScore} + 密度 ${Math.round(RISK_CONFIG.weights.density * 100)}% × ${analysis.densityScore} = <b>${analysis.score}</b>。阈值：≥${RISK_CONFIG.thresholds.high} 高风险，≥${RISK_CONFIG.thresholds.medium} 中风险。</p>
    `;
    els.outcomeSave.disabled = state.currentSaved;
    els.outcomeSave.textContent = state.currentSaved ? "已保存" : "保存记录";
    els.outcomeSave.title = state.currentSaved
      ? "当前结果已保存，复核状态会继续写回这条记录"
      : "保存当前质检记录到本地";
    const hasResult = state.detections.length > 0;
    els.reviewConfirm.disabled = !hasResult;
    els.reviewFalse.disabled = !hasResult;
    els.reviewDispose.disabled = false;
    els.outcomeReport.disabled = false;
    els.outcomeBatch.disabled = false;
  }

  function scheduleRerun() {
    if (!state.image || !state.modelReady) return;
    clearTimeout(state.rerunTimer);
    state.rerunTimer = setTimeout(runInference, 260);
  }

  function updateParamLabels() {
    els.confValue.textContent = state.conf.toFixed(2);
    els.iouValue.textContent = state.iou.toFixed(2);
    els.paramSummary.textContent = `conf ${state.conf.toFixed(2)} · iou ${state.iou.toFixed(2)}`;
    els.detailParams.textContent = `conf ${state.conf.toFixed(2)} · iou ${state.iou.toFixed(2)} · ${IMG_SIZE}px`;
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
        if (!db.objectStoreNames.contains(DB_REPORT_STORE)) {
          const reportStore = db.createObjectStore(DB_REPORT_STORE, { keyPath: "id" });
          reportStore.createIndex("createdAt", "createdAt");
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    return dbPromise;
  }

  async function dbGetAll(storeName = DB_STORE) {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, "readonly");
      const request = tx.objectStore(storeName).getAll();
      request.onsuccess = () => resolve(request.result || []);
      request.onerror = () => reject(request.error);
    });
  }

  async function dbPut(record, storeName = DB_STORE) {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, "readwrite");
      tx.objectStore(storeName).put(record);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async function dbDelete(id, storeName = DB_STORE) {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, "readwrite");
      tx.objectStore(storeName).delete(id);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async function dbClear(storeName = DB_STORE) {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, "readwrite");
      tx.objectStore(storeName).clear();
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async function saveCurrentRecord(options = {}) {
    if (!state.lastRun || !state.image) return null;
    try {
      const existing = state.currentRecordId
        ? state.history.find((item) => item.id === state.currentRecordId)
        : null;
      const id = state.currentRecordId || `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
      const createdAt = state.currentRecordCreatedAt || new Date().toISOString();
      const record = {
        id,
        createdAt,
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
        review: normalizeReviewForSave(state.currentReview || (existing && existing.review)),
      };
      await dbPut(record);
      state.currentRecordId = id;
      state.currentRecordCreatedAt = createdAt;
      state.currentReview = record.review;
      state.currentSaved = true;
      await loadHistory();
      if (!options.silent) showToast("已保存到质检记录");
      return record;
    } catch (error) {
      console.error(error);
      showToast("保存失败：浏览器可能禁用了本地存储", "error");
      return null;
    }
  }

  async function loadHistory() {
    try {
      const records = await dbGetAll();
      state.history = records
        .map((record) => ({ ...record, review: reviewOf(record) }))
        .sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)));
    } catch (error) {
      console.error(error);
      state.history = [];
    }
    const historyIds = new Set(state.history.map((record) => record.id));
    state.batchSelection.forEach((id) => {
      if (!historyIds.has(id)) state.batchSelection.delete(id);
    });
    if (state.currentRecordId) {
      const current = state.history.find((record) => record.id === state.currentRecordId);
      if (current) state.currentReview = reviewOf(current);
    }
    renderHistory();
    if (state.view === "reports") renderReportPage();
    renderOutcomePanel();
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
    const filtered = state.history.filter((record) => {
      if (!historyMatches(record, state.historyQuery)) return false;
      if (state.historyFilter === "all") return true;
      return reviewOf(record).status === state.historyFilter;
    });
    els.historyCount.textContent =
      state.historyFilter === "all" && !state.historyQuery
        ? `${filtered.length} 条记录`
        : `${filtered.length} / ${state.history.length} 条`;
    els.emptyHistory.classList.toggle("hidden", state.history.length > 0);
    els.historyList.classList.toggle("hidden", filtered.length === 0);
    $$("#historyFilters button").forEach((button) => {
      const active = button.dataset.filter === state.historyFilter;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    els.exportHistory.disabled = state.history.length === 0;
    els.clearHistory.disabled = state.history.length === 0;
    if (!filtered.length) {
      els.historyList.innerHTML =
        state.history.length > 0
          ? '<div class="filter-empty">没有符合当前搜索或复核状态筛选的记录。</div>'
          : "";
      return;
    }
    els.historyList.innerHTML = filtered
      .map((record) => {
        const date = new Date(record.createdAt);
        const dateText = Number.isNaN(date.getTime())
          ? record.createdAt
          : date.toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
        const classNames = [...new Set(record.detections.map((item) => CLASS_ZH[CLASS_NAMES[item.cls]]))];
        const analysis = analyzeRecord(record);
        const review = reviewOf(record);
        const reviewMeta = REVIEW_META[review.status] || REVIEW_META.pending;
        return `<article class="history-card" data-id="${escapeHtml(record.id)}">
          <div class="history-thumb"><img src="${record.annotated || record.original}" alt="" loading="lazy"></div>
          <div class="history-info">
            <div class="history-title">
              <strong>${escapeHtml(record.name)}</strong>
              <span class="history-date">${escapeHtml(dateText)}</span>
            </div>
            <div class="history-flags">
              ${riskBadge(analysis.riskLevel, analysis.riskLabel)}
              <span class="review-badge ${reviewMeta.className}">${escapeHtml(reviewMeta.label)}</span>
              ${review.disposition ? `<span class="disposition-chip">${escapeHtml(review.disposition)}</span>` : ""}
              ${review.reviewer ? `<span class="source-flag">复核 ${escapeHtml(review.reviewer)}</span>` : ""}
            </div>
            <div class="history-meta">
              <span class="mini-tag">${record.detections.length} 个目标</span>
              <span class="mini-tag">风险 ${analysis.score} 分</span>
              <span class="mini-tag">conf ${Number(record.conf).toFixed(2)}</span>
              <span>${escapeHtml(classNames.slice(0, 3).join(" · "))}${classNames.length > 3 ? " …" : ""}</span>
            </div>
          </div>
          <div class="history-actions">
            <button class="icon-button view" title="查看详情"><svg><use href="#i-eye"/></svg></button>
            <button class="icon-button review-action" title="复核 / 处置"><svg><use href="#i-review"/></svg></button>
            <button class="icon-button report-action" title="生成质检报告"><svg><use href="#i-report"/></svg></button>
            <button class="icon-button restore" title="重新载入"><svg><use href="#i-upload"/></svg></button>
            <button class="icon-button delete delete-action" title="删除"><svg><use href="#i-trash"/></svg></button>
          </div>
        </article>`;
      })
      .join("");
    $$(".history-card", els.historyList).forEach((card) => {
      const record = state.history.find((item) => item.id === card.dataset.id);
      $(".view", card).addEventListener("click", () => openRecordModal(record));
      $(".review-action", card).addEventListener("click", () => openReviewDialog(record, "confirmed"));
      $(".report-action", card).addEventListener("click", () =>
        createAndOpenReport([record], "single", { source: "质检记录", useForm: false })
      );
      $(".restore", card).addEventListener("click", () => restoreRecord(record));
      $(".delete-action", card).addEventListener("click", () => deleteRecord(record));
    });
    els.historyBatchReport.disabled = state.history.length === 0;
  }

  function openRecordModal(record) {
    if (!record) return;
    els.modalCard.classList.remove("report-modal-card");
    els.modalKicker.textContent = "RECORD";
    els.modalTitle.textContent = record.name;
    const date = new Date(record.createdAt);
    const dateText = Number.isNaN(date.getTime()) ? record.createdAt : date.toLocaleString("zh-CN");
    const review = reviewOf(record);
    const reviewMeta = REVIEW_META[review.status] || REVIEW_META.pending;
    const analysis = analyzeRecord(record);
    els.modalBody.innerHTML = `
      <div class="record-summary">
        <span>${escapeHtml(dateText)}</span>
        <span>${record.width} × ${record.height}</span>
        <span>conf ${Number(record.conf).toFixed(2)}</span>
        <span>iou ${Number(record.iou).toFixed(2)}</span>
        <span>${record.detections.length} 个目标</span>
        ${riskBadge(analysis.riskLevel, analysis.riskLabel)}
      </div>
      <div class="record-images">
        <figure><figcaption>原图</figcaption><img src="${record.original}" alt="原图"></figure>
        <figure><figcaption>标注结果</figcaption><img src="${record.annotated}" alt="检测结果"></figure>
      </div>
      <section class="record-review">
        <div class="record-review-head">
          <div><span class="panel-kicker">REVIEW &amp; DISPOSITION</span><h3>复核与处置</h3></div>
          <div class="record-review-actions">
            <span class="review-badge ${reviewMeta.className}">${escapeHtml(reviewMeta.label)}</span>
            <button class="secondary-button small" id="recordReviewButton" type="button">${review.status === "pending" ? "开始复核" : "修改复核"}</button>
          </div>
        </div>
        <div class="record-review-grid">
          <div><span>复核人</span><b>${escapeHtml(review.reviewer || "未填写")}</b></div>
          <div><span>处置结果</span><b>${escapeHtml(review.disposition || "未记录")}</b></div>
          <div><span>更新时间</span><b>${escapeHtml(review.updatedAt ? formatDateTime(review.updatedAt) : "未复核")}</b></div>
        </div>
        <p class="record-review-note">${escapeHtml(review.note || "暂无复核备注。")}</p>
        <p class="record-review-hint">复核状态只改变处置记录，不删除检测框；报告会保留生成时的复核快照。</p>
      </section>
      <table class="record-table">
        <thead><tr><th>#</th><th>类别</th><th>置信度</th><th>坐标</th></tr></thead>
        <tbody>${record.detections.map((item, index) => `<tr>
          <td>${index + 1}</td>
          <td>${escapeHtml(CLASS_ZH[CLASS_NAMES[item.cls]])}</td>
          <td>${Number(item.conf).toFixed(3)}</td>
          <td>${Number(item.x1).toFixed(1)}, ${Number(item.y1).toFixed(1)}, ${Number(item.x2).toFixed(1)}, ${Number(item.y2).toFixed(1)}</td>
        </tr>`).join("")}</tbody>
      </table>`;
    $("#recordReviewButton", els.modalBody).addEventListener("click", () =>
      openReviewDialog(record, review.status === "pending" ? "confirmed" : review.status)
    );
    els.modal.classList.add("open");
  }

  function closeModal() {
    els.modal.classList.remove("open");
    els.modalCard.classList.remove("report-modal-card");
  }

  async function ensureCurrentSaved() {
    if (!state.lastRun || !state.image) {
      showToast("请先完成一次质检", "error");
      return null;
    }
    if (state.currentRecordId) {
      const existing = state.history.find((record) => record.id === state.currentRecordId);
      if (existing) return existing;
    }
    return saveCurrentRecord({ silent: true });
  }

  function openReviewDialog(record, preferredStatus = "confirmed") {
    if (!record) return;
    const current = reviewOf(record);
    const status = ["confirmed", "false_positive", "closed"].includes(preferredStatus)
      ? preferredStatus
      : current.status === "pending"
        ? "confirmed"
        : current.status;
    const titles = {
      confirmed: "确认缺陷",
      false_positive: "标记误检",
      closed: "记录处置",
    };
    els.modalCard.classList.remove("report-modal-card");
    els.modalKicker.textContent = "REVIEW";
    els.modalTitle.textContent = `${titles[status] || "复核"} · ${record.name}`;
    els.modalBody.innerHTML = `
      <form class="review-form" id="reviewForm">
        <div class="review-intro">
          <div>
            <strong>复核对象</strong>
            <p>${escapeHtml(record.name)} · ${record.detections.length} 个检出框 · 风险 ${analyzeRecord(record).score} 分</p>
          </div>
          ${riskBadge(analyzeRecord(record).riskLevel, analyzeRecord(record).riskLabel)}
        </div>
        <fieldset class="review-status-field">
          <legend>复核状态</legend>
          <div class="review-status-options">
            <label class="review-option">
              <input type="radio" name="reviewStatus" value="confirmed" ${status === "confirmed" ? "checked" : ""}>
              <span>确认缺陷</span>
            </label>
            <label class="review-option">
              <input type="radio" name="reviewStatus" value="false_positive" ${status === "false_positive" ? "checked" : ""}>
              <span>标记误检</span>
            </label>
            <label class="review-option">
              <input type="radio" name="reviewStatus" value="closed" ${status === "closed" ? "checked" : ""}>
              <span>记录处置</span>
            </label>
          </div>
        </fieldset>
        <label class="review-field">
          <span>复核人 <em>可选</em></span>
          <input type="text" id="reviewReviewer" maxlength="40" autocomplete="name" value="${escapeHtml(current.reviewer)}" placeholder="例如：张工">
        </label>
        <label class="review-field">
          <span>处置结果 <em id="dispositionHint">记录处置时必填</em></span>
          <select id="reviewDisposition" ${status === "closed" ? "" : "disabled"}>
            <option value="">请选择处置结果</option>
            ${DISPOSITIONS.map(
              (item) =>
                `<option value="${escapeHtml(item)}" ${current.disposition === item && status === "closed" ? "selected" : ""}>${escapeHtml(item)}</option>`
            ).join("")}
          </select>
        </label>
        <label class="review-field">
          <span>备注 <em>可选，最多 300 字</em></span>
          <textarea id="reviewNote" rows="3" maxlength="300" placeholder="记录复核依据、返修方式或隔离原因">${escapeHtml(current.note)}</textarea>
        </label>
        <p class="review-form-note">标记误检只改变复核状态，不删除检测框。保存后自动记录当前浏览器时间，并写回该条质检记录。</p>
        <div class="review-form-actions">
          <button class="secondary-button" type="button" id="reviewCancel">取消</button>
          <button class="primary-button auto" type="submit" id="reviewSubmit">保存复核</button>
        </div>
      </form>
    `;
    const form = $("#reviewForm", els.modalBody);
    const dispositionSelect = $("#reviewDisposition", form);
    const syncDisposition = () => {
      const selected = form.querySelector('input[name="reviewStatus"]:checked');
      const closing = selected && selected.value === "closed";
      dispositionSelect.disabled = !closing;
      dispositionSelect.required = Boolean(closing);
      if (!closing) dispositionSelect.value = "";
    };
    form.querySelectorAll('input[name="reviewStatus"]').forEach((input) => {
      input.addEventListener("change", syncDisposition);
    });
    $("#reviewCancel", form).addEventListener("click", closeModal);
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const selected = form.querySelector('input[name="reviewStatus"]:checked');
      const nextStatus = selected ? selected.value : status;
      const disposition = nextStatus === "closed" ? dispositionSelect.value : "";
      if (nextStatus === "closed" && !disposition) {
        showToast("记录处置时必须选择处置结果", "error");
        dispositionSelect.focus();
        return;
      }
      const submit = $("#reviewSubmit", form);
      submit.disabled = true;
      await saveRecordReview(record.id, {
        status: nextStatus,
        disposition,
        reviewer: $("#reviewReviewer", form).value,
        note: $("#reviewNote", form).value,
      });
      submit.disabled = false;
    });
    els.modal.classList.add("open");
    $("#reviewReviewer", form).focus();
  }

  async function saveRecordReview(recordId, reviewInput) {
    const record = state.history.find((item) => item.id === recordId);
    if (!record) {
      showToast("记录不存在，请刷新后重试", "error");
      closeModal();
      return;
    }
    const review = normalizeReviewForSave({
      status: reviewInput.status,
      disposition: reviewInput.disposition,
      reviewer: reviewInput.reviewer,
      note: reviewInput.note,
      updatedAt: new Date().toISOString(),
    });
    try {
      await dbPut({ ...record, review });
      if (state.currentRecordId === recordId) {
        state.currentReview = review;
        state.currentSaved = true;
      }
      await loadHistory();
      closeModal();
      if (state.view === "reports") renderReportPage();
      const label = review.status === "closed" ? `已处置 · ${review.disposition}` : (REVIEW_META[review.status] || REVIEW_META.pending).label;
      showToast(`复核已保存：${label}`);
    } catch (error) {
      console.error(error);
      showToast("复核保存失败：浏览器本地存储不可用", "error");
    }
  }

  async function openCurrentReview(status) {
    if (!state.detections.length && status !== "closed") {
      showToast("当前未检出缺陷，可记录放行处置", "error");
      return;
    }
    const record = await ensureCurrentSaved();
    if (!record) return;
    const fresh = state.history.find((item) => item.id === record.id) || record;
    openReviewDialog(fresh, status);
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
      [
        "record_id", "created_at", "image", "width", "height", "conf", "iou",
        "risk_level", "risk_score", "review_status", "disposition", "reviewer", "review_note", "review_updated_at",
        "detection_index", "class", "confidence", "x1", "y1", "x2", "y2",
      ],
    ];
    for (const record of state.history) {
      const review = reviewOf(record);
      const analysis = analyzeRecord(record);
      const reviewCols = [
        analysis.riskLabel,
        analysis.score,
        (REVIEW_META[review.status] || REVIEW_META.pending).label,
        review.disposition,
        review.reviewer,
        review.note,
        review.updatedAt,
      ];
      if (!record.detections.length) {
        rows.push([
          record.id, record.createdAt, record.name, record.width, record.height, record.conf, record.iou,
          ...reviewCols, "", "", "", "", "", "", "",
        ]);
        continue;
      }
      record.detections.forEach((item, index) => {
        rows.push([
          record.id, record.createdAt, record.name, record.width, record.height, record.conf, record.iou,
          ...reviewCols,
          index + 1, CLASS_NAMES[item.cls], Number(item.conf.toFixed(6)).toString(),
          Number(item.x1.toFixed(2)).toString(), Number(item.y1.toFixed(2)).toString(),
          Number(item.x2.toFixed(2)).toString(), Number(item.y2.toFixed(2)).toString(),
        ]);
      });
    }
    const csv = rows
      .map((row) => row.map(csvCell).join(","))
      .join("\r\n");
    downloadFile("defectmind_history.csv", `\ufeff${csv}`, "text/csv;charset=utf-8");
    showToast("质检记录 CSV 已导出");
  }

  async function clearHistory() {
    if (!state.history.length) return;
    if (!window.confirm("确定清空当前浏览器中的全部检测记录吗？此操作不可撤销。")) return;
    try {
      await dbClear();
      await loadHistory();
      showToast("质检记录已清空");
    } catch (error) {
      showToast("清空失败", "error");
    }
  }

  function formatDateTime(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value || "—");
    return date.toLocaleString("zh-CN", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  function formatShortDate(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value || "—");
    return date.toLocaleString("zh-CN", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  function pct(value, digits = 1) {
    return `${(clamp(Number(value) || 0, 0, 1) * 100).toFixed(digits)}%`;
  }

  function reportIdNow() {
    const date = new Date();
    const pad = (value) => String(value).padStart(2, "0");
    const base = `QC-${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}-${pad(
      date.getHours()
    )}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
    let reportId = base;
    let suffix = 2;
    while (state.reports.some((report) => report.id === reportId)) {
      reportId = `${base}-${suffix}`;
      suffix += 1;
    }
    return reportId;
  }

  function stripExtension(name) {
    return String(name || "image").replace(/\.[^.]+$/, "");
  }

  function levelFromScore(score) {
    if (score >= RISK_CONFIG.thresholds.high) return "high";
    if (score >= RISK_CONFIG.thresholds.medium) return "medium";
    return score > 0 ? "low" : "none";
  }

  function classScoreOf(name) {
    return Number(RISK_CONFIG.classScore[name]) || 30;
  }

  function analyzeRecord(record) {
    const detections = Array.isArray(record.detections) ? record.detections : [];
    const width = Math.max(1, Number(record.width) || 1);
    const height = Math.max(1, Number(record.height) || 1);
    const imageArea = width * height;
    const byClass = new Map();
    let maxBoxRatio = 0;
    let totalBoxRatio = 0;
    let confidenceSum = 0;

    detections.forEach((detection) => {
      const boxWidth = Math.max(0, Number(detection.x2) - Number(detection.x1));
      const boxHeight = Math.max(0, Number(detection.y2) - Number(detection.y1));
      const areaRatio = imageArea > 0 ? (boxWidth * boxHeight) / imageArea : 0;
      maxBoxRatio = Math.max(maxBoxRatio, areaRatio);
      totalBoxRatio += areaRatio;
      confidenceSum += Number(detection.conf) || 0;
      const cls = Number(detection.cls);
      if (!byClass.has(cls)) {
        byClass.set(cls, {
          cls,
          name: CLASS_NAMES[cls] || String(cls),
          label: CLASS_ZH[CLASS_NAMES[cls]] || CLASS_NAMES[cls] || String(cls),
          count: 0,
          confidenceSum: 0,
          maxBoxRatio: 0,
        });
      }
      const item = byClass.get(cls);
      item.count += 1;
      item.confidenceSum += Number(detection.conf) || 0;
      item.maxBoxRatio = Math.max(item.maxBoxRatio, areaRatio);
    });

    const classes = Array.from(byClass.values())
      .map((item) => {
        const meta = CLASS_META[item.name] || { priority: 1, advice: "建议人工复核。" };
        const baseline = BASELINE_METRICS[item.name];
        const classScore = classScoreOf(item.name);
        return {
          cls: item.cls,
          name: item.name,
          label: item.label,
          count: item.count,
          avgConf: item.confidenceSum / item.count,
          maxBoxRatio: item.maxBoxRatio,
          priority: meta.priority,
          classScore,
          riskLevel: levelFromScore(classScore),
          advice: meta.advice,
          recall: baseline ? baseline.recall : null,
        };
      })
      .sort((a, b) => b.classScore - a.classScore || b.count - a.count);

    const avgConf = detections.length ? confidenceSum / detections.length : 0;
    const lowConfidence = detections.length > 0 && avgConf < RISK_CONFIG.confidence.low;
    const classScore = classes.length ? Math.max(...classes.map((item) => item.classScore)) : 0;
    const maxBoxScore = Math.round(clamp(maxBoxRatio / RISK_CONFIG.maxBoxFull, 0, 1) * 100);
    const totalBoxScore = Math.round(clamp(totalBoxRatio / RISK_CONFIG.totalBoxFull, 0, 1) * 100);
    const densityScore = detections.length
      ? Math.round(
          clamp(detections.length / RISK_CONFIG.densityCountFull, 0, 1) * 55 +
            clamp(totalBoxRatio / RISK_CONFIG.densityCoverageFull, 0, 1) * 45
        )
      : 0;
    const score = detections.length
      ? Math.round(
          classScore * RISK_CONFIG.weights.class +
            maxBoxScore * RISK_CONFIG.weights.maxBox +
            totalBoxScore * RISK_CONFIG.weights.totalBox +
            densityScore * RISK_CONFIG.weights.density
        )
      : 0;
    const riskLevel = levelFromScore(score);
    const risk = RISK_META[riskLevel];
    const topClass = classes[0];
    const confidenceHint = !detections.length
      ? "无目标"
      : avgConf < RISK_CONFIG.confidence.low
        ? "可靠性偏低"
        : avgConf < RISK_CONFIG.confidence.medium
          ? "可靠性一般"
          : "可靠性正常";
    const basis = topClass
      ? `${topClass.label}：类别风险权重 ${classScore}/100；最大单框面积 ${pct(
          maxBoxRatio
        )}；总缺陷面积 ${pct(totalBoxRatio)}；缺陷数量 ${detections.length}；风险分 ${score}（${risk.label}）。`
      : "未检出目标。";
    let reliability = "未检出目标；仍建议按产线规则抽检。";
    if (detections.length) {
      if (avgConf < RISK_CONFIG.confidence.low) {
        reliability = `平均置信度 ${avgConf.toFixed(
          3
        )}，低于 ${RISK_CONFIG.confidence.low.toFixed(2)}，模型可靠性偏低，存在误检或漏检风险，必须人工确认。`;
      } else if (avgConf < RISK_CONFIG.confidence.medium) {
        reliability = `平均置信度 ${avgConf.toFixed(3)}，模型可靠性一般，建议结合原图复核。`;
      } else {
        reliability = `平均置信度 ${avgConf.toFixed(3)}，模型可靠性正常，仍需按抽检规则复核。`;
      }
    }
    const disposition = detections.length
      ? `${risk.disposition}${lowConfidence ? "；模型可靠性偏低，需人工确认" : ""}`
      : risk.disposition;
    const review = reviewOf(record);

    return {
      recordId: record.id,
      name: record.name || "image.jpg",
      createdAt: record.createdAt,
      width,
      height,
      conf: Number(record.conf) || DEFAULT_CONF,
      iou: Number(record.iou) || DEFAULT_IOU,
      count: detections.length,
      classes,
      avgConf,
      maxBoxRatio,
      totalBoxRatio,
      lowConfidence,
      score,
      classScore,
      maxBoxScore,
      totalBoxScore,
      densityScore,
      riskLevel,
      riskLabel: risk.label,
      riskMeaning: risk.short,
      confidenceHint,
      disposition,
      basis,
      reliability,
      review,
    };
  }

  function cloneRecordForReport(record) {
    return {
      id: record.id,
      createdAt: record.createdAt,
      name: record.name,
      width: Number(record.width) || 0,
      height: Number(record.height) || 0,
      conf: Number(record.conf) || DEFAULT_CONF,
      iou: Number(record.iou) || DEFAULT_IOU,
      imgsz: Number(record.imgsz) || IMG_SIZE,
      timing: record.timing ? { ...record.timing } : null,
      detections: (record.detections || []).map((item) => ({ ...item })),
      original: record.original || "",
      annotated: record.annotated || record.original || "",
      review: record.review ? { ...record.review } : undefined,
    };
  }

  function currentRecordSnapshot() {
    if (!state.lastRun || !state.image) return null;
    return cloneRecordForReport({
      ...state.lastRun,
      id: `live-${Date.now()}`,
      name: state.imageName,
      width: state.image.naturalWidth,
      height: state.image.naturalHeight,
      conf: state.conf,
      iou: state.iou,
      imgsz: IMG_SIZE,
      timing: state.timing,
      detections: state.detections,
      review: state.currentReview,
      original: imageToDataUrl(state.image, 1100),
      annotated: els.resultCanvas.toDataURL("image/jpeg", 0.86),
    });
  }

  function aggregateClasses(analyses) {
    const byClass = new Map();
    analyses.forEach((analysis) => {
      analysis.classes.forEach((item) => {
        if (!byClass.has(item.cls)) {
          byClass.set(item.cls, {
            cls: item.cls,
            name: item.name,
            label: item.label,
            count: 0,
            images: 0,
            confidenceSum: 0,
            maxBoxRatio: 0,
            priority: item.priority,
            classScore: item.classScore,
            advice: item.advice,
            recall: item.recall,
          });
        }
        const target = byClass.get(item.cls);
        target.count += item.count;
        target.images += 1;
        target.confidenceSum += item.avgConf * item.count;
        target.maxBoxRatio = Math.max(target.maxBoxRatio, item.maxBoxRatio);
      });
    });
    return Array.from(byClass.values())
      .map((item) => ({
        ...item,
        avgConf: item.confidenceSum / Math.max(1, item.count),
        riskLevel: levelFromScore(item.classScore),
      }))
      .sort((a, b) => b.classScore - a.classScore || b.count - a.count);
  }

  function highestRisk(analyses) {
    let level = "none";
    analyses.forEach((analysis) => {
      if (RISK_META[analysis.riskLevel].order > RISK_META[level].order) level = analysis.riskLevel;
    });
    return level;
  }

  function collectRecommendations(classes) {
    if (!classes.length) {
      return [{ title: "未检出明显缺陷", text: "建议按常规频次抽检；如需产线判定，请先完成阈值标定。" }];
    }
    return classes.map((item) => {
      const recallNote =
        item.recall !== null && item.recall < 0.5
          ? ` 历史测试 Recall ${pct(item.recall)}，该类存在漏检风险，必须重点复核。`
          : "";
      return { title: item.label, text: `${item.advice}${recallNote}` };
    });
  }

  function buildReport(records, type, options = {}) {
    const storedRecords = records.map(cloneRecordForReport);
    const analyses = storedRecords.map(analyzeRecord);
    const riskCounts = { high: 0, medium: 0, low: 0, none: 0 };
    analyses.forEach((analysis) => {
      riskCounts[analysis.riskLevel] += 1;
    });
    const overallRisk = highestRisk(analyses);
    const overallScore = analyses.reduce((highest, analysis) => Math.max(highest, analysis.score), 0);
    const classSummary = aggregateClasses(analyses);
    const recommendations = collectRecommendations(classSummary);
    const reviewCounts = reviewStats(storedRecords);
    const highRiskRatio = storedRecords.length ? riskCounts.high / storedRecords.length : 0;
    const createdAt = new Date().toISOString();
    const confValues = [...new Set(storedRecords.map((record) => Number(record.conf).toFixed(2)))];
    const iouValues = [...new Set(storedRecords.map((record) => Number(record.iou).toFixed(2)))];
    const useForm = options.useForm !== false;
    const jobId = (options.jobId || (useForm ? els.reportJobId.value : "") || "").trim();
    const fallbackTitle =
      type === "single"
        ? `${stripExtension(storedRecords[0].name)} · 单图质检报告`
        : `批次质检报告 · ${formatShortDate(createdAt)}`;
    const title =
      (options.title || (useForm ? els.reportTitle.value : "") || "").trim() || fallbackTitle;
    const totalDetections = analyses.reduce((sum, analysis) => sum + analysis.count, 0);
    const averageMaxBoxRatio =
      analyses.length > 0
        ? analyses.reduce((sum, analysis) => sum + analysis.maxBoxRatio, 0) / analyses.length
        : 0;
    const disposition = RISK_META[overallRisk][type === "batch" ? "batchDisposition" : "disposition"];
    return {
      id: reportIdNow(),
      createdAt,
      type,
      title,
      jobId,
      source: options.source || (type === "batch" ? "质检记录批次" : "质检结果"),
      model: {
        name: "YOLO26n · ONNX Runtime Web",
        conf: confValues.length === 1 ? confValues[0] : "混合",
        iou: iouValues.length === 1 ? iouValues[0] : "混合",
        imgsz: IMG_SIZE,
        runtime: "浏览器本地推理 · WASM",
      },
      baseline: {
        precision: 0.7783,
        recall: 0.6651,
        map50: 0.7626,
        map5095: 0.4327,
      },
      records: storedRecords,
      analyses,
      riskCounts,
      reviewCounts,
      overallRisk,
      overallScore,
      highRiskRatio,
      totalDetections,
      averageMaxBoxRatio,
      classSummary,
      recommendations,
      disposition,
    };
  }

  async function loadReports() {
    try {
      state.reports = (await dbGetAll(DB_REPORT_STORE)).sort((a, b) =>
        String(b.createdAt).localeCompare(String(a.createdAt))
      );
    } catch (error) {
      console.error(error);
      state.reports = [];
    }
    state.reportsLoaded = true;
    if (state.view === "reports") renderReportPage();
  }

  function getReport(reportId) {
    return state.reports.find((report) => report.id === reportId);
  }

  function getSingleSourcePreviewRecord() {
    if (state.singleReportSource === "current" && state.lastRun && state.image) {
      return {
        ...state.lastRun,
        id: "current",
        name: state.imageName,
        width: state.image.naturalWidth,
        height: state.image.naturalHeight,
        conf: state.conf,
        iou: state.iou,
        imgsz: IMG_SIZE,
        review: state.currentReview,
        original: state.imageUrl,
        annotated: els.resultCanvas.toDataURL("image/jpeg", 0.8),
      };
    }
    return state.history.find((record) => record.id === state.singleReportSource) || null;
  }

  function riskBadge(level, label = "") {
    const risk = RISK_META[level] || RISK_META.none;
    return `<span class="risk-badge ${level}">${escapeHtml(label || risk.label)}</span>`;
  }

  function paperRisk(level, label = "") {
    const tones = {
      high: ["#c94848", "#ffeded", "#f3cccc"],
      medium: ["#b96724", "#fff2e8", "#f3d9c4"],
      low: ["#287a56", "#eaf7f0", "#cbe8da"],
      none: ["#6b7079", "#f1f2f4", "#dfe2e8"],
    };
    const risk = RISK_META[level] || RISK_META.none;
    const [color, background, border] = tones[level] || tones.none;
    return `<span style="display:inline-block;padding:1px 7px;border:1px solid ${border};border-radius:999px;background:${background};color:${color};font-size:8px;white-space:nowrap">${escapeHtml(
      label || risk.label
    )}</span>`;
  }

  function paperReview(status) {
    const tones = {
      pending: ["#7a641f", "#fbf6e8", "#eadfc0"],
      confirmed: ["#287a56", "#eaf7f0", "#cbe8da"],
      false_positive: ["#b05026", "#fff1ea", "#f0d3c5"],
      closed: ["#4057a8", "#edf1ff", "#ccd7f4"],
      unreviewed: ["#6b7079", "#f1f2f4", "#dfe2e8"],
    };
    const meta = REVIEW_META[status] || REVIEW_META.pending;
    const [color, background, border] = tones[status] || tones.pending;
    return `<span style="display:inline-block;padding:1px 7px;border:1px solid ${border};border-radius:999px;background:${background};color:${color};font-size:9px;white-space:nowrap">${escapeHtml(
      meta.label
    )}</span>`;
  }

  function paperReviewOfRecord(record) {
    const review = snapshotReviewOf(record);
    return review ? paperReview(review.status) : paperReview("unreviewed");
  }

  function renderReportKpis() {
    const coveredImages = state.reports.reduce((sum, report) => sum + report.records.length, 0);
    const highReports = state.reports.filter((report) => report.overallRisk === "high").length;
    const latest = state.reports[0];
    els.reportKpis.innerHTML = `
      <div class="report-kpi"><span>已归档报告</span><strong>${state.reports.length}</strong><small>单图与批次合计</small></div>
      <div class="report-kpi"><span>报告覆盖图片</span><strong>${coveredImages}</strong><small>同图可进入多份报告</small></div>
      <div class="report-kpi"><span>高风险报告</span><strong>${highReports}</strong><small>含至少一张高风险图片</small></div>
      <div class="report-kpi"><span>最近生成</span><strong>${latest ? escapeHtml(formatShortDate(latest.createdAt)) : "—"}</strong><small>${latest ? escapeHtml(latest.id) : "暂无报告"}</small></div>
    `;
  }

  function renderSingleReportPreview() {
    const record = getSingleSourcePreviewRecord();
    if (!record) {
      els.singleReportPreview.innerHTML =
        '<div class="source-empty">暂无可用于报告的质检结果。请先到质检工作台完成一次质检，或保存质检记录。</div>';
      return;
    }
    const analysis = analyzeRecord(record);
    const image = record.annotated || record.original || "";
    els.singleReportPreview.innerHTML = `
      <div class="source-card">
        ${image ? `<img src="${image}" alt="">` : ""}
        <div>
          <strong>${escapeHtml(record.name)}</strong>
          <p>${escapeHtml(formatDateTime(record.createdAt))} · ${analysis.count} 个目标 · ${analysis.classes.length} 类</p>
          <div class="source-flags">
            ${riskBadge(analysis.riskLevel, analysis.riskLabel)}
            ${reviewBadge(analysis.review.status)}
            <span class="source-flag">风险 ${analysis.score} 分</span>
            <span class="source-flag">平均置信度 ${analysis.avgConf.toFixed(3)}</span>
            <span class="source-flag">最大框面积 ${pct(analysis.maxBoxRatio)}</span>
          </div>
          <p>${escapeHtml(analysis.disposition)}</p>
        </div>
      </div>
    `;
  }

  function selectedBatchRecords() {
    return state.history.filter((record) => state.batchSelection.has(record.id));
  }

  function renderBatchReportPreview() {
    const records = selectedBatchRecords();
    if (!records.length) {
      els.batchReportPreview.innerHTML =
        '<div class="source-empty">勾选质检记录后，这里会显示批次风险汇总与建议处置。</div>';
      return;
    }
    const analyses = records.map(analyzeRecord);
    const counts = { high: 0, medium: 0, low: 0, none: 0 };
    analyses.forEach((analysis) => {
      counts[analysis.riskLevel] += 1;
    });
    const overall = highestRisk(analyses);
    const detections = analyses.reduce((sum, analysis) => sum + analysis.count, 0);
    const reviewCounts = reviewStats(records);
    const highRatio = records.length ? counts.high / records.length : 0;
    const classText = aggregateClasses(analyses)
      .slice(0, 4)
      .map((item) => `${item.label} ${item.count}`)
      .join(" · ");
    els.batchReportPreview.innerHTML = `
      <div class="batch-summary-grid">
        <div><span>已选图片</span><b>${records.length}</b></div>
        <div><span>检出目标</span><b>${detections}</b></div>
        <div><span>高风险图片</span><b>${counts.high}</b></div>
        <div><span>待复核</span><b>${reviewCounts.pending}</b></div>
      </div>
      <div class="source-flags">
        ${riskBadge(overall, RISK_META[overall].label)}
        <span class="source-flag">高风险占比 ${pct(highRatio)}</span>
        <span class="source-flag">中风险 ${counts.medium}</span>
        <span class="source-flag">低风险 ${counts.low}</span>
        <span class="source-flag">未检出 ${counts.none}</span>
      </div>
      <p class="batch-summary-note">${escapeHtml(classText || "未检出缺陷类别")}。${escapeHtml(
        RISK_META[overall].batchDisposition
      )}。批次结论按最严重单图风险兜底。</p>
    `;
  }

  function updateBatchSelectionUi() {
    const count = state.batchSelection.size;
    els.batchSelectionCount.textContent = `已选 ${count} 条 · 上限 ${MAX_BATCH_RECORDS}`;
    $$(".batch-record", els.batchRecordList).forEach((label) => {
      const checked = state.batchSelection.has(label.dataset.recordId);
      label.classList.toggle("selected", checked);
      const input = $('input[type="checkbox"]', label);
      if (input) input.checked = checked;
    });
    renderBatchReportPreview();
    els.createReportButton.disabled = state.reportMode === "batch" ? count === 0 : !getSingleSourcePreviewRecord();
  }

  function setReportMode(mode) {
    if (!["single", "batch"].includes(mode)) return;
    state.reportMode = mode;
    $$("button", els.reportMode).forEach((button) =>
      button.classList.toggle("active", button.dataset.mode === mode)
    );
    els.singleReportPane.classList.toggle("hidden", mode !== "single");
    els.batchReportPane.classList.toggle("hidden", mode !== "batch");
    els.reportTitle.placeholder =
      mode === "batch" ? "留空则按批次时间自动命名" : "留空则使用图片名自动命名";
    els.createReportButton.disabled =
      mode === "batch" ? state.batchSelection.size === 0 : !getSingleSourcePreviewRecord();
  }

  function renderReportSources() {
    const currentAvailable = Boolean(state.lastRun && state.image);
    const options = [];
    if (currentAvailable) {
      options.push({
        value: "current",
        label: `当前质检 · ${state.imageName} · ${state.detections.length} 个目标`,
      });
    }
    state.history.forEach((record) => {
      options.push({
        value: record.id,
        label: `${formatShortDate(record.createdAt)} · ${record.name} · ${record.detections.length} 个目标`,
      });
    });
    if (!options.length) {
      els.singleRecordSelect.innerHTML = '<option value="">暂无可选检测结果</option>';
      els.singleRecordSelect.disabled = true;
    } else {
      els.singleRecordSelect.disabled = false;
      els.singleRecordSelect.innerHTML = options
        .map((item) => `<option value="${escapeHtml(item.value)}">${escapeHtml(item.label)}</option>`)
        .join("");
      const valid = options.some((item) => item.value === state.singleReportSource);
      if (!valid) state.singleReportSource = options[0].value;
      els.singleRecordSelect.value = state.singleReportSource;
    }
    renderSingleReportPreview();

    if (!state.history.length) {
      els.batchRecordList.innerHTML =
        '<div class="source-empty">还没有质检记录。请先在质检工作台完成质检并保存，再生成批次报告。</div>';
    } else {
      els.batchRecordList.innerHTML = state.history
        .map((record) => {
          const analysis = analyzeRecord(record);
          return `<label class="batch-record ${
            state.batchSelection.has(record.id) ? "selected" : ""
          }" data-record-id="${escapeHtml(record.id)}">
            <input type="checkbox" ${state.batchSelection.has(record.id) ? "checked" : ""}>
            <img src="${record.annotated || record.original || ""}" alt="">
            <span class="batch-record-info">
              <strong>${escapeHtml(record.name)}</strong>
              <small>${escapeHtml(formatShortDate(record.createdAt))} · ${analysis.count} 个目标 · ${escapeHtml(
                analysis.classes
                  .slice(0, 2)
                  .map((item) => item.label)
                  .join(" / ") || "未检出"
              )}</small>
            </span>
            <span class="batch-record-flags">
              ${riskBadge(analysis.riskLevel, analysis.riskLabel)}
              ${reviewBadge(reviewOf(record).status)}
            </span>
          </label>`;
        })
        .join("");
    }
    updateBatchSelectionUi();
    setReportMode(state.reportMode);
  }

  function renderReportArchive() {
    els.reportCount.textContent = `${state.reports.length} 份`;
    els.emptyReports.classList.toggle("hidden", state.reports.length > 0);
    els.reportList.classList.toggle("hidden", state.reports.length === 0);
    if (!state.reports.length) {
      els.reportList.innerHTML = "";
      return;
    }
    els.reportList.innerHTML = state.reports
      .map((report) => {
        const typeLabel = report.type === "batch" ? "批次报告" : "单图报告";
        const risk = RISK_META[report.overallRisk] || RISK_META.none;
        const reviewCounts = report.reviewCounts || reviewStats(report.records || []);
        return `<article class="report-card" data-report-id="${escapeHtml(report.id)}">
          <div class="report-card-main">
            <div class="report-card-top">
              <span class="report-id">${escapeHtml(report.id)}</span>
              <span class="source-flag">${typeLabel}</span>
              ${riskBadge(report.overallRisk, risk.label)}
            </div>
            <h3>${escapeHtml(report.title)}</h3>
            <div class="report-card-meta">
              <span>${escapeHtml(formatDateTime(report.createdAt))}</span>
              <span>${report.records.length} 张图片</span>
              <span>${report.totalDetections} 个目标</span>
              ${Number.isFinite(report.overallScore) ? `<span>风险 ${report.overallScore} 分</span>` : ""}
              <span>待复核 ${reviewCounts.pending}</span>
              ${report.jobId ? `<span>工单 ${escapeHtml(report.jobId)}</span>` : ""}
              <span>${escapeHtml(risk.disposition)}</span>
            </div>
          </div>
          <div class="report-card-actions">
            <button class="icon-button preview" title="预览报告"><svg><use href="#i-eye"/></svg></button>
            <button class="icon-button print" title="打印 / 另存为 PDF"><svg><use href="#i-print"/></svg></button>
            <button class="icon-button csv" title="导出 CSV"><svg><use href="#i-download"/></svg></button>
            <button class="icon-button delete" title="删除报告"><svg><use href="#i-trash"/></svg></button>
          </div>
        </article>`;
      })
      .join("");
    $$(".report-card", els.reportList).forEach((card) => {
      const report = getReport(card.dataset.reportId);
      $(".preview", card).addEventListener("click", () => openReportModal(report.id));
      $(".print", card).addEventListener("click", () => printReport(report));
      $(".csv", card).addEventListener("click", () => exportReportCsv(report));
      $(".delete", card).addEventListener("click", () => deleteReport(report.id));
    });
  }

  function renderReportPage() {
    if (!els.reportKpis) return;
    renderReportKpis();
    renderReportSources();
    renderReportArchive();
  }

  function reportMetaRows(report) {
    return `<div class="report-paper-meta">
      <div><span>报告类型</span><b>${report.type === "batch" ? "批次质检报告" : "单图质检报告"}</b></div>
      <div><span>报告编号</span><b>${escapeHtml(report.id)}</b></div>
      <div><span>生成时间</span><b>${escapeHtml(formatDateTime(report.createdAt))}</b></div>
      <div><span>批次 / 工单</span><b>${escapeHtml(report.jobId || "未填写")}</b></div>
      <div><span>检测模型</span><b>${escapeHtml(report.model.name)}</b></div>
      <div><span>推理参数</span><b>conf ${escapeHtml(String(report.model.conf))} · iou ${escapeHtml(
        String(report.model.iou)
      )} · ${report.model.imgsz}px</b></div>
    </div>`;
  }

  function reportKpiBlock(report) {
    const reviewCounts = report.reviewCounts || reviewStats(report.records);
    return `<div class="report-paper-kpis">
      <div><span>图片数量</span><b>${report.records.length}</b></div>
      <div><span>检出目标</span><b>${report.totalDetections}</b></div>
      <div><span>高风险图片</span><b>${report.riskCounts.high}</b></div>
      <div><span>${reviewCounts.unreviewed ? "未复核" : "待复核"}</span><b>${
        reviewCounts.unreviewed || reviewCounts.pending
      }</b></div>
      <div><span>已处置</span><b>${reviewCounts.closed}</b></div>
    </div>`;
  }

  function reportReviewSection(report) {
    const records = report.records || [];
    const counts = report.reviewCounts || reviewStats(records);
    if (report.type === "single") {
      const record = records[0] || {};
      const review = snapshotReviewOf(record);
      const meta = REVIEW_META[review ? review.status : "unreviewed"];
      return `<div class="report-paper-section">
        <h3>复核与处置</h3>
        <div class="report-paper-kpis report-review-stats">
          <div><span>复核状态</span><b style="font-size:13px">${escapeHtml(meta.label)}</b></div>
          <div><span>复核人</span><b style="font-size:13px">${escapeHtml(review && review.reviewer ? review.reviewer : "未填写")}</b></div>
          <div><span>处置结果</span><b style="font-size:13px">${escapeHtml(review && review.disposition ? review.disposition : "未记录")}</b></div>
          <div><span>更新时间</span><b style="font-size:11px">${escapeHtml(
            review && review.updatedAt ? formatDateTime(review.updatedAt) : "未复核"
          )}</b></div>
        </div>
        <p class="report-review-note">${escapeHtml(
          review && review.note ? review.note : "暂无复核备注。报告为生成时快照，后续记录变化不会回写本报告。"
        )}</p>
      </div>`;
    }
    const rows = records
      .map((record, index) => {
        const review = snapshotReviewOf(record);
        const meta = REVIEW_META[review ? review.status : "unreviewed"];
        return `<tr>
          <td>${index + 1}</td>
          <td><strong>${escapeHtml(record.name)}</strong></td>
          <td>${paperReviewOfRecord(record)}</td>
          <td>${escapeHtml(review && review.reviewer ? review.reviewer : "—")}</td>
          <td>${escapeHtml(review && review.disposition ? review.disposition : "—")}</td>
          <td>${escapeHtml(review && review.updatedAt ? formatDateTime(review.updatedAt) : "未复核")}</td>
        </tr>`;
      })
      .join("");
    return `<div class="report-paper-section">
      <h3>复核与处置</h3>
      <div class="report-review-stats">
        <span>待复核 <b>${counts.pending}</b></span>
        <span>已确认 <b>${counts.confirmed}</b></span>
        <span>误检 <b>${counts.false_positive}</b></span>
        <span>已处置 <b>${counts.closed}</b></span>
        ${counts.unreviewed ? `<span>未复核 <b>${counts.unreviewed}</b></span>` : ""}
      </div>
      <table class="report-paper-table">
        <thead><tr><th>#</th><th>图片</th><th>复核状态</th><th>复核人</th><th>处置</th><th>更新时间</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <p class="report-review-note">报告保留生成时的复核快照；生成后的记录变更不会自动改写本报告。</p>
    </div>`;
  }

  function reportSingleDetails(report) {
    const record = report.records[0];
    const analysis = report.analyses[0];
    const detectionRows = record.detections.length
      ? record.detections
          .map(
            (item, index) => `<tr>
              <td>${String(index + 1).padStart(2, "0")}</td>
              <td><strong>${escapeHtml(CLASS_ZH[CLASS_NAMES[item.cls]] || CLASS_NAMES[item.cls])}</strong></td>
              <td>${Number(item.conf).toFixed(3)}</td>
              <td>${Number(item.x1).toFixed(1)}, ${Number(item.y1).toFixed(1)}, ${Number(item.x2).toFixed(
                1
              )}, ${Number(item.y2).toFixed(1)}</td>
            </tr>`
          )
          .join("")
      : '<tr><td colspan="4">当前阈值下未检出目标。</td></tr>';
    const classRows = analysis.classes.length
      ? analysis.classes
          .map(
            (item) => `<tr>
              <td><strong>${escapeHtml(item.label)}</strong><br><span style="color:#8a9099">${escapeHtml(
                item.name
              )}</span></td>
              <td>${item.count}</td>
              <td>${item.avgConf.toFixed(3)}</td>
              <td>${pct(item.maxBoxRatio)}</td>
              <td>${paperRisk(item.riskLevel)}</td>
            </tr>`
          )
          .join("")
      : '<tr><td colspan="5">无检出类别。</td></tr>';
    return `
      <div class="report-paper-section">
        <h3>检测结果</h3>
        <div class="report-paper-kpis">
          <div><span>图片尺寸</span><b style="font-size:13px">${record.width}×${record.height}</b></div>
          <div><span>平均置信度</span><b>${analysis.avgConf.toFixed(3)}</b></div>
          <div><span>最大单框面积</span><b style="font-size:14px">${pct(analysis.maxBoxRatio)}</b></div>
          <div><span>图像级风险</span><b style="font-size:13px">${escapeHtml(analysis.riskLabel)} · ${analysis.score} 分</b></div>
        </div>
      </div>
      <div class="report-paper-section">
        <h3>原图与检测结果</h3>
        <div class="report-paper-images">
          <figure><figcaption>原图</figcaption>${record.original ? `<img src="${record.original}" alt="">` : ""}</figure>
          <figure><figcaption>检测结果</figcaption>${record.annotated ? `<img src="${record.annotated}" alt="">` : ""}</figure>
        </div>
      </div>
      <div class="report-paper-section">
        <h3>缺陷明细</h3>
        <table class="report-paper-table">
          <thead><tr><th>#</th><th>类别</th><th>置信度</th><th>位置 x1, y1, x2, y2</th></tr></thead>
          <tbody>${detectionRows}</tbody>
        </table>
      </div>
      <div class="report-paper-section">
        <h3>分类别风险</h3>
        <table class="report-paper-table">
          <thead><tr><th>类别</th><th>数量</th><th>平均置信度</th><th>最大框面积</th><th>风险</th></tr></thead>
          <tbody>${classRows}</tbody>
        </table>
      </div>`;
  }

  function reportBatchDetails(report) {
    const imageRows = report.records
      .map((record, index) => {
        const analysis = report.analyses[index];
        return `<tr>
          <td>${index + 1}</td>
          <td><strong>${escapeHtml(record.name)}</strong><br><span style="color:#8a9099">${escapeHtml(
            formatShortDate(record.createdAt)
          )}</span></td>
          <td>${analysis.count}</td>
          <td>${analysis.avgConf.toFixed(3)}</td>
          <td>${pct(analysis.maxBoxRatio)}</td>
          <td>${analysis.score}</td>
          <td>${paperRisk(analysis.riskLevel, analysis.riskLabel)}</td>
          <td>${paperReviewOfRecord(record)}</td>
          <td>${escapeHtml(analysis.disposition)}</td>
        </tr>`;
      })
      .join("");
    const classRows = report.classSummary.length
      ? report.classSummary
          .map(
            (item) => `<tr>
              <td><strong>${escapeHtml(item.label)}</strong><br><span style="color:#8a9099">${escapeHtml(
                item.name
              )}</span></td>
              <td>${item.count}</td>
              <td>${item.images}</td>
              <td>${item.avgConf.toFixed(3)}</td>
              <td>${pct(item.maxBoxRatio)}</td>
              <td>${paperRisk(item.riskLevel)}</td>
            </tr>`
          )
          .join("")
      : '<tr><td colspan="6">未检出缺陷类别。</td></tr>';
    const examples = report.records.slice(0, 6).map(
      (record) => `<figure><figcaption>${escapeHtml(record.name)}</figcaption><img src="${
        record.annotated || record.original || ""
      }" alt=""></figure>`
    );
    return `
      <div class="report-paper-section">
        <h3>批次图片结果</h3>
        <table class="report-paper-table">
          <thead><tr><th>#</th><th>图片</th><th>目标</th><th>平均置信度</th><th>最大框面积</th><th>风险分</th><th>风险</th><th>复核</th><th>建议</th></tr></thead>
          <tbody>${imageRows}</tbody>
        </table>
      </div>
      <div class="report-paper-section">
        <h3>缺陷类别汇总</h3>
        <table class="report-paper-table">
          <thead><tr><th>类别</th><th>目标数</th><th>涉及图片</th><th>平均置信度</th><th>最大框面积</th><th>风险</th></tr></thead>
          <tbody>${classRows}</tbody>
        </table>
      </div>
      ${
        examples.length
          ? `<div class="report-paper-section"><h3>批次示例（最多 6 张）</h3><div class="report-paper-images">${examples.join(
              ""
            )}</div></div>`
          : ""
      }`;
  }

  function renderReportPaper(report) {
    const risk = RISK_META[report.overallRisk] || RISK_META.none;
    const calloutTone = report.overallRisk === "high" ? "danger" : report.overallRisk === "medium" ? "warn" : "";
    const reviewCounts = report.reviewCounts || reviewStats(report.records);
    const highRatio = Number.isFinite(report.highRiskRatio)
      ? report.highRiskRatio
      : report.records.length
        ? report.riskCounts.high / report.records.length
        : 0;
    const recs = report.recommendations
      .map(
        (item) =>
          `<li><b>${escapeHtml(item.title)}：</b>${escapeHtml(item.text)}</li>`
      )
      .join("");
    return `<div class="report-paper">
      <div class="report-paper-head">
        <div>
          <div class="report-paper-brand">DEFECTMIND · QUALITY INSPECTION</div>
          <h2>${escapeHtml(report.title)}</h2>
          <p class="report-paper-sub">钢材表面缺陷图像级辅助质检报告 · ${escapeHtml(report.source)}</p>
        </div>
        <div class="report-paper-id">
          <b>${escapeHtml(report.id)}</b>
          ${escapeHtml(formatDateTime(report.createdAt))}<br>
          ${report.jobId ? `工单：${escapeHtml(report.jobId)}` : "未填写工单号"}
        </div>
      </div>

      <div class="report-paper-section">${reportMetaRows(report)}</div>
      <div class="report-paper-section">${reportKpiBlock(report)}</div>

      <div class="report-paper-section">
        <h3>综合结论</h3>
        <div class="report-callout ${calloutTone}">
          <b>${escapeHtml(risk.label)} · ${escapeHtml(risk.disposition)}</b>
          <p>${escapeHtml(
            report.analyses.length === 1
              ? `${report.analyses[0].basis}${report.analyses[0].reliability}`
              : `本批次共 ${report.records.length} 张图片，其中高风险 ${report.riskCounts.high} 张（${pct(
                  highRatio
                )}）、中风险 ${report.riskCounts.medium} 张、低风险 ${report.riskCounts.low} 张、未检出 ${
                  report.riskCounts.none
                } 张；待复核 ${reviewCounts.pending} 张；最高单图风险分 ${
                  Number.isFinite(report.overallScore) ? report.overallScore : "见明细"
                }。${risk.batchDisposition}。`
          )}</p>
        </div>
      </div>

      ${reportReviewSection(report)}

      ${report.type === "single" ? reportSingleDetails(report) : reportBatchDetails(report)}

      <div class="report-paper-section">
        <h3>参考处置建议</h3>
        <ul class="report-paper-list">${recs}</ul>
      </div>

      <div class="report-paper-foot">
        方法说明：风险分为图像级启发式提示，按“类别权重 42% + 最大单框面积 24% + 总缺陷覆盖 16% + 覆盖密度 18%”计算，阈值 ≥${RISK_CONFIG.thresholds.high} 为高风险、≥${RISK_CONFIG.thresholds.medium} 为中风险；不代表真实缺陷深度或产品寿命。平均置信度只用于可靠性提示，不参与质量严重度主评分。结论使用“建议放行 / 建议复检 / 建议拦截”，不替代人工终判。<br>
        模型边界：YOLO26n 在 test 集上 Precision ${pct(report.baseline.precision)}、Recall ${pct(
      report.baseline.recall
    )}、mAP50-95 ${report.baseline.map5095.toFixed(4)}；龟裂类历史 Recall 为 23.4%，漏检风险较高。本报告由浏览器本地生成，图片与记录未上传服务器。
      </div>
    </div>`;
  }

  function openReportModal(reportId) {
    const report = getReport(reportId);
    if (!report) {
      showToast("报告不存在或已被删除", "error");
      return;
    }
    els.modalCard.classList.add("report-modal-card");
    els.modalKicker.textContent = "QUALITY REPORT";
    els.modalTitle.textContent = `${report.type === "batch" ? "批次" : "单图"}质检报告 · ${report.id}`;
    els.modalBody.innerHTML = `
      <div class="report-actions">
        <button class="primary-button" id="modalPrintReport"><svg><use href="#i-print"/></svg>打印 / 另存为 PDF</button>
        <button class="secondary-button" id="modalExportReportCsv"><svg><use href="#i-download"/></svg>导出 CSV</button>
      </div>
      ${renderReportPaper(report)}
    `;
    $("#modalPrintReport", els.modalBody).addEventListener("click", () => printReport(report));
    $("#modalExportReportCsv", els.modalBody).addEventListener("click", () => exportReportCsv(report));
    els.modal.classList.add("open");
  }

  function printReport(report) {
    if (!report) return;
    const win = window.open("", "_blank");
    if (!win) {
      showToast("浏览器拦截了报告窗口，请允许弹窗后重试", "error");
      return;
    }
    const base = escapeHtml(window.location.href);
    const paper = renderReportPaper(report);
    win.document.write(`<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <base href="${base}">
  <title>${escapeHtml(report.title)} · ${escapeHtml(report.id)}</title>
  <link rel="stylesheet" href="styles.css?v=5">
  <style>
    html{background:#fff;color-scheme:light}
    body{background:#e9ebef;margin:0;padding:0;color:#16181c}
    .print-shell{width:min(920px,calc(100% - 32px));margin:0 auto;padding:24px 0 44px}
    .print-toolbar{position:sticky;top:0;z-index:10;display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:14px;padding:9px 12px;border:1px solid #d5d9e0;border-radius:7px;background:rgba(255,255,255,.94);box-shadow:0 6px 18px rgba(20,24,32,.08);font:12px "Microsoft YaHei UI","PingFang SC",sans-serif;color:#4f555f;backdrop-filter:blur(12px)}
    .print-toolbar button{padding:7px 12px;border:0;border-radius:5px;background:#5e6ad2;color:#fff;font:inherit;cursor:pointer}
    .report-paper{box-shadow:0 12px 34px rgba(20,24,32,.12);border-radius:4px}
    @page{size:A4;margin:13mm}
    @media print{
      html,body{background:#fff !important;color-scheme:light !important}
      .print-shell{width:auto;margin:0;padding:0}
      .print-toolbar{display:none}
      .report-paper{box-shadow:none;border-radius:0;padding:0;background:#fff}
      .report-paper-section{break-inside:avoid}
      .report-paper-images{break-inside:avoid}
      .report-paper-table{break-inside:auto}
      tr{break-inside:avoid}
    }
  </style>
</head>
<body>
  <div class="print-shell">
    <div class="print-toolbar">
      <span>在打印窗口中选择“另存为 PDF”即可导出。图片与数据仅在本机处理。</span>
      <button onclick="window.print()">打印 / 另存为 PDF</button>
    </div>
    ${paper}
  </div>
</body>
</html>`);
    win.document.close();
    win.focus();
    showToast("已打开打印页，请选择“另存为 PDF”");
  }

  function csvCell(value) {
    let text = String(value == null ? "" : value);
    if (/^[=+\-@]/.test(text)) text = `'${text}`;
    return `"${text.replace(/"/g, '""')}"`;
  }

  function exportReportCsv(report) {
    if (!report) return;
    const rows = [
      [
        "报告编号",
        "报告标题",
        "报告类型",
        "生成时间",
        "批次工单",
        "报告风险",
        "建议处置",
        "图片",
        "图片时间",
        "宽度",
        "高度",
        "图片风险",
        "图片建议",
        "复核状态",
        "复核人",
        "处置结果",
        "复核时间",
        "类别",
        "置信度",
        "x1",
        "y1",
        "x2",
        "y2",
      ],
    ];
    report.records.forEach((record, index) => {
      const analysis = report.analyses[index];
      const review = snapshotReviewOf(record);
      const reviewCols = [
        (REVIEW_META[review ? review.status : "unreviewed"] || REVIEW_META.unreviewed).label,
        review ? review.reviewer : "",
        review ? review.disposition : "",
        review ? review.updatedAt : "",
      ];
      if (!record.detections.length) {
        rows.push([
          report.id,
          report.title,
          report.type === "batch" ? "批次报告" : "单图报告",
          report.createdAt,
          report.jobId,
          RISK_META[report.overallRisk].label,
          RISK_META[report.overallRisk].disposition,
          record.name,
          record.createdAt,
          record.width,
          record.height,
          analysis.riskLabel,
          analysis.disposition,
          ...reviewCols,
          "",
          "",
          "",
          "",
          "",
          "",
        ]);
        return;
      }
      record.detections.forEach((item) => {
        rows.push([
          report.id,
          report.title,
          report.type === "batch" ? "批次报告" : "单图报告",
          report.createdAt,
          report.jobId,
          RISK_META[report.overallRisk].label,
          RISK_META[report.overallRisk].disposition,
          record.name,
          record.createdAt,
          record.width,
          record.height,
          analysis.riskLabel,
          analysis.disposition,
          ...reviewCols,
          CLASS_NAMES[item.cls],
          Number(item.conf).toFixed(6),
          Number(item.x1).toFixed(2),
          Number(item.y1).toFixed(2),
          Number(item.x2).toFixed(2),
          Number(item.y2).toFixed(2),
        ]);
      });
    });
    const csv = rows.map((row) => row.map(csvCell).join(",")).join("\r\n");
    downloadFile(`${report.id}.csv`, `\ufeff${csv}`, "text/csv;charset=utf-8");
    showToast("报告 CSV 已导出，可直接用 Excel 打开");
  }

  async function deleteReport(reportId) {
    const report = getReport(reportId);
    if (!report) return;
    if (!window.confirm(`确定删除报告 ${report.id} 吗？此操作不会删除原始检测记录。`)) return;
    try {
      await dbDelete(report.id, DB_REPORT_STORE);
      await loadReports();
      showToast("报告已删除");
    } catch (error) {
      console.error(error);
      showToast("删除报告失败", "error");
    }
  }

  async function createAndOpenReport(records, type, options = {}) {
    const usable = records.filter(Boolean);
    if (!usable.length) {
      showToast("没有可用于生成报告的数据", "error");
      return;
    }
    els.createReportButton.disabled = true;
    try {
      const report = buildReport(usable, type, options);
      await dbPut(report, DB_REPORT_STORE);
      await loadReports();
      els.reportTitle.value = "";
      els.reportJobId.value = "";
      openReportModal(report.id);
      showToast(`报告 ${report.id} 已归档`);
    } catch (error) {
      console.error(error);
      showToast("报告归档失败：浏览器本地存储空间可能不足", "error");
    } finally {
      setReportMode(state.reportMode);
    }
  }

  async function createReportFromBuilder() {
    if (state.reportMode === "batch") {
      createAndOpenReport(selectedBatchRecords(), "batch");
      return;
    }
    if (state.singleReportSource === "current") {
      const saved = await ensureCurrentSaved();
      const record = saved || currentRecordSnapshot();
      if (!record) {
        showToast("当前没有可用的检测结果", "error");
        return;
      }
      createAndOpenReport([record], "single", { source: saved ? "质检记录" : "当前检测结果（未归档）" });
      return;
    }
    const record = state.history.find((item) => item.id === state.singleReportSource);
    if (!record) {
      showToast("所选质检记录不存在", "error");
      return;
    }
    createAndOpenReport([record], "single", { source: "质检记录" });
  }

  async function generateReportFromCurrent() {
    const saved = await ensureCurrentSaved();
    const record = saved || currentRecordSnapshot();
    if (!record) {
      showToast("请先完成一次质检", "error");
      return;
    }
    createAndOpenReport([record], "single", {
      source: saved ? "质检记录" : "当前检测结果（未归档）",
      useForm: false,
    });
  }

  function openReportBuilder(mode) {
    if (mode === "batch" && state.batchSelection.size === 0 && state.history.length) {
      state.batchSelection = new Set(state.history.slice(0, 5).map((record) => record.id));
    }
    setView("reports");
    setReportMode(mode);
    updateBatchSelectionUi();
    requestAnimationFrame(() => {
      $(".report-builder").scrollIntoView({ behavior: "smooth", block: "start" });
    });
    if (mode === "batch") {
      showToast(`已选 ${state.batchSelection.size} 条质检记录，单份批次最多 ${MAX_BATCH_RECORDS} 条`);
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
    els.bootRetry.addEventListener("click", () => window.location.reload());
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
    els.outcomeSave.addEventListener("click", async () => {
      await saveCurrentRecord();
      renderOutcomePanel();
    });
    els.reviewConfirm.addEventListener("click", () => openCurrentReview("confirmed"));
    els.reviewFalse.addEventListener("click", () => openCurrentReview("false_positive"));
    els.reviewDispose.addEventListener("click", () => openCurrentReview("closed"));
    els.outcomeReport.addEventListener("click", generateReportFromCurrent);
    els.outcomeBatch.addEventListener("click", async () => {
      const saved = await ensureCurrentSaved();
      if (!saved) return;
      state.batchSelection.add(saved.id);
      openReportBuilder("batch");
    });
    els.resetParams.addEventListener("click", resetParams);
    els.exportJson.addEventListener("click", exportCurrentJson);
    els.historySearch.addEventListener("input", () => {
      state.historyQuery = els.historySearch.value.trim();
      renderHistory();
    });
    els.historyFilters.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-filter]");
      if (!button) return;
      state.historyFilter = button.dataset.filter;
      renderHistory();
    });
    els.exportHistory.addEventListener("click", exportHistoryCsv);
    els.clearHistory.addEventListener("click", clearHistory);
    els.historyBatchReport.addEventListener("click", () => openReportBuilder("batch"));
    els.refreshReports.addEventListener("click", async () => {
      await loadReports();
      showToast("报告归档已刷新");
    });
    els.reportMode.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-mode]");
      if (!button) return;
      setReportMode(button.dataset.mode);
    });
    els.singleRecordSelect.addEventListener("change", () => {
      state.singleReportSource = els.singleRecordSelect.value;
      renderSingleReportPreview();
      els.createReportButton.disabled = !getSingleSourcePreviewRecord();
    });
    els.batchRecordList.addEventListener("change", (event) => {
      const checkbox = event.target.closest('input[type="checkbox"]');
      if (!checkbox) return;
      const row = checkbox.closest(".batch-record");
      if (!row) return;
      if (checkbox.checked) {
        if (state.batchSelection.size >= MAX_BATCH_RECORDS) {
          checkbox.checked = false;
          showToast(`单份批次报告最多选择 ${MAX_BATCH_RECORDS} 条记录`, "error");
          return;
        }
        state.batchSelection.add(row.dataset.recordId);
      } else {
        state.batchSelection.delete(row.dataset.recordId);
      }
      updateBatchSelectionUi();
    });
    els.selectRecentRecords.addEventListener("click", () => {
      state.batchSelection = new Set(state.history.slice(0, 5).map((record) => record.id));
      updateBatchSelectionUi();
    });
    els.selectAllRecords.addEventListener("click", () => {
      state.batchSelection = new Set(
        state.history.slice(0, MAX_BATCH_RECORDS).map((record) => record.id)
      );
      updateBatchSelectionUi();
      if (state.history.length > MAX_BATCH_RECORDS) {
        showToast(`已选择最近 ${MAX_BATCH_RECORDS} 条，单个批次报告最多 ${MAX_BATCH_RECORDS} 条`);
      }
    });
    els.clearSelectedRecords.addEventListener("click", () => {
      state.batchSelection.clear();
      updateBatchSelectionUi();
    });
    els.createReportButton.addEventListener("click", createReportFromBuilder);
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
      if (event.key === "3") setView("reports");
      if (event.key === "4") setView("compare");
      if (event.key === "Enter" && state.view === "workbench") runInference();
      if (event.key === "Escape") closeModal();
    });
    window.addEventListener("hashchange", () => {
      const view = window.location.hash.replace("#", "");
      if (["workbench", "history", "compare", "reports"].includes(view)) setView(view, false);
    });
  }

  async function boot() {
    buildSampleGrid();
    updateParamLabels();
    bindEvents();
    const initialView = window.location.hash.replace("#", "");
    setView(
      ["workbench", "history", "compare", "reports"].includes(initialView) ? initialView : "workbench",
      false
    );
    await loadHistory();
    await loadReports();
    renderCompare();
    initModel();
  }

  boot();
})();
