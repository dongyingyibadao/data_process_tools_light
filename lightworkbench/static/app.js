"use strict";

const $ = (selector) => document.querySelector(selector);
const state = {
  root: "", path: "", listing: null, episode: "", detail: null, stream: null,
  ranges: [], history: [], frame: 0, preview: false, dragStart: null, dragEnd: null,
  nextKeptFrame: null, requestSerial: 0, operationId: null, pendingMode: null,
  operations: new Map(), eventSources: new Map(),
};
const rowHeight = 48;

const sourceRoot = $("#sourceRoot");
const outputRoot = $("#outputRoot");
const operator = $("#operator");
outputRoot.value = localStorage.getItem("light.outputRoot") || "";
operator.value = localStorage.getItem("light.operator") || "";
outputRoot.addEventListener("change", () => localStorage.setItem("light.outputRoot", outputRoot.value.trim()));
operator.addEventListener("change", () => localStorage.setItem("light.operator", operator.value.trim()));

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.hidden = true; }, 3600);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  let payload = null;
  try { payload = await response.json(); } catch (_) { payload = {}; }
  if (!response.ok) {
    const error = new Error(payload.detail || `请求失败 (${response.status})`);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

function episodeUrl(relative) {
  return relative.split("/").map(encodeURIComponent).join("/");
}

async function browse(path = "", refresh = false) {
  const root = sourceRoot.value.trim();
  if (!root) return toast("源目录不能为空");
  const serial = ++state.requestSerial;
  $("#browserEmpty").hidden = false;
  $("#browserEmpty").textContent = "正在读取目录…";
  $("#folderList").hidden = true;
  $("#episodeViewport").hidden = true;
  try {
    const data = await api(`/api/browse?root=${encodeURIComponent(root)}&path=${encodeURIComponent(path)}&refresh=${refresh}`);
    if (serial !== state.requestSerial) return;
    if (state.root && state.root !== data.root) closeEditor();
    state.root = data.root;
    state.path = data.path;
    state.listing = data;
    renderBrowse();
  } catch (error) {
    $("#browserEmpty").textContent = error.message;
    toast(error.message);
  }
}

function renderBrowse() {
  const data = state.listing;
  $("#browserTitle").textContent = data.path ? data.path.split("/").at(-1) : data.root;
  const items = data.view === "folders" ? data.folders : data.episodes;
  $("#itemCount").textContent = items.length;
  $("#browserEmpty").hidden = items.length > 0;
  $("#browserEmpty").textContent = data.view === "folders" ? "没有可见子目录" : "没有 Episode";
  const crumbs = $("#breadcrumbs");
  crumbs.replaceChildren();
  data.breadcrumbs.forEach((crumb, index) => {
    if (index) {
      const sep = document.createElement("span");
      sep.className = "crumb-sep";
      sep.textContent = "/";
      crumbs.append(sep);
    }
    const button = document.createElement("button");
    button.className = "crumb";
    button.textContent = crumb.name;
    button.addEventListener("click", () => browse(crumb.path));
    crumbs.append(button);
  });
  if (data.view === "folders") renderFolders(data.folders);
  else renderEpisodeViewport();
}

function renderFolders(folders) {
  const list = $("#folderList");
  list.replaceChildren();
  folders.forEach((folder) => {
    const button = document.createElement("button");
    button.className = "folder-row";
    const icon = document.createElement("span");
    icon.className = "folder-icon";
    icon.textContent = "▰";
    const copy = document.createElement("span");
    copy.className = "row-copy";
    const title = document.createElement("strong");
    title.textContent = folder.name;
    const sub = document.createElement("small");
    sub.textContent = folder.episodeCount
      ? `${folder.episodeCount} 个 Episode`
      : folder.path;
    copy.append(title, sub);
    button.append(icon, copy);
    button.addEventListener("click", () => browse(folder.path));
    list.append(button);
  });
  list.hidden = false;
  $("#episodeViewport").hidden = true;
}

function renderEpisodeViewport() {
  const viewport = $("#episodeViewport");
  viewport.hidden = false;
  $("#folderList").hidden = true;
  $("#episodeCanvas").style.height = `${state.listing.episodes.length * rowHeight}px`;
  renderVirtualRows();
}

function renderVirtualRows() {
  if (!state.listing || state.listing.view !== "episodes") return;
  const viewport = $("#episodeViewport");
  const canvas = $("#episodeCanvas");
  const start = Math.max(0, Math.floor(viewport.scrollTop / rowHeight) - 2);
  const visible = Math.min(26, Math.ceil(viewport.clientHeight / rowHeight) + 4);
  const values = state.listing.episodes.slice(start, start + visible);
  canvas.replaceChildren();
  values.forEach((item, offset) => {
    const button = document.createElement("button");
    button.className = `episode-row${item.episode === state.episode ? " active" : ""}`;
    button.style.top = `${(start + offset) * rowHeight}px`;
    const copy = document.createElement("span");
    copy.className = "row-copy";
    const title = document.createElement("strong");
    title.textContent = item.name;
    const sub = document.createElement("small");
    sub.textContent = item.parent;
    copy.append(title, sub);
    button.append(copy);
    button.addEventListener("click", () => openEpisode(item.episode));
    canvas.append(button);
  });
}
$("#episodeViewport").addEventListener("scroll", renderVirtualRows, {passive: true});
window.addEventListener("resize", renderVirtualRows);

async function openEpisode(relative) {
  const serial = ++state.requestSerial;
  const video = $("#video");
  video.pause();
  video.removeAttribute("src");
  video.load();
  state.episode = relative;
  state.detail = null;
  state.stream = null;
  state.ranges = [];
  state.history = [];
  state.frame = 0;
  state.preview = false;
  state.nextKeptFrame = null;
  renderVirtualRows();

  $("#welcome").hidden = true;
  $("#editor").hidden = false;
  $("#episodeName").textContent = relative.split("/").at(-1);
  $("#taskName").textContent = relative;
  $("#validationBadge").className = "validation-badge loading";
  $("#validationBadge").textContent = "读取中";
  $("#validationBadge").disabled = true;
  $("#streamTabs").replaceChildren();
  $("#rangeList").replaceChildren();
  $("#videoUnavailable").hidden = true;
  $("#trim").disabled = true;
  $("#noTrim").disabled = true;
  $("#preview").classList.remove("active");
  $("#preview").textContent = "预览结果";

  try {
    const detail = await api(`/api/episodes/${episodeUrl(relative)}?root=${encodeURIComponent(state.root)}`);
    if (serial !== state.requestSerial) return;
    state.detail = detail;
    const saved = localStorage.getItem("light.camera");
    const playable = detail.streams.filter((item) => item.browserPlayable);
    state.stream = playable.find((item) => item.name === saved)
      || ["head_right", "rgbd_head_color", "hand_right"].map((name) => playable.find((item) => item.name === name)).find(Boolean)
      || playable[0] || null;
    renderEditor();
  } catch (error) {
    if (serial !== state.requestSerial) return;
    const badge = $("#validationBadge");
    badge.className = "validation-badge invalid";
    badge.textContent = "读取失败";
    badge.disabled = false;
    $("#issues").textContent = error.message;
    toast(error.message);
  }
}

function renderEditor() {
  const detail = state.detail;
  $("#welcome").hidden = true;
  $("#editor").hidden = false;
  $("#episodeName").textContent = detail.episode.split("/").at(-1);
  $("#taskName").textContent = detail.task || detail.episode;
  $("#overlayTotal").textContent = Math.max(0, detail.frameCount - 1);
  $("#scrubber").max = Math.max(0, detail.frameCount - 1);
  $("#frameNumber").max = Math.max(0, detail.frameCount - 1);
  const badge = $("#validationBadge");
  badge.className = `validation-badge ${detail.valid ? "valid" : "invalid"}`;
  badge.textContent = detail.valid ? `${detail.streams.length} 路同步有效` : "严格校验未通过";
  badge.disabled = detail.issues.length === 0;
  const issues = $("#issues");
  issues.textContent = detail.issues.join("\n");
  renderStreams();
  setStream(state.stream);
  renderRanges();
  updateFrame(0, true);
  setSubmitEnabled();
}

function renderStreams() {
  const tabs = $("#streamTabs");
  tabs.replaceChildren();
  state.detail.streams.forEach((stream) => {
    const button = document.createElement("button");
    button.className = `stream-tab${state.stream && state.stream.name === stream.name ? " active" : ""}`;
    button.textContent = stream.name;
    button.disabled = !stream.browserPlayable;
    button.title = stream.browserPlayable ? `${stream.width}×${stream.height} · ${stream.codec}` : (stream.error || `${stream.codec || "未知编码"} · 不可播放`);
    button.addEventListener("click", () => setStream(stream));
    tabs.append(button);
  });
}

function setStream(stream) {
  state.stream = stream;
  renderStreams();
  const video = $("#video");
  const unavailable = $("#videoUnavailable");
  if (!stream) {
    video.removeAttribute("src");
    video.load();
    unavailable.hidden = false;
    return;
  }
  const frame = state.frame;
  const resumePreview = state.preview && state.nextKeptFrame !== null;
  localStorage.setItem("light.camera", stream.name);
  video.src = stream.mediaUrl;
  video.load();
  video.addEventListener("loadedmetadata", () => {
    seekFrame(frame);
    if (resumePreview) video.play().catch((error) => toast(error.message));
  }, {once: true});
  unavailable.hidden = true;
}

function clampFrame(value) {
  return Math.max(0, Math.min(state.detail.frameCount - 1, Math.round(Number(value) || 0)));
}
function seekFrame(value) {
  if (!state.detail) return;
  const frame = clampFrame(value);
  const video = $("#video");
  if (state.stream) video.currentTime = Math.min(frame / state.detail.fps, Math.max(0, (video.duration || Infinity) - .0001));
  updateFrame(frame, true);
}
function updateFrame(value, controls = false) {
  if (!state.detail) return;
  state.frame = clampFrame(value);
  $("#overlayFrame").textContent = state.frame;
  $("#frameNumber").value = state.frame;
  $("#scrubber").value = state.frame;
  $("#timecode").textContent = formatTime(state.frame / state.detail.fps);
  renderTrack();
  if (controls) $("#playPause").textContent = $("#video").paused ? "▶" : "Ⅱ";
}
function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60).toString().padStart(2, "0");
  const rest = (seconds % 60).toFixed(3).padStart(6, "0");
  return `${minutes}:${rest}`;
}

function keptFrameAtOrAfter(value) {
  let frame = Math.max(0, Math.round(value));
  for (const [start, end] of state.ranges) {
    if (frame < start) return frame;
    if (frame < end) frame = end;
  }
  return frame < state.detail.frameCount ? frame : null;
}

function lastKeptFrame() {
  let frame = state.detail.frameCount - 1;
  for (let index = state.ranges.length - 1; index >= 0; index -= 1) {
    const [start, end] = state.ranges[index];
    if (frame >= start && frame < end) frame = start - 1;
  }
  return frame >= 0 ? frame : null;
}

function handleVideoProgress(mediaTime) {
  if (!state.detail) return;
  const video = $("#video");
  let frame = clampFrame(Math.floor(mediaTime * state.detail.fps + .001));
  if (state.preview && state.nextKeptFrame !== null && !video.paused) {
    const desired = keptFrameAtOrAfter(frame);
    if (desired === null) {
      const last = lastKeptFrame();
      state.nextKeptFrame = null;
      video.pause();
      if (last !== null) {
        video.currentTime = last / state.detail.fps;
        frame = last;
      }
    } else if (desired !== frame) {
      state.nextKeptFrame = desired;
      video.currentTime = desired / state.detail.fps;
      frame = desired;
    } else {
      state.nextKeptFrame = desired;
    }
  }
  updateFrame(frame);
}

const videoNode = $("#video");
if ("requestVideoFrameCallback" in videoNode) {
  const onVideoFrame = (_, metadata) => {
    handleVideoProgress(metadata.mediaTime);
    videoNode.requestVideoFrameCallback(onVideoFrame);
  };
  videoNode.requestVideoFrameCallback(onVideoFrame);
} else {
  videoNode.addEventListener("timeupdate", () => handleVideoProgress(videoNode.currentTime));
}
$("#video").addEventListener("play", () => { $("#playPause").textContent = "Ⅱ"; });
$("#video").addEventListener("pause", () => { $("#playPause").textContent = "▶"; });
$("#playPause").addEventListener("click", () => {
  if (!state.stream) return;
  if ($("#video").paused) $("#video").play().catch((error) => toast(error.message));
  else $("#video").pause();
});
$("#stepBack").addEventListener("click", () => { $("#video").pause(); seekFrame(state.frame - 1); });
$("#stepForward").addEventListener("click", () => { $("#video").pause(); seekFrame(state.frame + 1); });
$("#toStart").addEventListener("click", () => seekFrame(0));
$("#toEnd").addEventListener("click", () => seekFrame(state.detail.frameCount - 1));
$("#scrubber").addEventListener("input", (event) => seekFrame(event.target.value));
$("#frameNumber").addEventListener("change", (event) => seekFrame(event.target.value));

function normalizeRanges(values) {
  const max = state.detail.frameCount;
  const sorted = values.map(([a,b]) => [Math.max(0, Math.min(a,b)), Math.min(max, Math.max(a,b))]).filter(([a,b]) => b > a).sort((a,b) => a[0]-b[0]);
  const merged = [];
  sorted.forEach(([start,end]) => {
    const last = merged.at(-1);
    if (last && start <= last[1]) last[1] = Math.max(last[1], end);
    else merged.push([start,end]);
  });
  return merged;
}
function saveHistory() { state.history.push(state.ranges.map((item) => [...item])); }
function renderRanges() {
  const list = $("#rangeList");
  list.replaceChildren();
  state.ranges.forEach(([start,end], index) => {
    const button = document.createElement("button");
    button.className = "range-chip";
    button.textContent = `${start}–${end - 1} ×`;
    button.title = "移除此区间";
    button.addEventListener("click", () => { saveHistory(); state.ranges.splice(index,1); renderRanges(); });
    list.append(button);
  });
  const removed = state.ranges.reduce((sum,[start,end]) => sum + end-start, 0);
  const kept = state.detail.frameCount - removed;
  $("#removedCount").textContent = removed;
  $("#resultFrames").textContent = `保留 ${kept} 帧`;
  $("#resultDuration").textContent = `${(kept/state.detail.fps).toFixed(2)} 秒`;
  $("#undo").disabled = state.history.length === 0;
  $("#clear").disabled = state.ranges.length === 0;
  renderTrack();
  setSubmitEnabled();
}
function renderTrack() {
  if (!state.detail) return;
  const track = $("#rangeTrack");
  track.querySelectorAll(".range-block,.playhead").forEach((node) => node.remove());
  const count = state.detail.frameCount;
  state.ranges.forEach(([start,end]) => addBlock(start,end,"range-block",count));
  if (state.dragStart !== null && state.dragEnd !== null) addBlock(Math.min(state.dragStart,state.dragEnd), Math.max(state.dragStart,state.dragEnd)+1,"range-block draft",count);
  const playhead = document.createElement("span");
  playhead.className = "playhead";
  playhead.style.left = `${100 * state.frame / Math.max(1,count-1)}%`;
  track.append(playhead);
}
function addBlock(start,end,className,count) {
  const block = document.createElement("span");
  block.className = className;
  block.style.left = `${100*start/count}%`;
  block.style.width = `${100*(end-start)/count}%`;
  $("#rangeTrack").append(block);
}
function pointerFrame(event) {
  const rect = $("#rangeTrack").getBoundingClientRect();
  return clampFrame(Math.floor((event.clientX - rect.left) / rect.width * state.detail.frameCount));
}
$("#rangeTrack").addEventListener("pointerdown", (event) => {
  if (!state.detail) return;
  $("#rangeTrack").setPointerCapture(event.pointerId);
  state.dragStart = pointerFrame(event);
  state.dragEnd = state.dragStart;
  renderTrack();
});
$("#rangeTrack").addEventListener("pointermove", (event) => {
  if (state.dragStart === null) return;
  state.dragEnd = pointerFrame(event);
  renderTrack();
});
$("#rangeTrack").addEventListener("pointerup", (event) => {
  if (state.dragStart === null) return;
  state.dragEnd = pointerFrame(event);
  saveHistory();
  state.ranges = normalizeRanges([...state.ranges, [Math.min(state.dragStart,state.dragEnd), Math.max(state.dragStart,state.dragEnd)+1]]);
  state.dragStart = state.dragEnd = null;
  renderRanges();
});
$("#undo").addEventListener("click", () => { if (state.history.length) { state.ranges = state.history.pop(); renderRanges(); } });
$("#clear").addEventListener("click", () => { if (state.ranges.length) { saveHistory(); state.ranges = []; renderRanges(); } });
$("#preview").addEventListener("click", () => {
  const video = $("#video");
  if (state.preview) {
    video.pause();
    state.preview = false;
    state.nextKeptFrame = null;
    $("#preview").classList.remove("active");
    $("#preview").textContent = "预览结果";
    return;
  }
  if (!state.detail || !state.stream) {
    toast("当前 Episode 没有可预览的视频流");
    return;
  }
  const first = keptFrameAtOrAfter(0);
  if (first === null) {
    video.pause();
    toast("删除区间覆盖了全部帧，没有可预览结果");
    return;
  }
  state.preview = true;
  state.nextKeptFrame = first;
  $("#preview").classList.add("active");
  $("#preview").textContent = "退出预览";
  seekFrame(first);
  video.play().catch((error) => {
    state.preview = false;
    state.nextKeptFrame = null;
    $("#preview").classList.remove("active");
    $("#preview").textContent = "预览结果";
    toast(error.message);
  });
});

function setSubmitEnabled() {
  if (!state.detail) return;
  const removed = state.ranges.reduce((sum,[a,b]) => sum+b-a,0);
  const valid = state.detail.valid && state.detail.frameCount - removed >= 2;
  $("#trim").disabled = !valid || state.ranges.length === 0;
  $("#noTrim").disabled = !state.detail.valid;
}

async function submit(mode, overwrite = false) {
  const output = outputRoot.value.trim();
  const name = operator.value.trim();
  if (!output || !name) return toast("cleaned 根目录和操作人员均为必填");
  localStorage.setItem("light.outputRoot", output);
  localStorage.setItem("light.operator", name);
  state.pendingMode = mode;
  const body = {
    mode, sourceRoot: state.root, episode: state.episode, outputRoot: output,
    operator: name, sourceToken: state.detail.sourceToken,
    ranges: mode === "trim" ? state.ranges : [], overwrite,
  };
  try {
    const result = await api("/api/operations", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
    state.operationId = result.operationId;
    state.operations.set(result.operationId, result);
    renderQueue();
    followOperation(result.operationId, result.eventsUrl);
    toast(`任务已进入队列，第 ${result.queuePosition || 1} 位`);
  } catch (error) {
    if (error.status === 409 && error.message.includes("目标已存在")) {
      $("#overwritePath").textContent = error.message.replace(/^.*?:\s*/, "");
      $("#overwriteDialog").showModal();
    } else toast(error.message);
  }
}
$("#trim").addEventListener("click", () => submit("trim"));
$("#noTrim").addEventListener("click", () => submit("no_trim"));
$("#confirmOverwrite").addEventListener("click", () => setTimeout(() => submit(state.pendingMode, true), 0));

const terminalStatuses = new Set(["completed", "completed_csv_failed", "failed"]);

function statusLabel(status) {
  return {
    queued: "排队中",
    running: "运行中",
    completed: "已完成",
    completed_csv_failed: "CSV 待重试",
    failed: "失败",
  }[status] || status;
}

function renderQueue() {
  const values = [...state.operations.values()];
  const active = values.filter((item) => item.status === "queued" || item.status === "running");
  const count = $("#queueCount");
  count.textContent = active.length;
  count.hidden = active.length === 0;
  $("#queueSummary").textContent = active.length
    ? `${active.filter((item) => item.status === "running").length} 个运行中 · ${active.filter((item) => item.status === "queued").length} 个排队中`
    : "无活动任务";

  const rank = {running: 0, queued: 1, completed_csv_failed: 2, failed: 3, completed: 4};
  values.sort((first, second) => {
    const statusOrder = (rank[first.status] ?? 9) - (rank[second.status] ?? 9);
    if (statusOrder) return statusOrder;
    if (first.status === "queued") return (first.queuePosition || 0) - (second.queuePosition || 0);
    return String(second.submittedAt || "").localeCompare(String(first.submittedAt || ""));
  });

  const list = $("#queueList");
  list.replaceChildren();
  if (!values.length) {
    const empty = document.createElement("div");
    empty.className = "queue-empty";
    empty.textContent = "尚无任务";
    list.append(empty);
    return;
  }
  values.forEach((operation) => {
    const item = document.createElement("article");
    item.className = `queue-item ${operation.status}`;
    const head = document.createElement("div");
    head.className = "queue-item-head";
    const title = document.createElement("strong");
    title.textContent = operation.episode.split("/").at(-1);
    title.title = operation.episode;
    const badge = document.createElement("span");
    badge.className = "queue-status";
    badge.textContent = statusLabel(operation.status);
    head.append(title, badge);

    const meta = document.createElement("div");
    meta.className = "queue-meta";
    const mode = operation.mode === "trim" ? "剪切" : "无需剪切";
    const position = operation.status === "queued" ? ` · 第 ${operation.queuePosition} 位` : "";
    meta.textContent = `${mode}${position} · ${operation.message}`;

    const progress = document.createElement("div");
    progress.className = "queue-progress";
    const fill = document.createElement("span");
    fill.style.width = `${Math.max(0, Math.min(100, Number(operation.progress || 0) * 100))}%`;
    progress.append(fill);
    item.append(head, meta, progress);

    if (operation.error) {
      const error = document.createElement("p");
      error.className = "queue-error";
      error.textContent = operation.error;
      item.append(error);
    }
    if (operation.status === "completed_csv_failed") {
      const retry = document.createElement("button");
      retry.className = "secondary queue-retry";
      retry.textContent = "重试 CSV";
      retry.addEventListener("click", () => retryCsv(operation.id));
      item.append(retry);
    }
    list.append(item);
  });
}

function followOperation(id, url = `/api/operations/${id}/events`) {
  if (state.eventSources.has(id)) return;
  const source = new EventSource(url);
  state.eventSources.set(id, source);
  source.addEventListener("progress", (event) => {
    const data = JSON.parse(event.data);
    state.operations.set(id, data);
    $("#operationStatus").textContent = data.status;
    renderQueue();
    if (terminalStatuses.has(data.status)) {
      source.close();
      state.eventSources.delete(id);
      if (data.status === "completed") toast(`输出完成：${data.result.outputPath}`);
      else if (data.status === "completed_csv_failed") toast("输出已发布，但 CSV 记录失败");
      else toast(data.error || "操作失败");
    }
  });
  source.onerror = () => {
    source.close();
    state.eventSources.delete(id);
    toast("任务进度连接已断开，打开队列可重新读取状态");
  };
}

async function retryCsv(id) {
  try {
    const data = await api(`/api/operations/${id}/retry-csv`, {method:"POST"});
    state.operations.set(id, data);
    renderQueue();
    toast("CSV 历史记录已补写");
  } catch (error) {
    toast(error.message);
  }
}

async function loadOperations() {
  try {
    const data = await api("/api/operations");
    state.operations.clear();
    [...data.queued, ...data.running, ...data.completed].forEach((item) => state.operations.set(item.id, item));
    renderQueue();
    [...data.queued, ...data.running].forEach((item) => followOperation(item.id));
  } catch (error) {
    toast(error.message);
  }
}

async function loadSettings() {
  try {
    let data = await api("/api/operations/settings");
    const remembered = Number(localStorage.getItem("light.concurrency"));
    if (Number.isInteger(remembered) && remembered >= 1 && remembered <= 4 && remembered !== data.concurrency) {
      data = await api("/api/operations/settings", {
        method: "PATCH",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({concurrency: remembered}),
      });
    }
    $("#concurrency").value = String(data.concurrency);
  } catch (error) {
    toast(error.message);
  }
}

$("#concurrency").addEventListener("change", async (event) => {
  const concurrency = Number(event.target.value);
  try {
    const data = await api("/api/operations/settings", {
      method: "PATCH",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({concurrency}),
    });
    localStorage.setItem("light.concurrency", String(data.concurrency));
    event.target.value = String(data.concurrency);
  } catch (error) {
    toast(error.message);
  }
});
$("#queueToggle").addEventListener("click", () => {
  $("#queuePanel").hidden = false;
  loadOperations();
});
$("#queueClose").addEventListener("click", () => { $("#queuePanel").hidden = true; });
$("#validationBadge").addEventListener("click", () => {
  if (!$("#validationBadge").disabled) $("#issuesDialog").showModal();
});

$("#openRoot").addEventListener("click", () => browse(""));
$("#refresh").addEventListener("click", () => browse(state.path || "", true));
sourceRoot.addEventListener("keydown", (event) => { if (event.key === "Enter") browse(""); });

function closeEditor() {
  const video = $("#video");
  video.pause();
  video.removeAttribute("src");
  video.load();
  state.episode = "";
  state.detail = null;
  state.stream = null;
  state.ranges = [];
  state.preview = false;
  state.nextKeptFrame = null;
  $("#preview").textContent = "预览结果";
  $("#editor").hidden = true;
  $("#welcome").innerHTML = '<span class="welcome-mark">CUT</span><h1>选择一个 Episode</h1>';
  $("#welcome").hidden = false;
}

loadSettings();
loadOperations();
