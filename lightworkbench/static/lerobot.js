"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const DEFAULT_ROOT = "/inspire/qb-ilm/project/robot-decision/public/demo2/lerobot_data_08_29_26_cz_merged/";
const PAGE_SIZE = 100;
const appBaseUrl = new URL(".", window.location.href);

function appUrl(path) {
  const value = String(path);
  if (/^(?:[a-z][a-z\d+.-]*:)?\/\//i.test(value)) return value;
  return new URL(value.replace(/^\/+/, ""), appBaseUrl).toString();
}

const state = {
  root: "",
  summary: null,
  tasks: [],
  selectedTask: null,
  episodes: [],
  episodeTotal: 0,
  page: 1,
  pageSize: PAGE_SIZE,
  pageCount: 0,
  query: "",
  selectedEpisodeId: null,
  detail: null,
  videos: [],
  masterVideo: null,
  requestSerial: 0,
  detailSerial: 0,
  syncFrame: 0,
};

const rootInput = $("#rootInput");
const taskList = $("#taskList");
const episodeList = $("#episodeList");
const videoElements = $$("[data-camera-video]");

rootInput.value = localStorage.getItem("lerobot.root") || DEFAULT_ROOT;

function firstDefined(object, keys, fallback = undefined) {
  if (!object || typeof object !== "object") return fallback;
  for (const key of keys) {
    if (object[key] !== undefined && object[key] !== null) return object[key];
  }
  return fallback;
}

function finiteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function formatInteger(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.round(number).toLocaleString("zh-CN") : "-";
}

function formatDuration(seconds) {
  const value = Math.max(0, finiteNumber(seconds));
  const minutes = Math.floor(value / 60).toString().padStart(2, "0");
  const remainder = (value % 60).toFixed(3).padStart(6, "0");
  return `${minutes}:${remainder}`;
}

function toast(message) {
  const node = $("#toast");
  node.textContent = String(message || "未知错误");
  node.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.hidden = true; }, 4000);
}

async function api(path, signal) {
  const response = await fetch(appUrl(path), {signal});
  let payload = {};
  try { payload = await response.json(); } catch (_) { /* Response may be empty. */ }
  if (!response.ok) {
    const message = firstDefined(payload, ["detail", "message", "error"], `请求失败 (${response.status})`);
    throw new Error(typeof message === "string" ? message : JSON.stringify(message));
  }
  return payload;
}

function apiUrl(path, params = {}) {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") search.set(key, String(value));
  });
  return `${path}?${search}`;
}

function normalizeTask(task, position) {
  if (typeof task === "string") return {index: position, name: task, count: 0};
  const index = firstDefined(task, ["task_index", "taskIndex", "index", "id"], position);
  return {
    index,
    name: String(firstDefined(task, ["task", "name", "text", "description"], `任务 ${index}`)),
    count: finiteNumber(firstDefined(task, ["episode_count", "episodeCount", "count", "num_episodes"], 0)),
  };
}

function normalizeTasks(payload) {
  let values = firstDefined(payload, ["tasks", "task_list", "taskList"], []);
  if (!Array.isArray(values) && values && typeof values === "object") {
    values = Object.entries(values).map(([index, value]) => {
      if (typeof value === "string") return {task_index: index, task: value};
      return {...value, task_index: firstDefined(value, ["task_index", "index", "id"], index)};
    });
  }
  return Array.isArray(values) ? values.map(normalizeTask) : [];
}

function normalizeEpisode(item, position) {
  const value = item && typeof item === "object" ? item : {episode_index: item};
  const id = firstDefined(value, ["episode_index", "episodeIndex", "episode_id", "episodeId", "index", "id"], position);
  const frameCount = finiteNumber(firstDefined(value, ["frame_count", "frameCount", "length", "frames", "num_frames"], 0));
  const episodeTasks = normalizeEpisodeTasks(value);
  return {
    raw: value,
    id,
    label: String(firstDefined(value, ["name", "episode", "label"], `episode_${String(id).padStart(6, "0")}`)),
    task: String(firstDefined(value, ["task", "task_name", "taskName", "text"], episodeTasks.map((task) => task.name).join(" / "))),
    taskIndex: firstDefined(value, ["task_index", "taskIndex"], episodeTasks[0]?.index ?? null),
    frameCount,
    duration: finiteNumber(firstDefined(value, ["duration_s", "duration", "duration_seconds", "durationSeconds"], 0)),
    fps: finiteNumber(firstDefined(value, ["fps", "video_fps"], 0)),
    source: String(firstDefined(value, ["source_episode", "sourceEpisode", "source", "source_path"], "")),
  };
}

function normalizeEpisodeTasks(value) {
  const tasks = firstDefined(value, ["tasks", "task_list", "taskList"], []);
  if (!Array.isArray(tasks)) return [];
  return tasks.map((task, index) => normalizeTask(task, index));
}

function setLoadingState(kind) {
  const dot = $("#statusDot");
  dot.className = `status-dot ${kind}`;
  $("#openButton").disabled = kind === "loading";
  $("#refreshButton").disabled = kind === "loading";
}

async function openRoot() {
  const root = rootInput.value.trim();
  if (!root) return toast("数据根目录不能为空");
  const serial = ++state.requestSerial;
  setLoadingState("loading");
  $("#datasetName").textContent = "正在读取数据集...";
  $("#taskEmpty").hidden = false;
  $("#taskEmpty").textContent = "正在加载任务...";
  taskList.hidden = true;
  closeEpisode();
  try {
    const payload = await api(apiUrl("/api/lerobot/summary", {root}));
    if (serial !== state.requestSerial) return;
    state.root = String(firstDefined(payload, ["root", "dataset_root", "path"], root));
    state.summary = payload.summary && typeof payload.summary === "object" ? payload.summary : payload;
    state.tasks = normalizeTasks(payload).length ? normalizeTasks(payload) : normalizeTasks(state.summary);
    state.selectedTask = null;
    state.page = 1;
    state.query = "";
    $("#episodeSearch").value = "";
    localStorage.setItem("lerobot.root", state.root);
    rootInput.value = state.root;
    renderSummary();
    renderTasks();
    setLoadingState("ready");
    await loadEpisodes();
  } catch (error) {
    if (serial !== state.requestSerial) return;
    setLoadingState("error");
    $("#datasetName").textContent = "打开失败";
    $("#taskEmpty").textContent = error.message;
    toast(error.message);
  }
}

function renderSummary() {
  const summary = state.summary || {};
  const trimmed = state.root.replace(/\/+$/, "");
  $("#datasetName").textContent = String(firstDefined(summary, ["dataset", "name", "dataset_name", "datasetName"], trimmed.split("/").pop() || state.root));
  $("#datasetVersion").textContent = String(firstDefined(summary, ["codebase_version", "codebaseVersion", "version"], ""));
  $("#totalTasks").textContent = formatInteger(firstDefined(summary, ["total_tasks", "totalTasks"], state.tasks.length));
  $("#totalEpisodes").textContent = formatInteger(firstDefined(summary, ["total_episodes", "totalEpisodes", "episode_count"], 0));
  $("#totalFrames").textContent = formatInteger(firstDefined(summary, ["total_frames", "totalFrames", "frame_count"], 0));
  $("#datasetFps").textContent = finiteNumber(firstDefined(summary, ["fps", "video_fps"], 0)) || "-";
  $("#taskCount").textContent = formatInteger(state.tasks.length);
}

function renderTasks() {
  const query = $("#taskSearch").value.trim().toLocaleLowerCase();
  const filtered = state.tasks.filter((task) => task.name.toLocaleLowerCase().includes(query));
  taskList.replaceChildren();
  const totalEpisodes = finiteNumber(firstDefined(state.summary, ["total_episodes", "totalEpisodes", "episode_count"], 0));
  const allRow = taskButton({index: null, name: "全部任务", count: totalEpisodes}, state.selectedTask === null, true);
  if (!query || "全部任务".includes(query)) taskList.append(allRow);
  filtered.forEach((task) => taskList.append(taskButton(task, String(task.index) === String(state.selectedTask), false)));
  taskList.hidden = taskList.childElementCount === 0;
  $("#taskEmpty").hidden = !taskList.hidden;
  $("#taskEmpty").textContent = state.tasks.length ? "没有匹配的任务" : "数据集没有任务";
}

function taskButton(task, active, isAll) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `task-row${active ? " active" : ""}`;
  button.setAttribute("role", "option");
  button.setAttribute("aria-selected", String(active));
  const index = document.createElement("span");
  index.className = "task-index";
  index.textContent = isAll ? "ALL" : String(task.index);
  const copy = document.createElement("span");
  copy.className = "row-copy";
  const title = document.createElement("strong");
  title.textContent = task.name;
  title.title = task.name;
  const sub = document.createElement("small");
  sub.textContent = isAll ? "跨任务浏览" : `任务索引 ${task.index}`;
  copy.append(title, sub);
  const count = document.createElement("span");
  count.className = "row-count";
  count.textContent = task.count ? formatInteger(task.count) : "";
  button.append(index, copy, count);
  button.addEventListener("click", () => selectTask(task));
  return button;
}

function selectTask(task) {
  const next = task.index === null ? null : task.index;
  if (String(next) === String(state.selectedTask) || (next === null && state.selectedTask === null)) return;
  state.selectedTask = next;
  state.page = 1;
  closeEpisode();
  renderTasks();
  loadEpisodes();
}

async function loadEpisodes() {
  if (!state.root) return;
  const serial = ++state.requestSerial;
  const selectedTask = state.tasks.find((task) => String(task.index) === String(state.selectedTask));
  $("#episodePaneTitle").textContent = selectedTask ? selectedTask.name : "全部数据";
  episodeList.hidden = true;
  $("#episodeEmpty").hidden = false;
  $("#episodeEmpty").textContent = "正在加载 Episode...";
  $("#previousPage").disabled = true;
  $("#nextPage").disabled = true;
  try {
    const payload = await api(apiUrl("/api/lerobot/episodes", {
      root: state.root,
      task_index: state.selectedTask,
      page: state.page,
      page_size: state.pageSize,
      q: state.query,
    }));
    if (serial !== state.requestSerial) return;
    const items = firstDefined(payload, ["episodes", "items", "data", "results"], []);
    state.episodes = Array.isArray(items) ? items.map(normalizeEpisode) : [];
    state.episodeTotal = finiteNumber(firstDefined(payload, ["total", "total_count", "count", "totalEpisodes"], state.episodes.length));
    state.page = Math.max(1, finiteNumber(firstDefined(payload, ["page", "current_page"], state.page), state.page));
    state.pageSize = Math.max(1, finiteNumber(firstDefined(payload, ["page_size", "pageSize", "limit"], state.pageSize), state.pageSize));
    state.pageCount = Math.max(0, finiteNumber(firstDefined(payload, ["pages", "pageCount", "page_count", "totalPages", "total_pages"], Math.ceil(state.episodeTotal / state.pageSize))));
    renderEpisodes();
  } catch (error) {
    if (serial !== state.requestSerial) return;
    state.episodes = [];
    state.episodeTotal = 0;
    state.pageCount = 0;
    renderEpisodes(error.message);
    toast(error.message);
  }
}

function renderEpisodes(error = "") {
  episodeList.replaceChildren();
  state.episodes.forEach((episode) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `episode-row${String(episode.id) === String(state.selectedEpisodeId) ? " active" : ""}`;
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", String(String(episode.id) === String(state.selectedEpisodeId)));
    const index = document.createElement("span");
    index.className = "task-index";
    index.textContent = String(episode.id);
    const copy = document.createElement("span");
    copy.className = "row-copy";
    const title = document.createElement("strong");
    title.textContent = episode.label;
    const sub = document.createElement("small");
    const facts = [];
    if (episode.frameCount) facts.push(`${formatInteger(episode.frameCount)} 帧`);
    if (episode.duration) facts.push(formatDuration(episode.duration));
    if (episode.task && state.selectedTask === null) facts.push(episode.task);
    sub.textContent = facts.join(" · ") || `Episode ${episode.id}`;
    sub.title = sub.textContent;
    copy.append(title, sub);
    button.append(index, copy);
    button.addEventListener("click", () => openEpisode(episode.id));
    episodeList.append(button);
  });
  episodeList.hidden = state.episodes.length === 0;
  $("#episodeEmpty").hidden = !episodeList.hidden;
  $("#episodeEmpty").textContent = error || (state.query ? "没有匹配的 Episode" : "此任务没有 Episode");
  $("#episodeTotal").textContent = formatInteger(state.episodeTotal);
  $("#pageLabel").textContent = state.pageCount ? `${state.page} / ${state.pageCount}` : "0 / 0";
  $("#previousPage").disabled = state.page <= 1;
  $("#nextPage").disabled = !state.pageCount || state.page >= state.pageCount;
}

function detailBody(payload) {
  const candidate = firstDefined(payload, ["detail", "episode"], payload);
  return candidate && typeof candidate === "object" ? candidate : payload;
}

function normalizeVideos(detail, payload) {
  let values = firstDefined(detail, ["videos", "streams", "media"], firstDefined(payload, ["videos", "streams", "media"], []));
  if (Array.isArray(values)) {
    return values.map((value, index) => normalizeVideo(value, firstDefined(value, ["name", "key", "camera"], String(index))));
  }
  if (values && typeof values === "object") {
    return Object.entries(values).map(([name, value]) => normalizeVideo(value, name));
  }
  return [];
}

function normalizeVideo(value, fallbackName) {
  const object = value && typeof value === "object" ? value : {url: value};
  const fromTimestamp = finiteNumber(firstDefined(object, ["from_timestamp", "fromTimestamp", "start", "startTime"], 0));
  const toTimestamp = finiteNumber(firstDefined(object, ["to_timestamp", "toTimestamp", "end", "endTime"], 0));
  return {
    name: String(firstDefined(object, ["name", "key", "camera", "video_key"], fallbackName)),
    url: String(firstDefined(object, ["media_url", "mediaUrl", "url", "src"], "")),
    playable: firstDefined(object, ["exists"], true) !== false && firstDefined(object, ["browser_playable", "browserPlayable", "playable"], true) !== false,
    error: String(firstDefined(object, ["error", "message"], "")),
    fromTimestamp,
    toTimestamp,
    duration: finiteNumber(firstDefined(object, ["duration_s", "duration", "duration_seconds", "durationSeconds"], Math.max(0, toTimestamp - fromTimestamp))),
  };
}

async function openEpisode(id) {
  if (!state.root) return;
  const serial = ++state.detailSerial;
  pauseAll();
  state.selectedEpisodeId = id;
  state.detail = null;
  renderEpisodes();
  $("#viewerWelcome").hidden = true;
  $("#viewer").hidden = false;
  $("#episodeTitle").textContent = `Episode ${id}`;
  $("#episodeTask").textContent = "正在加载...";
  resetVideos("正在读取视频...");
  setTransportEnabled(false);
  try {
    const payload = await api(apiUrl(`/api/lerobot/episodes/${encodeURIComponent(String(id))}`, {root: state.root}));
    if (serial !== state.detailSerial) return;
    const detail = detailBody(payload);
    state.detail = detail;
    state.videos = normalizeVideos(detail, payload);
    renderEpisodeDetail(id, detail);
    mountVideos();
  } catch (error) {
    if (serial !== state.detailSerial) return;
    $("#episodeTask").textContent = "读取失败";
    resetVideos(error.message);
    toast(error.message);
  }
}

function renderEpisodeDetail(requestedId, detail) {
  const listItem = state.episodes.find((item) => String(item.id) === String(requestedId));
  const episodeTasks = normalizeEpisodeTasks(detail);
  const id = firstDefined(detail, ["episode_index", "episodeIndex", "episode_id", "id", "index"], requestedId);
  const title = firstDefined(detail, ["name", "episode", "label"], listItem?.label || `episode_${String(id).padStart(6, "0")}`);
  const task = firstDefined(detail, ["task", "task_name", "taskName", "text"], episodeTasks.map((item) => item.name).join(" / ") || listItem?.task || "未标注任务");
  const fps = finiteNumber(firstDefined(detail, ["fps", "video_fps"], firstDefined(state.summary, ["fps", "video_fps"], 0)));
  const frames = finiteNumber(firstDefined(detail, ["frame_count", "frameCount", "length", "frames", "num_frames"], listItem?.frameCount || 0));
  const duration = finiteNumber(firstDefined(detail, ["duration_s", "duration", "duration_seconds", "durationSeconds"], listItem?.duration || (fps ? frames / fps : 0)));
  const dataPath = firstDefined(detail.data, ["path"], "");
  const sourceFallback = listItem?.source || `Episode ${id}${dataPath ? ` · ${dataPath}` : ""}`;
  const source = firstDefined(detail, ["source_episode", "sourceEpisode", "source", "source_path"], sourceFallback);
  state.detailView = {id, title: String(title), task: String(task), fps, frames, duration, source: String(source)};
  $("#episodeTitle").textContent = String(title);
  $("#episodeTask").textContent = String(task);
  $("#metaEpisode").textContent = String(id);
  $("#metaFrames").textContent = formatInteger(frames);
  $("#metaDuration").textContent = formatDuration(duration);
  $("#metaFps").textContent = fps ? `${fps} FPS` : "-";
  $("#metaSource").textContent = String(source);
  $("#metaSource").title = String(source);
  $("#timeline").max = duration || 1;
  $("#timeline").value = 0;
  $("#currentTime").textContent = formatDuration(0);
  $("#totalTime").textContent = formatDuration(duration);
  $("#currentFrame").textContent = "0";
  $("#lastFrame").textContent = String(Math.max(0, frames - 1));
  const index = state.episodes.findIndex((item) => String(item.id) === String(requestedId));
  $("#previousEpisode").disabled = index <= 0 && state.page <= 1;
  $("#nextEpisode").disabled = index < 0 || (index >= state.episodes.length - 1 && state.page >= state.pageCount);
}

function videoSlot(name) {
  const normalized = String(name).toLocaleLowerCase();
  if (normalized.includes("hand_left") || normalized.includes("left_hand") || /(^|[._-])left($|[._-])/.test(normalized)) return "left";
  if (normalized.includes("hand_right") || normalized.includes("right_hand") || /(^|[._-])right($|[._-])/.test(normalized)) return "right";
  if (normalized.includes("head") || normalized.includes("front") || normalized.includes("rgbd")) return "head";
  return null;
}

function mountVideos() {
  resetVideos("此机位无视频");
  const assigned = new Set();
  state.videos.forEach((stream) => {
    let slot = videoSlot(stream.name);
    if (!slot || assigned.has(slot)) slot = ["head", "left", "right"].find((candidate) => !assigned.has(candidate));
    if (!slot) return;
    assigned.add(slot);
    const panel = $(`[data-slot="${slot}"]`);
    const video = $("video", panel);
    const message = $("[data-camera-message]", panel);
    $("[data-camera-name]", panel).textContent = stream.name;
    $("[data-camera-name]", panel).title = stream.name;
    if (!stream.url || !stream.playable) {
      message.textContent = stream.error || "浏览器不可播放";
      message.hidden = false;
      return;
    }
    video.src = appUrl(stream.url);
    video.dataset.clipStart = String(stream.fromTimestamp);
    video.dataset.clipEnd = String(stream.toTimestamp);
    video.dataset.clipDuration = String(stream.duration);
    video.playbackRate = finiteNumber($("#playbackRate").value, 1);
    video.load();
    video.addEventListener("loadedmetadata", () => {
      video.currentTime = Math.min(stream.fromTimestamp, Math.max(0, video.duration - .001));
      message.hidden = true;
      selectMasterVideo();
      updateDurationFromMedia();
      setTransportEnabled(true);
    }, {once: true});
    video.addEventListener("error", () => {
      message.textContent = "视频加载失败";
      message.hidden = false;
      selectMasterVideo();
    }, {once: true});
  });
  selectMasterVideo();
}

function resetVideos(message) {
  cancelAnimationFrame(state.syncFrame);
  state.syncFrame = 0;
  state.masterVideo = null;
  $$(".video-panel").forEach((panel) => {
    const video = $("video", panel);
    video.pause();
    video.removeAttribute("src");
    delete video.dataset.clipStart;
    delete video.dataset.clipEnd;
    delete video.dataset.clipDuration;
    video.load();
    $("[data-camera-name]", panel).textContent = "未提供";
    const messageNode = $("[data-camera-message]", panel);
    messageNode.textContent = message;
    messageNode.hidden = false;
  });
}

function playableVideos() {
  return videoElements.filter((video) => video.src && video.readyState >= 1 && !video.error);
}

function selectMasterVideo() {
  const playable = playableVideos();
  state.masterVideo = playable.find((video) => video.dataset.cameraVideo === "head") || playable[0] || null;
}

function effectiveDuration() {
  const declared = finiteNumber(state.detailView?.duration, 0);
  if (declared) return declared;
  const durations = playableVideos().map((video) => finiteNumber(video.dataset.clipDuration, 0)).filter((duration) => duration > 0);
  return durations.length ? Math.min(...durations) : 0;
}

function clipStart(video) {
  return finiteNumber(video?.dataset.clipStart, 0);
}

function mediaTime(video, episodeTime) {
  const start = clipStart(video);
  const clipEnd = finiteNumber(video.dataset.clipEnd, 0);
  const fileEnd = Number.isFinite(video.duration) ? video.duration : Infinity;
  const end = clipEnd > start ? Math.min(clipEnd, fileEnd) : fileEnd;
  return Math.max(start, Math.min(start + episodeTime, Math.max(start, end - .0001)));
}

function updateDurationFromMedia() {
  const duration = effectiveDuration();
  if (!duration) return;
  $("#timeline").max = duration;
  $("#totalTime").textContent = formatDuration(duration);
  if (!state.detailView.duration) $("#metaDuration").textContent = formatDuration(duration);
}

function setTransportEnabled(enabled) {
  ["#stepBack", "#playPause", "#stepForward", "#timeline", "#playbackRate"].forEach((selector) => {
    $(selector).disabled = !enabled;
  });
}

async function playAll() {
  const videos = playableVideos();
  if (!videos.length) return toast("当前 Episode 没有可播放视频");
  selectMasterVideo();
  const time = Math.max(0, finiteNumber(state.masterVideo?.currentTime, 0) - clipStart(state.masterVideo));
  videos.forEach((video) => {
    const target = mediaTime(video, time);
    if (Math.abs(video.currentTime - target) > .03) video.currentTime = target;
  });
  const results = await Promise.allSettled(videos.map((video) => video.play()));
  if (results.every((result) => result.status === "rejected")) return toast("浏览器阻止了视频播放");
  $("#playPause").textContent = "Ⅱ";
  $("#playPause").title = "暂停";
  $("#playPause").setAttribute("aria-label", "暂停");
  syncLoop();
}

function pauseAll() {
  playableVideos().forEach((video) => video.pause());
  cancelAnimationFrame(state.syncFrame);
  state.syncFrame = 0;
  $("#playPause").textContent = "▶";
  $("#playPause").title = "播放";
  $("#playPause").setAttribute("aria-label", "播放");
  updateTransport();
}

function syncLoop() {
  cancelAnimationFrame(state.syncFrame);
  if (!state.masterVideo || state.masterVideo.paused || state.masterVideo.ended) {
    pauseAll();
    return;
  }
  const time = Math.max(0, state.masterVideo.currentTime - clipStart(state.masterVideo));
  if (effectiveDuration() && time >= effectiveDuration() - .001) {
    seekAll(effectiveDuration());
    pauseAll();
    return;
  }
  playableVideos().forEach((video) => {
    const target = mediaTime(video, time);
    if (video !== state.masterVideo && Math.abs(video.currentTime - target) > .08) video.currentTime = target;
  });
  updateTransport(time);
  state.syncFrame = requestAnimationFrame(syncLoop);
}

function seekAll(seconds) {
  const duration = effectiveDuration();
  const time = Math.max(0, Math.min(finiteNumber(seconds), Math.max(0, duration - .0001)));
  playableVideos().forEach((video) => { video.currentTime = mediaTime(video, time); });
  updateTransport(time);
}

function updateTransport(forcedTime) {
  const time = forcedTime ?? Math.max(0, finiteNumber(state.masterVideo?.currentTime, 0) - clipStart(state.masterVideo));
  const fps = finiteNumber(state.detailView?.fps, finiteNumber(firstDefined(state.summary, ["fps", "video_fps"], 0), 0));
  const maxFrame = Math.max(0, finiteNumber(state.detailView?.frames, 0) - 1);
  const frame = Math.min(maxFrame || Infinity, Math.max(0, Math.floor(time * (fps || 1) + .001)));
  $("#timeline").value = time;
  $("#currentTime").textContent = formatDuration(time);
  $("#currentFrame").textContent = String(Number.isFinite(frame) ? frame : 0);
}

function stepFrame(direction) {
  pauseAll();
  const fps = finiteNumber(state.detailView?.fps, finiteNumber(firstDefined(state.summary, ["fps", "video_fps"], 30), 30)) || 30;
  const current = state.masterVideo
    ? Math.max(0, finiteNumber(state.masterVideo.currentTime, 0) - clipStart(state.masterVideo))
    : finiteNumber($("#timeline").value, 0);
  seekAll(current + direction / fps);
}

function closeEpisode() {
  state.detailSerial += 1;
  state.selectedEpisodeId = null;
  state.detail = null;
  state.detailView = null;
  state.videos = [];
  resetVideos("等待 Episode");
  $("#viewer").hidden = true;
  $("#viewerWelcome").hidden = false;
}

async function adjacentEpisode(direction) {
  const index = state.episodes.findIndex((item) => String(item.id) === String(state.selectedEpisodeId));
  const adjacent = state.episodes[index + direction];
  if (adjacent) return openEpisode(adjacent.id);
  const targetPage = state.page + direction;
  if (targetPage < 1 || targetPage > state.pageCount) return;
  await changePage(targetPage);
  const target = direction > 0 ? state.episodes[0] : state.episodes.at(-1);
  if (target) openEpisode(target.id);
}

async function changePage(page) {
  state.page = page;
  closeEpisode();
  await loadEpisodes();
  episodeList.scrollTop = 0;
}

$("#openButton").addEventListener("click", openRoot);
$("#refreshButton").addEventListener("click", openRoot);
rootInput.addEventListener("keydown", (event) => { if (event.key === "Enter") openRoot(); });
$("#taskSearch").addEventListener("input", renderTasks);
$("#episodeSearch").addEventListener("input", () => {
  clearTimeout(state.searchTimer);
  state.searchTimer = setTimeout(() => {
    state.query = $("#episodeSearch").value.trim();
    state.page = 1;
    closeEpisode();
    loadEpisodes();
  }, 250);
});
$("#previousPage").addEventListener("click", () => changePage(state.page - 1));
$("#nextPage").addEventListener("click", () => changePage(state.page + 1));
$("#previousEpisode").addEventListener("click", () => adjacentEpisode(-1));
$("#nextEpisode").addEventListener("click", () => adjacentEpisode(1));
$("#playPause").addEventListener("click", () => state.masterVideo && !state.masterVideo.paused ? pauseAll() : playAll());
$("#stepBack").addEventListener("click", () => stepFrame(-1));
$("#stepForward").addEventListener("click", () => stepFrame(1));
$("#timeline").addEventListener("input", () => seekAll($("#timeline").value));
$("#playbackRate").addEventListener("change", () => {
  const rate = finiteNumber($("#playbackRate").value, 1);
  videoElements.forEach((video) => { video.playbackRate = rate; });
});
document.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
  if ($("#viewer").hidden) return;
  if (event.code === "Space") { event.preventDefault(); state.masterVideo && !state.masterVideo.paused ? pauseAll() : playAll(); }
  if (event.key === "ArrowLeft") { event.preventDefault(); stepFrame(-1); }
  if (event.key === "ArrowRight") { event.preventDefault(); stepFrame(1); }
});
videoElements.forEach((video) => {
  video.addEventListener("ended", pauseAll);
  video.addEventListener("click", () => state.masterVideo && !state.masterVideo.paused ? pauseAll() : playAll());
});

setTransportEnabled(false);
openRoot();
