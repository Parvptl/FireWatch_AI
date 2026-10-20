"use strict";
// FireWatch dashboard: every value shown comes from the backend API.

const $ = (id) => document.getElementById(id);
function loadToken() {
  try { return localStorage.getItem("firewatch_token") || ""; } catch (e) { return ""; }
}
let token = loadToken();
let lastSelectedStatus = null;
let ticking = false;
let selectedJob = null;
let authRequired = false;
let maxUploadMb = null;

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
}
function fmtTime(s) {
  if (s === null || s === undefined) return "–";
  const m = Math.floor(s / 60), sec = (s - m * 60).toFixed(1).padStart(4, "0");
  return `${m}:${sec}`;
}
function fmtDate(iso) { return iso ? new Date(iso).toLocaleString() : "–"; }
function headers(extra = {}) { return token ? {...extra, Authorization: `Bearer ${token}`} : extra; }
function fileUrl(jobId, name) {
  const q = token ? `?token=${encodeURIComponent(token)}` : "";
  return `/api/jobs/${jobId}/files/${encodeURIComponent(name)}${q}`;
}

async function api(path, opts = {}) {
  const res = await fetch(path, {...opts, headers: headers(opts.headers || {})});
  let body = null;
  try { body = await res.json(); } catch (e) { body = null; }
  if (!res.ok) {
    const msg = body && body.error ? `${body.error.code}: ${body.error.message}` : `HTTP ${res.status}`;
    if (res.status === 401) $("auth-card").hidden = false;
    throw new Error(msg);
  }
  return body;
}

function showBanner(msg) {
  const b = $("banner");
  if (!msg) { b.classList.add("hidden"); return; }
  b.textContent = msg; b.classList.remove("hidden");
}

async function refreshHealth() {
  try {
    const h = await api("/api/health");
    authRequired = h.auth_required;
    $("auth-card").hidden = !(authRequired && !token);
    const m = h.model;
    const colour = h.status === "ok" ? "#22c55e" : "#f59e0b";
    $("health").innerHTML = `<span class="dot" style="background:${colour}"></span>` +
      `service ${esc(h.status)} · model ${esc(m.state)} (${esc(m.device)}, ${esc(m.imgsz)} px)` +
      ` · queue ${h.queue.waiting}/${h.queue.max_queued} · running ${h.queue.running.length}`;
    $("stream-card").hidden = !h.stream_enabled;
    if (m.state === "missing" || m.state === "error") showBanner(`Model unavailable: ${m.error || m.weights}`);
    else showBanner("");
  } catch (e) {
    $("health").textContent = `service unreachable: ${e.message}`;
  }
}

async function refreshVideos() {
  try {
    if (maxUploadMb === null) {
      const c = await api("/api/config");
      maxUploadMb = c.video.max_upload_mb;
    }
    const r = await api("/api/videos");
    $("input-dir").textContent = `Input directory: ${r.input_dir} (only videos here can be analysed)`;
    const sel = $("video-select");
    sel.innerHTML = r.videos.length ? "" : `<option value="">no videos in input directory</option>`;
    for (const v of r.videos) {
      const o = document.createElement("option");
      o.value = v.path; o.textContent = `${v.path} (${(v.size_bytes / 1e6).toFixed(1)} MB)`;
      sel.appendChild(o);
    }
  } catch (e) { showBanner(`Cannot list videos: ${e.message}`); }
}

async function submitVideo() {
  const v = $("video-select").value;
  if (!v) return;
  try {
    const job = await api("/api/jobs", {method: "POST", headers: {"Content-Type": "application/json"},
                                        body: JSON.stringify({video: v})});
    selectedJob = job.id; showBanner(""); await refreshJobs(); await showDetail(job.id);
  } catch (e) { showBanner(`Submission failed: ${e.message}`); }
}

async function uploadVideo() {
  const f = $("upload-file").files[0];
  if (!f) { $("upload-status").textContent = "choose a file first"; return; }
  if (maxUploadMb && f.size > maxUploadMb * 1024 * 1024) {
    showBanner(`Upload refused: ${(f.size / 1e6).toFixed(1)} MB exceeds the ${maxUploadMb} MB limit (video.max_upload_mb).`);
    return;
  }
  $("upload-btn").disabled = true;
  $("upload-status").textContent = `uploading ${(f.size / 1e6).toFixed(1)} MB…`;
  try {
    const r = await api(`/api/uploads?submit=1&filename=${encodeURIComponent(f.name)}`,
                        {method: "POST", headers: {"Content-Type": "application/octet-stream"}, body: f});
    $("upload-status").textContent = `uploaded as ${r.video}`;
    selectedJob = r.job.id; await refreshVideos(); await refreshJobs(); await showDetail(r.job.id);
  } catch (e) {
    $("upload-status").textContent = ""; showBanner(`Upload failed: ${e.message}`);
  } finally { $("upload-btn").disabled = false; }
}

async function startStream() {
  const url = $("stream-url").value.trim();
  if (!url) return;
  try {
    const job = await api("/api/streams", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({url, name: $("stream-name").value.trim() || null,
                            max_duration_s: Number($("stream-max").value || 0)})});
    selectedJob = job.id; await refreshJobs(); await showDetail(job.id);
  } catch (e) { showBanner(`Stream failed: ${e.message}`); }
}

async function cancelJob(id) {
  try { await api(`/api/jobs/${id}/cancel`, {method: "POST"}); await refreshJobs(); if (selectedJob === id) await showDetail(id); }
  catch (e) { showBanner(`Cancel failed: ${e.message}`); }
}

function progressCell(j) {
  if (j.status === "completed") return "100 %";
  if (j.progress === null || j.progress === undefined) return `${j.frames_done} frames`;
  const p = Math.round(j.progress * 100);
  return `<span class="bar"><span style="width:${p}%"></span></span> ${p} %`;
}

async function refreshJobs() {
  try {
    const r = await api("/api/jobs?limit=50");
    const tb = document.querySelector("#jobs tbody");
    tb.innerHTML = "";
    for (const j of r.jobs) {
      const tr = document.createElement("tr");
      tr.className = "clickable";
      const active = j.status === "queued" || j.status === "running";
      tr.innerHTML = `<td>${esc(fmtDate(j.created_at))}</td><td>${esc(j.kind)}: ${esc(j.source_name)}</td>` +
        `<td><span class="status ${esc(j.status)}">${esc(j.status)}</span></td><td>${progressCell(j)}</td>` +
        `<td>${j.summary.events ?? "–"}</td><td>${j.summary.alerts ?? "–"}</td>` +
        `<td>${active ? `<button class="secondary" data-cancel="${j.id}">cancel</button>` : ""}</td>`;
      tr.addEventListener("click", (ev) => {
        if (ev.target.dataset.cancel) { ev.stopPropagation(); cancelJob(ev.target.dataset.cancel); return; }
        selectedJob = j.id; showDetail(j.id);
      });
      tb.appendChild(tr);
    }
    return r.jobs;
  } catch (e) { showBanner(`Cannot load jobs: ${e.message}`); return []; }
}

function dl(el, pairs) {
  el.innerHTML = pairs.filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
}

async function showDetail(id) {
  try {
    const [job, ev, dets] = await Promise.all([
      api(`/api/jobs/${id}`), api(`/api/jobs/${id}/events`), api(`/api/jobs/${id}/detections?limit=25`)]);
    $("detail").classList.remove("hidden");
    $("d-id").textContent = `${job.id} · ${job.status}`;
    $("d-error").textContent = job.error || ""; $("d-error").classList.toggle("hidden", !job.error);
    $("d-warning").textContent = job.warning || ""; $("d-warning").classList.toggle("hidden", !job.warning);
    const m = job.meta || {}, s = job.summary || {}, mod = job.model || {};
    dl($("d-meta"), [["Input", job.source_name], ["Kind", job.kind], ["Created", fmtDate(job.created_at)],
      ["Started", fmtDate(job.started_at)], ["Finished", job.finished_at ? fmtDate(job.finished_at) : ""],
      ["Resolution", m.width ? `${m.width}×${m.height}` : ""], ["FPS", m.fps], ["Duration", m.duration_s != null ? fmtTime(m.duration_s) : ""],
      ["Analysed until", fmtTime(job.video_time_s)], ["Frames analysed", job.frames_done],
      ["Checkpoint", mod.weights ? `${mod.weights.split("/").slice(-3).join("/")} (${(mod.sha256 || "").slice(0, 12)})` : ""],
      ["Detection threshold", mod.conf_threshold], ["Image size", mod.imgsz]]);
    const ctb = document.querySelector("#d-classes tbody"); ctb.innerHTML = "";
    for (const [cls, v] of Object.entries(s.per_class || {})) {
      ctb.innerHTML += `<tr><td class="cls-${esc(cls)}">${esc(cls)}</td><td>${v.frames_with_detections}</td>` +
        `<td>${v.boxes}</td><td>${v.max_confidence ? v.max_confidence.toFixed(3) : "–"}</td></tr>`;
    }
    dl($("d-summary"), [["Events", s.events], ["Alerts", s.alerts], ["Suppressed alerts (cooldown)", s.suppressed_alerts],
      ["Discarded candidates", s.discarded_candidates], ["Processing time", s.processing_s != null ? `${s.processing_s} s` : ""],
      ["Analysed frames/s", s.analysed_frames_per_s], ["Frame stride", s.stride]]);
    const etb = document.querySelector("#d-events tbody"); etb.innerHTML = "";
    for (const e of ev.events) {
      const end = e.end_s ?? e.last_seen_s;
      const img = (name) => name ? `<a href="${fileUrl(id, name)}" target="_blank"><img class="thumb" src="${fileUrl(id, name)}"></a>` : "–";
      etb.innerHTML += `<tr><td class="cls-${esc(e.class_name)}">${esc(e.class_name)}</td><td>${esc(e.status)}` +
        `${e.alert_suppressed ? " (alert suppressed)" : ""}</td><td>${fmtTime(e.start_s)}</td><td>${fmtTime(e.alert_s)}</td>` +
        `<td>${e.end_s != null ? fmtTime(e.end_s) : "ongoing"}</td><td>${(end - e.start_s).toFixed(1)} s</td>` +
        `<td>${e.hit_frames}</td><td>${e.peak_confidence.toFixed(3)}</td><td>${img(e.alert_keyframe || e.keyframe)}</td>` +
        `<td>${img(e.peak_keyframe)}</td></tr>`;
    }
    $("d-noevents").classList.toggle("hidden", ev.events.length > 0);
    const dtb = document.querySelector("#d-dets tbody"); dtb.innerHTML = "";
    for (const it of dets.items) {
      const txt = it.detections.map((d) => `${d.class_name} ${d.confidence.toFixed(2)} [${d.bbox.map((v) => Math.round(v)).join(", ")}]`).join("; ");
      dtb.innerHTML += `<tr><td>${it.frame}</td><td>${fmtTime(it.time_s)}</td><td>${esc(txt)}</td></tr>`;
    }
    $("d-detcount").textContent = `${dets.frames_with_detections} analysed frames contain detections (first ${dets.items.length} shown; full list: /api/jobs/${id}/detections).`;
    $("d-video").innerHTML = s.annotated_video
      ? `Annotated video (analysed frames, ${esc(s.annotated_video_codec)}): <a href="${fileUrl(id, s.annotated_video)}" download>download annotated.mp4</a>`
      : "";
  } catch (e) { showBanner(`Cannot load job ${id}: ${e.message}`); }
}

async function refreshAlerts() {
  try {
    const r = await api("/api/alerts?limit=100");
    const tb = document.querySelector("#alerts tbody"); tb.innerHTML = "";
    for (const a of r.alerts) {
      tb.innerHTML += `<tr class="clickable" data-job="${esc(a.job_id)}"><td>${esc(fmtDate(a.created_at))}</td><td>${esc(a.source_name)}</td>` +
        `<td class="cls-${esc(a.class_name)}">${esc(a.class_name)}</td><td>${fmtTime(a.time_s)}</td>` +
        `<td>${a.confidence.toFixed(3)}</td><td>${a.keyframe ? `<a href="${fileUrl(a.job_id, a.keyframe)}" target="_blank">frame ${a.frame_index}</a>` : a.frame_index}</td></tr>`;
    }
    tb.querySelectorAll("tr").forEach((tr) => tr.addEventListener("click", (ev) => {
      if (ev.target.tagName === "A") return;
      selectedJob = tr.dataset.job; showDetail(tr.dataset.job);
    }));
    $("no-alerts").classList.toggle("hidden", r.alerts.length > 0);
  } catch (e) { /* banner already shown by other calls */ }
}

async function tick() {
  if (ticking) return;  // never overlap polls
  ticking = true;
  try {
    await refreshHealth();
    const jobs = await refreshJobs();
    await refreshAlerts();
    if (selectedJob) {
      const j = jobs.find((x) => x.id === selectedJob);
      const active = j && (j.status === "queued" || j.status === "running");
      // refresh while running and once more when the status changes (e.g. to completed)
      if (j && (active || j.status !== lastSelectedStatus)) await showDetail(selectedJob);
      lastSelectedStatus = j ? j.status : null;
    }
  } finally { ticking = false; }
}

$("save-token").addEventListener("click", () => {
  token = $("token").value.trim();
  try { localStorage.setItem("firewatch_token", token); } catch (e) { /* storage unavailable */ }
  $("auth-card").hidden = true; tick(); refreshVideos();
});
$("refresh-videos").addEventListener("click", refreshVideos);
$("submit-video").addEventListener("click", submitVideo);
$("upload-btn").addEventListener("click", uploadVideo);
$("stream-btn").addEventListener("click", startStream);
$("d-close").addEventListener("click", () => { selectedJob = null; $("detail").classList.add("hidden"); });

refreshVideos();
tick();
setInterval(tick, 2000);
