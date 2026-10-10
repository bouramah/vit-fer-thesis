import { EMOTIONS, MODELS, t, setLang, getLang, apply, emoName } from "./i18n.js";

const CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"];
const MP_VERSION = "0.10.14";
const MP_MODEL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";
// MediaPipe ne sert qu'à animer les cadres entre deux réponses du serveur. Son cadre (carré, sans le front)
// est converti en cadre « type YuNet », mesuré sur nos photos de test : hauteur ≈ 1,17 × côté MediaPipe,
// largeur ≈ 0,78 × hauteur, centre remonté de 12 % de la hauteur.
const MP_TO_YUNET = { size: 1.17, aspect: 0.78, dy: -0.12 };
const SERVER_WIDTH = 800;   // largeur des images envoyées pour la détection YuNet
function mpToYunet(bb) {
  const h = Math.max(bb.width, bb.height) * MP_TO_YUNET.size, w = h * MP_TO_YUNET.aspect;
  const cx = bb.originX + bb.width / 2, cy = bb.originY + bb.height / 2 + MP_TO_YUNET.dy * h;
  return { x: cx - w / 2, y: cy - h / 2, w, h };
}

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const pct = (v) => `${Math.round(v * 100)} %`;

// Modèle par vue : ResNet-50 en direct (le plus rapide sur CPU), DBFN ailleurs ; le choix de l'utilisateur est mémorisé par vue.
const DEFAULT_MODEL = { live: "resnet" };
const state = { model: "dbfn", modelByView: {}, cfg: { crop_scale: 1.3, crop_dy: -0.05, max_faces: 8 }, view: "home" };
const langHooks = [];

// =====================================================================
// API
// =====================================================================
async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  if (!r.ok) throw new Error(`${path} ${r.status}`);
  return r.json();
}

// =====================================================================
// Navigation, langue, modèle
// =====================================================================
function route() {
  const v = (location.hash || "#home").slice(1);
  const name = ["home", "live", "photo", "challenge", "guide", "about"].includes(v) ? v : "home";
  if (state.view === "live" && name !== "live") live.stop();
  if (state.view === "challenge" && name !== "challenge") challenge.abort();
  state.view = name;
  const m = state.modelByView[name] || DEFAULT_MODEL[name] || "dbfn";
  if (m !== state.model) setModel(m);
  $$(".view").forEach((s) => s.classList.toggle("on", s.id === `view-${name}`));
  $$(".nav a").forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#${name}`));
  if (name === "home") refreshStats();
  window.scrollTo(0, 0);
}

function setModel(m) {
  state.model = m;
  state.modelByView[state.view] = m;
  $$("#modelSeg button").forEach((b) => b.classList.toggle("on", b.dataset.model === m));
  $("#modelSel").value = m;
  live.onModel();
  photo.onModel();
}

function initChrome() {
  $$("#modelSeg button").forEach((b) => b.addEventListener("click", () => setModel(b.dataset.model)));
  $("#modelSel").addEventListener("change", (e) => setModel(e.target.value));
  $$("#langSeg button").forEach((b) => b.addEventListener("click", () => {
    setLang(b.dataset.lang);
    $$("#langSeg button").forEach((x) => x.classList.toggle("on", x.dataset.lang === getLang()));
    langHooks.forEach((f) => f());
  }));
  $$("#langSeg button").forEach((x) => x.classList.toggle("on", x.dataset.lang === getLang()));
  setModel("dbfn");
  window.addEventListener("hashchange", route);
}

async function refreshStats() {
  try {
    const s = await api("/api/stats");
    $("#stContrib").textContent = s.contributions;
    $("#stFeedback").textContent = s.feedback;
    $("#stCorrect").textContent = s.feedback ? pct(s.feedback_correct / s.feedback) : "–";
  } catch { /* hors ligne : on garde les tirets */ }
}

// =====================================================================
// Barres de probabilité
// =====================================================================
function modelsOf(res) { return ["resnet", "vit", "dbfn"].filter((m) => res && res[m]); }

function renderBars(host, res, extra = "") {
  if (!res) { host.innerHTML = ""; return; }
  const ms = modelsOf(res);
  const main = res.dbfn ? "dbfn" : ms[0];
  const top = res[main].top;
  let html = `<div class="verdict">${EMOTIONS[top].emoji} ${emoName(top)} <span class="muted small">${pct(res[main].confidence)}</span></div>${extra}`;
  if (ms.length > 1) {
    html += `<div class="chips">${ms.map((m) => `<div class="chip" style="--c:${EMOTIONS[res[m].top].color}"><small>${MODELS[m].name}</small>
      <b>${EMOTIONS[res[m].top].emoji} ${emoName(res[m].top)}</b><span>${pct(res[m].confidence)}</span></div>`).join("")}</div>`;
  }
  for (const m of ms.length > 1 ? ["dbfn"] : ms) {
    const p = res[m].probs;
    const best = p.indexOf(Math.max(...p));
    html += `<div class="bars"><h4><span>${MODELS[m].name}</span><span>${EMOTIONS[CLASSES[best]].emoji} ${pct(p[best])}</span></h4>`;
    CLASSES.forEach((c, i) => {
      html += `<div class="bar ${i === best ? "top" : ""}"><span>${EMOTIONS[c].emoji}</span><span>${emoName(c)}</span>
        <div class="track"><div class="fill" style="width:${(p[i] * 100).toFixed(1)}%;background:${EMOTIONS[c].color}"></div></div>
        <span class="v">${Math.round(p[i] * 100)}</span></div>`;
    });
    html += "</div>";
  }
  host.innerHTML = html;
}

// =====================================================================
// Caméra et détection de visages
// =====================================================================
const video = $("#video");
let stream = null, streamFacing = null;
async function startCamera(facing = "user") {
  if (stream && streamFacing === facing) return;
  if (stream) stopCamera();
  stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: facing }, audio: false });
  streamFacing = facing;
  video.srcObject = stream;
  await video.play();
  while (!video.videoWidth) await sleep(30);
}
function stopCamera() {
  if (stream) stream.getTracks().forEach((tr) => tr.stop());
  stream = null; streamFacing = null;
  video.srcObject = null;
}

let mpDetector = null, mpTried = false;
async function getDetector() {
  if (mpTried) return mpDetector;
  mpTried = true;
  try {
    const vision = await import(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/vision_bundle.mjs`);
    const files = await vision.FilesetResolver.forVisionTasks(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`);
    for (const delegate of ["GPU", "CPU"]) {
      try {
        mpDetector = await vision.FaceDetector.createFromOptions(files, {
          baseOptions: { modelAssetPath: MP_MODEL, delegate }, runningMode: "VIDEO", minDetectionConfidence: 0.55,
        });
        break;
      } catch (e) { console.warn("MediaPipe", delegate, e); }
    }
  } catch (e) { console.warn("MediaPipe indisponible", e); }
  return mpDetector;
}

// Carré de recadrage commun avec le serveur (faces.py) à partir d'un cadre de visage.
function cropSquare(b) {
  const side = Math.max(b.w, b.h) * state.cfg.crop_scale;
  const cx = b.x + b.w / 2, cy = b.y + b.h / 2 + state.cfg.crop_dy * b.h;
  return { sx: cx - side / 2, sy: cy - side / 2, side };
}
const cropCanvas = document.createElement("canvas");
function cropToDataURL(src, box, size = 224, square = null) {
  const q = square || cropSquare(box);
  cropCanvas.width = cropCanvas.height = size;
  const c = cropCanvas.getContext("2d");
  c.fillStyle = "#000";
  c.fillRect(0, 0, size, size);
  c.drawImage(src, q.sx, q.sy, q.side, q.side, 0, 0, size, size);
  return cropCanvas.toDataURL("image/jpeg", 0.88);
}

const iou = (a, b) => {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a.w * a.h + b.w * b.h - inter + 1e-6);
};

// Suivi léger des visages d'une image à l'autre, avec lissage des cadres et des probabilités.
class Tracker {
  constructor() { this.tracks = []; this.nextId = 1; }
  // src = "mp" (animation, 30 i/s) ou "srv" (YuNet, fait autorité : crée les visages et porte les prédictions)
  update(dets, now, src = "srv", gain = 0.6) {
    const used = new Set();
    for (const d of dets) {
      let best = null, bestIou = 0.25;
      for (const tr of this.tracks) {
        if (used.has(tr)) continue;
        const v = iou(tr.box, d);
        if (v > bestIou) { best = tr; bestIou = v; }
      }
      if (best) {
        used.add(best);
        for (const k of ["x", "y", "w", "h"]) best.box[k] += gain * (d[k] - best.box[k]);
        best[src === "mp" ? "mpSeen" : "srvSeen"] = now;
        if (d.res) smooth(best, d.res);
        if (d.shot) best.shot = d.shot;
      } else if (src === "srv") {
        const tr = { id: this.nextId++, box: { x: d.x, y: d.y, w: d.w, h: d.h }, srvSeen: now, mpSeen: 0, res: null };
        if (d.res) smooth(tr, d.res);
        if (d.shot) tr.shot = d.shot;
        this.tracks.push(tr);
        used.add(tr);
      }
    }
    this.tracks = this.tracks.filter((tr) => this.alive(tr, now));
  }
  alive(tr, now) { return now - tr.srvSeen < 1200 || (now - tr.mpSeen < 250 && now - tr.srvSeen < 3000); }
  visible(now) { return this.tracks.filter((tr) => this.alive(tr, now)).slice(0, state.cfg.max_faces); }
  largest(now) { return this.visible(now).sort((a, b) => b.box.w * b.box.h - a.box.w * a.box.h)[0]; }
}

function smooth(tr, res, alpha = 0.55) {
  if (!tr.res) { tr.res = JSON.parse(JSON.stringify(res)); return; }
  for (const m of Object.keys(res)) {
    if (!tr.res[m]) { tr.res[m] = JSON.parse(JSON.stringify(res[m])); continue; }
    const p = tr.res[m].probs.map((v, i) => v + alpha * (res[m].probs[i] - v));
    const best = p.indexOf(Math.max(...p));
    tr.res[m] = { probs: p, top: CLASSES[best], confidence: p[best] };
  }
  for (const m of Object.keys(tr.res)) if (!res[m]) delete tr.res[m];
}

// Moteur temps réel partagé par le mode direct et le défi.
class LiveEngine {
  constructor(canvas, modelFn) {
    this.canvas = canvas; this.ctx = canvas.getContext("2d");
    this.modelFn = modelFn; this.tracker = new Tracker();
    this.running = false; this.selected = null;
    this.detFps = 0; this.latency = 0; this.srvFps = 0;
    this.onFrame = null;
    canvas.addEventListener("click", (e) => this.pick(e));
  }
  async start() {
    await startCamera();
    const det = await getDetector();
    this.fit();
    this.canvas.hidden = false;
    this.running = true;
    this.tracker = new Tracker();
    this.loopDetect(det);
    this.loopInfer();
  }
  // Sur téléphone, les dimensions de la vidéo changent après le démarrage (passage en portrait, changement de caméra) :
  // le canvas doit les suivre, sinon les cadres sont dessinés dans un autre repère que l'image.
  fit() {
    const w = video.videoWidth, h = video.videoHeight;
    if (!w || !h || (this.canvas.width === w && this.canvas.height === h)) return;
    this.canvas.width = w; this.canvas.height = h;
    this.canvas.parentElement.style.aspectRatio = `${w} / ${h}`;
    this.tracker = new Tracker();
  }
  stop() { this.running = false; this.canvas.hidden = true; this.canvas.parentElement.style.aspectRatio = ""; }
  loopDetect(det) {
    let frames = 0, t0 = performance.now(), lastTs = -1;
    const tick = () => {
      if (!this.running) return;
      const now = performance.now();
      if (det && video.readyState >= 2 && now !== lastTs) {
        lastTs = now;
        try {
          const r = det.detectForVideo(video, now);
          this.tracker.update(r.detections.map((d) => mpToYunet(d.boundingBox)), now, "mp", 0.5);
        } catch (e) { console.warn(e); }
        frames++;
        if (now - t0 > 1000) { this.detFps = Math.round((frames * 1000) / (now - t0)); frames = 0; t0 = now; }
      }
      this.draw(now);
      if (this.onFrame) this.onFrame(now);
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  async loopInfer() {
    while (this.running) {
      const t0 = performance.now();
      try { await this.analyzeFrame(); } catch (e) { console.warn(e); await sleep(600); }
      const dt = performance.now() - t0;
      const fps = 1000 / Math.max(dt, 1);
      this.srvFps = this.srvFps ? 0.8 * this.srvFps + 0.2 * fps : fps;
      await sleep(Math.max(20, 120 - dt));
    }
  }
  async analyzeFrame() {
    // YuNet côté serveur : détection de tous les visages (même éloignés) + prédiction, recadrage identique au mode photo.
    const vw = video.videoWidth, vh = video.videoHeight, t = performance.now();
    // L'image analysée est conservée en pleine résolution : le défi enregistre exactement ce que le modèle a vu.
    const full = document.createElement("canvas");
    full.width = vw; full.height = vh;
    full.getContext("2d").drawImage(video, 0, 0, vw, vh);
    const w = Math.min(SERVER_WIDTH, vw), h = Math.round((vh / vw) * w);
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    c.getContext("2d").drawImage(full, 0, 0, w, h);
    const r = await api("/api/analyze", { image: c.toDataURL("image/jpeg", 0.85), model: this.modelFn() });
    if (video.videoWidth !== vw || video.videoHeight !== vh) return;   // la vidéo a changé de format entre-temps
    const k = vw / w;
    this.latency = r.latency_ms;
    this.tracker.update(r.faces.map((f) => {
      const box = { x: f.box[0] * k, y: f.box[1] * k, w: f.box[2] * k, h: f.box[3] * k };
      const res = Object.fromEntries(["resnet", "vit", "dbfn"].filter((m) => f[m]).map((m) => [m, f[m]]));
      return { ...box, res, shot: { frame: full, box, res, t } };
    }), performance.now(), "srv", 0.7);
  }
  pick(e) {
    const r = this.canvas.getBoundingClientRect();
    const sx = this.canvas.width / r.width, sy = this.canvas.height / r.height;
    const x = this.canvas.width - (e.clientX - r.left) * sx, y = (e.clientY - r.top) * sy;
    const tr = this.tracker.visible(performance.now()).find((t) => x >= t.box.x && x <= t.box.x + t.box.w && y >= t.box.y && y <= t.box.y + t.box.h);
    this.selected = tr ? tr.id : null;
  }
  current(now) {
    const vis = this.tracker.visible(now);
    return vis.find((tr) => tr.id === this.selected) || this.tracker.largest(now);
  }
  draw(now) {
    this.fit();
    const { ctx, canvas } = this;
    const W = canvas.width;
    ctx.save();
    ctx.translate(W, 0); ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, W, canvas.height);
    ctx.restore();
    const cur = this.current(now);
    for (const tr of this.tracker.visible(now)) {
      const b = tr.box, x = W - b.x - b.w;
      const main = tr.res && (tr.res.dbfn || tr.res[modelsOf(tr.res)[0]]);
      const col = main ? EMOTIONS[main.top].color : "#ffffff";
      ctx.lineWidth = tr === cur ? 5 : 3;
      ctx.strokeStyle = col;
      roundRect(ctx, x, b.y, b.w, b.h, 16); ctx.stroke();
      if (!tr.res) continue;
      const labels = modelsOf(tr.res).length > 1
        ? modelsOf(tr.res).map((m) => `${MODELS[m].short} ${EMOTIONS[tr.res[m].top].emoji} ${Math.round(tr.res[m].confidence * 100)}%`)
        : [`${EMOTIONS[main.top].emoji} ${emoName(main.top)} ${Math.round(main.confidence * 100)}%`];
      const fs = Math.max(16, Math.min(30, b.w / 9));
      ctx.font = `700 ${fs}px Inter, system-ui, sans-serif`;
      let y = b.y - 8 - (labels.length - 1) * (fs + 10);
      for (const lab of labels) {
        const tw = ctx.measureText(lab).width;
        ctx.fillStyle = col; roundRect(ctx, x, y - fs - 4, tw + 18, fs + 12, 10); ctx.fill();
        ctx.fillStyle = "#0b1020"; ctx.fillText(lab, x + 9, y + 2);
        y += fs + 10;
      }
    }
  }
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

// =====================================================================
// Mode direct
// =====================================================================
const live = (() => {
  const eng = new LiveEngine($("#liveCanvas"), () => state.model);
  let panelT = 0;
  eng.onFrame = (now) => {
    if (now - panelT < 200) return;
    panelT = now;
    const cur = eng.current(now);
    const n = eng.tracker.visible(now).length;
    $("#liveStatus").textContent = n
      ? `${n} ${t("live.faces")} · ${t("live.infer")} ${Math.round(eng.srvFps || 0)} ${t("fps")} (${Math.round(eng.latency)} ${t("lat")})${eng.detFps ? ` · ${t("live.track")} ${eng.detFps} ${t("fps")}` : ""}`
      : t("live.noface");
    renderBars($("#livePanel"), cur && cur.res);
  };
  async function start() {
    try {
      $("#liveStart").disabled = true;
      await eng.start();
      $("#liveEmpty").hidden = true;
      $("#liveStop").disabled = false; $("#liveCapture").disabled = false;
      $("#liveWarn").hidden = true;
    } catch (e) {
      console.error(e);
      $("#liveWarn").hidden = false; $("#liveWarn").textContent = t("cam.error");
    } finally { $("#liveStart").disabled = false; }
  }
  function stop() {
    eng.stop(); stopCamera();
    $("#liveEmpty").hidden = false;
    $("#liveStop").disabled = true; $("#liveCapture").disabled = true;
    $("#livePanel").innerHTML = ""; $("#liveStatus").textContent = "–";
  }
  function capture() {
    const c = document.createElement("canvas");
    c.width = video.videoWidth; c.height = video.videoHeight;
    c.getContext("2d").drawImage(video, 0, 0);
    const data = c.toDataURL("image/jpeg", 0.92);
    stop();
    location.hash = "#photo";
    photo.load(data, "live");
  }
  $("#liveStart").addEventListener("click", start);
  $("#liveStop").addEventListener("click", stop);
  $("#liveCapture").addEventListener("click", capture);
  return { stop: () => { if (eng.running) stop(); }, onModel: () => {} };
})();

// =====================================================================
// Mode photo
// =====================================================================
const photo = (() => {
  let img = null, dataUrl = null, source = "photo", faces = [];
  const canvas = $("#photoCanvas"), ctx = canvas.getContext("2d");

  async function load(url, src = "photo") {
    source = src;
    const im = new Image();
    im.src = url; await im.decode();
    const k = Math.min(1, 1600 / Math.max(im.width, im.height));
    const c = document.createElement("canvas");
    c.width = Math.round(im.width * k); c.height = Math.round(im.height * k);
    c.getContext("2d").drawImage(im, 0, 0, c.width, c.height);
    dataUrl = c.toDataURL("image/jpeg", 0.92);
    img = c;
    await analyze();
  }
  async function analyze() {
    if (!dataUrl) return;
    $("#photoMsg").textContent = t("photo.analyzing");
    try {
      const r = await api("/api/analyze", { image: dataUrl, model: state.model });
      faces = r.faces.map((f, i) => ({ ...f, idx: i + 1, crop: cropToDataURL(img, { x: f.box[0], y: f.box[1], w: f.box[2], h: f.box[3] }, 448) }));
      $("#photoMsg").textContent = faces.length ? "" : t("photo.none");
      render();
    } catch (e) { console.error(e); $("#photoMsg").textContent = t("c.err"); }
  }
  function render() {
    if (!img) return;
    $("#photoRes").hidden = false;
    $("#drop").classList.add("compact");
    canvas.width = img.width; canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    const host = $("#photoFaces");
    host.innerHTML = "";
    for (const f of faces) {
      const [x, y, w, h] = f.box;
      const res = { resnet: f.resnet, vit: f.vit, dbfn: f.dbfn };
      const main = res.dbfn || res[modelsOf(res)[0]];
      const col = EMOTIONS[main.top].color;
      ctx.lineWidth = Math.max(3, img.width / 300); ctx.strokeStyle = col;
      roundRect(ctx, x, y, w, h, 12); ctx.stroke();
      const fs = Math.max(16, Math.min(40, w / 6));
      ctx.font = `800 ${fs}px Inter, system-ui, sans-serif`;
      const lab = `${f.idx} ${EMOTIONS[main.top].emoji}`;
      const tw = ctx.measureText(lab).width;
      ctx.fillStyle = col; roundRect(ctx, x, y - fs - 12, tw + 16, fs + 10, 8); ctx.fill();
      ctx.fillStyle = "#0b1020"; ctx.fillText(lab, x + 8, y - 8);

      const card = document.createElement("div");
      card.className = "face-card";
      card.innerHTML = `<div class="head"><img src="${f.crop}" alt=""><div><b>${t("photo.face")} ${f.idx}</b><br>
        <button class="btn ghost small-btn" data-ex>🔍 ${t("photo.explain")}</button></div></div><div class="b"></div><div class="fb"></div>`;
      renderBars($(".b", card), res);
      $("[data-ex]", card).addEventListener("click", () => explain(f));
      feedbackWidget($(".fb", card), main.top, (label) => { f.true_label = label; });
      host.appendChild(card);
    }
    $("#photoBar").hidden = !faces.length;
    $("#photoBarInfo").textContent = `${faces.length} ${t("live.faces")}`;
  }
  $("#photoContrib").addEventListener("click", () => openConsent(faces.map((f) => ({
    image: f.crop, source, true_label: f.true_label || null,
    predictions: { resnet: f.resnet, vit: f.vit, dbfn: f.dbfn },
  }))));
  async function explain(f) {
    const r = await api("/api/explain", { face: f.crop });
    $("#exFace").src = f.crop; $("#exCam").src = r.overlay;
    $("#exLabel").textContent = `DBFN : ${EMOTIONS[r.class].emoji} ${emoName(r.class)}`;
    $("#explainDlg").showModal();
  }
  // Glisser-déposer et sélection de fichier
  const drop = $("#drop");
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) readFile(f); });
  $("#fileIn").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) readFile(f); e.target.value = ""; });
  function readFile(file) { const r = new FileReader(); r.onload = () => load(r.result, "photo"); r.readAsDataURL(file); }
  // Prise de photo avec aperçu : la personne se voit, déclenche elle-même (compte à rebours de 3 s).
  let facing = "user", shooting = false;
  async function openCam() {
    try {
      await startCamera(facing);
      const pv = $("#camPreview");
      pv.srcObject = stream;
      pv.classList.toggle("mirror", facing === "user");
      await pv.play().catch(() => {});
      const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
      $("#camSwitch").hidden = cams.length < 2;
      if (!$("#camDlg").open) $("#camDlg").showModal();
    } catch { $("#photoMsg").textContent = t("cam.error"); }
  }
  function closeCam() { $("#camDlg").close(); stopCamera(); shooting = false; $("#camCount").hidden = true; }
  $("#photoSnap").addEventListener("click", () => { facing = "user"; openCam(); });
  $("#camSwitch").addEventListener("click", () => { facing = facing === "user" ? "environment" : "user"; openCam(); });
  $("#camCancel").addEventListener("click", closeCam);
  $("#camDlg").addEventListener("cancel", () => { stopCamera(); shooting = false; });
  $("#camShoot").addEventListener("click", async () => {
    if (shooting) return;
    shooting = true;
    const cnt = $("#camCount");
    for (const n of [3, 2, 1]) {
      cnt.hidden = false; cnt.textContent = n; cnt.style.animation = "none"; void cnt.offsetWidth; cnt.style.animation = "";
      await sleep(900);
      if (!shooting) return;
    }
    cnt.hidden = true;
    const f = $("#camFlash"); f.classList.remove("go"); void f.offsetWidth; f.classList.add("go");
    const c = document.createElement("canvas");
    c.width = video.videoWidth; c.height = video.videoHeight;
    c.getContext("2d").drawImage(video, 0, 0);
    await sleep(250);
    closeCam();
    load(c.toDataURL("image/jpeg", 0.92), "snapshot");
  });
  langHooks.push(render);
  return { load, onModel: () => { if (state.view === "photo") analyze(); } };
})();

// =====================================================================
// Retour « la prédiction est-elle juste ? »
// =====================================================================
function feedbackWidget(host, predicted, onLabel, source = "photo") {
  const send = (label) => {
    api("/api/feedback", { model: state.model === "compare" ? "dbfn" : state.model, predicted, true_label: label, correct: label === predicted, source }).catch(() => {});
    onLabel(label);
    host.innerHTML = `<span class="ok">✓ ${t("fb.thanks")}</span>`;
  };
  host.innerHTML = `${t("fb.q")} <button class="btn ghost" data-y>👍 ${t("fb.yes")}</button> <button class="btn ghost" data-n>👎 ${t("fb.no")}</button>`;
  $("[data-y]", host).addEventListener("click", () => send(predicted));
  $("[data-n]", host).addEventListener("click", () => {
    host.innerHTML = `${t("fb.which")}<div class="opts">${CLASSES.map((c) => `<button class="btn ghost" data-c="${c}">${EMOTIONS[c].emoji} ${emoName(c)}</button>`).join("")}</div>`;
    $$("[data-c]", host).forEach((b) => b.addEventListener("click", () => send(b.dataset.c)));
  });
}

// =====================================================================
// Défi des 7 émotions
// =====================================================================
const challenge = (() => {
  const eng = new LiveEngine($("#chCanvas"), () => "compare");
  const HOLD_MS = 600, LIMIT_MS = 20000, THRESH = 0.55, GRACE_MS = 1500;
  let seq = [], step = 0, results = [], stepStart = 0, holdStart = 0, best = null, busy = false, active = false;

  function shuffle(a) { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }

  function showStep() {
    const c = seq[step];
    $("#chEmoji").textContent = EMOTIONS[c].emoji;
    $("#chName").textContent = emoName(c);
    $("#chTip").textContent = t(`ch.tip.${c}`);
    $("#chRing").style.setProperty("--c", EMOTIONS[c].color);
    $("#chRing").style.setProperty("--p", 0);
    $("#chStep").innerHTML = seq.map((_, i) => `<i class="${i < step ? (results[i].success ? "done" : "miss") : i === step ? "cur" : ""}"></i>`).join("");
    stepStart = performance.now(); holdStart = 0; best = null;
  }

  // Image et prédiction proviennent de la même analyse serveur (pas de l'image vidéo du moment, ni des probabilités lissées).
  function fromShot(shot) {
    return { image: cropToDataURL(shot.frame, shot.box, 448), predictions: JSON.parse(JSON.stringify(shot.res)) };
  }

  eng.onFrame = (now) => {
    if (!active || busy) return;
    const target = seq[step];
    const tr = eng.tracker.largest(now);
    const p = tr && tr.res && tr.res.dbfn ? tr.res.dbfn.probs[CLASSES.indexOf(target)] : 0;
    $("#chRing").style.setProperty("--p", Math.round(p * 100));
    const left = Math.max(0, 1 - (now - stepStart) / LIMIT_MS);
    $("#chTimer").style.width = `${left * 100}%`;
    // Meilleure analyse de l'étape, en ignorant les images prises pendant le délai de grâce
    // (la personne affiche encore l'expression précédente).
    const shot = tr && tr.shot;
    if (shot && shot.res.dbfn && shot.t - stepStart > GRACE_MS) {
      const ps = shot.res.dbfn.probs[CLASSES.indexOf(target)];
      if (!best || ps > best.p) best = { p: ps, shot };
    }
    if (p >= THRESH) {
      holdStart = holdStart || now;
      if (now - holdStart >= HOLD_MS) finishStep(true, tr);
    } else holdStart = 0;
    if (left === 0) finishStep(false, null);
  };

  async function finishStep(success, tr, skipped = false) {
    busy = true;
    const target = seq[step];
    // Émotion passée volontairement : rien n'est conservé (la personne ne l'a pas tentée).
    const pick = skipped ? null : best ? best.shot : success && tr && tr.shot ? tr.shot : null;
    const shot = pick ? fromShot(pick) : null;
    results.push({ target, success, skipped, ...(shot || { image: null, predictions: null }) });
    if (success) { flash(); burst(EMOTIONS[target].emoji); }
    await sleep(success ? 900 : 200);
    step++;
    busy = false;
    if (step >= seq.length) end(); else showStep();
  }

  function flash() { const f = $("#chFlash"); f.classList.remove("go"); void f.offsetWidth; f.classList.add("go"); }
  function burst(emoji) {
    const r = $("#chRing").getBoundingClientRect();
    for (let i = 0; i < 14; i++) {
      const s = document.createElement("span");
      s.className = "burst"; s.textContent = emoji;
      s.style.left = `${r.left + r.width / 2}px`; s.style.top = `${r.top + r.height / 2}px`;
      const a = Math.random() * Math.PI * 2, d = 90 + Math.random() * 140;
      s.style.setProperty("--dx", `${Math.cos(a) * d}px`); s.style.setProperty("--dy", `${Math.sin(a) * d}px`);
      document.body.appendChild(s); setTimeout(() => s.remove(), 1200);
    }
  }

  async function start() {
    try {
      $("#chStart").disabled = true;
      seq = shuffle([...CLASSES]); step = 0; results = [];
      $("#chIntro").hidden = true; $("#chEnd").hidden = true; $("#chRun").hidden = false;
      $$("#modelSeg button").forEach((b) => (b.disabled = true)); $("#modelSel").disabled = true;
      await eng.start();
      active = true;
      showStep();
    } catch (e) {
      console.error(e); abort(); alert(t("cam.error"));
    } finally { $("#chStart").disabled = false; }
  }

  function end() {
    active = false; eng.stop(); stopCamera();
    $$("#modelSeg button").forEach((b) => (b.disabled = false)); $("#modelSel").disabled = false;
    $("#chRun").hidden = true; $("#chEnd").hidden = false;
    renderRecap();
  }

  function renderRecap() {
    if (!results.length) return;
    $("#chScore").textContent = results.filter((r) => r.success).length;
    $("#chRecap").innerHTML = results.map((r) => {
      const rows = r.predictions ? ["resnet", "vit", "dbfn"].map((m) => {
        const top = r.predictions[m].top, ok = top === r.target;
        return `<div class="m"><span>${MODELS[m].name}</span><span class="${ok ? "ok" : "ko"}">${EMOTIONS[top].emoji} ${ok ? "✓" : "✗"}</span></div>`;
      }).join("") : `<div class="muted">–</div>`;
      return `<div class="rc">${r.image ? `<img src="${r.image}" alt="">` : ""}<div class="meta"><b>${EMOTIONS[r.target].emoji} ${emoName(r.target)}</b>${rows}</div></div>`;
    }).join("");
    $("#chContrib").disabled = !results.some((r) => r.image);
  }

  function abort() {
    active = false; eng.stop(); stopCamera();
    $$("#modelSeg button").forEach((b) => (b.disabled = false)); $("#modelSel").disabled = false;
    $("#chRun").hidden = true; $("#chEnd").hidden = true; $("#chIntro").hidden = false;
  }

  $("#chStart").addEventListener("click", start);
  $("#chSkip").addEventListener("click", () => { if (active && !busy) finishStep(false, null, true); });
  $("#chAgain").addEventListener("click", () => { $("#chEnd").hidden = true; start(); });
  $("#chContrib").addEventListener("click", () => openConsent(results.filter((r) => r.image).map((r) => ({
    image: r.image, source: "challenge", target: r.target, success: r.success, predictions: r.predictions,
  }))));
  langHooks.push(() => { if (active) showStep(); renderRecap(); });
  return { abort: () => { if (active || eng.running) abort(); } };
})();

// =====================================================================
// Guide des émotions
// =====================================================================
// Visages illustrés : chaque dessin reprend les indices décrits dans la fiche (sourcils, yeux, bouche, rides).
const FACE = {
  //            sourcils : [y extérieur, y intérieur, y contrôle, x intérieur], yeux : [ry, décalage iris], bouche
  neutral:  { brows: [50, 50, 47, 52], eyes: [5, 0], mouth: `<path d="M48 99 Q60 100 72 99" class="ln"/>` },
  angry:    { brows: [45, 57, 50, 55], eyes: [4, 1], lid: "angry",
              extra: `<path d="M57 52 L58 58 M63 52 L62 58" class="th"/>`,
              mouth: `<path d="M45 101 Q60 96 75 101 Q60 103 45 101 Z" class="mo"/><path d="M48 100.5 Q60 98 72 100.5" class="tw"/>` },
  disgust:  { brows: [50, 54, 50, 53], eyes: [3, 0], lid: "squint",
              extra: `<path d="M53 72 Q55 70 57 72 M63 72 Q65 70 67 72 M52 76 Q54 74 56 76 M64 76 Q66 74 68 76" class="th"/>`,
              mouth: `<path d="M46 101 Q52 92 58 96 Q62 93 68 94 Q73 96 74 101 Q60 104 46 101 Z" class="mo"/><path d="M52 97 Q60 95 70 96" class="tw"/>` },
  fear:     { brows: [45, 41, 44, 53], eyes: [8, 2],
              extra: `<path d="M53 33 Q60 35 67 33 M54 37 Q60 39 66 37" class="th"/>`,
              mouth: `<path d="M41 100 Q60 95 79 100 Q60 109 41 100 Z" class="mo"/><path d="M44 99.5 Q60 97 76 99.5" class="tw"/>` },
  happy:    { brows: [49, 49, 45, 52], eyes: [0, 0], lid: "smile",
              extra: `<circle cx="36" cy="80" r="7" class="ck"/><circle cx="84" cy="80" r="7" class="ck"/><path d="M24 62 L20 60 M24 66 L20 67 M96 62 L100 60 M96 66 L100 67" class="th"/>`,
              mouth: `<path d="M40 93 Q60 120 80 93 Q60 99 40 93 Z" class="mo"/><path d="M42 94 Q60 100 78 94 L77 98 Q60 103 43 98 Z" class="tw"/>` },
  sad:      { brows: [51, 42, 47, 52], eyes: [4, 1], lid: "sad",
              extra: `<path d="M54 117 Q60 113 66 117" class="th"/>`,
              mouth: `<path d="M46 105 Q60 94 74 105" class="ln"/>` },
  surprise: { brows: [44, 44, 33, 52], eyes: [8, 1],
              extra: `<path d="M44 30 Q60 27 76 30 M46 35 Q60 32 74 35" class="th"/>`,
              mouth: `<ellipse cx="60" cy="104" rx="8" ry="11" class="mo"/>` },
};
function eyeSVG(cx, f, side) {
  const [ry, dy] = f.eyes;
  if (f.lid === "smile") return `<path d="M${cx - 8} 65 Q${cx} 57 ${cx + 8} 65" class="ln"/>`;
  let s = `<ellipse cx="${cx}" cy="64" rx="8" ry="${ry}" class="ew"/><circle cx="${cx}" cy="${64 + dy}" r="${Math.min(3.6, ry)}" class="ir"/>`;
  const o = cx + side * 9, i = cx - side * 9;   // coin extérieur / intérieur
  if (f.lid === "angry") s += `<path d="M${o} ${64 - ry} L${i} ${64 - ry + 3}" class="ln"/>`;
  if (f.lid === "sad") s += `<path d="M${i} ${63 - ry} L${o} ${66 - ry}" class="ln"/>`;
  if (f.lid === "squint") s += `<path d="M${cx - 8} 68 Q${cx} 65 ${cx + 8} 68" class="ln"/>`;
  return s;
}
function faceSVG(c, label) {
  const f = FACE[c], [yo, yi, yc, xi] = f.brows;
  const brow = (side) => {
    const x = (v) => 60 + side * (v - 60);
    return `<path d="M${x(28)} ${yo} Q${x(40)} ${yc} ${x(xi)} ${yi}" class="br"/>`;
  };
  return `<svg viewBox="0 0 120 140" role="img" aria-label="${label}">
    <ellipse cx="60" cy="72" rx="44" ry="54" class="sk"/>
    <path d="M16 72 Q12 16 60 16 Q108 16 104 72 Q102 30 60 28 Q18 30 16 72 Z" class="hr"/>
    ${f.extra || ""}${brow(-1)}${brow(1)}${eyeSVG(42, f, -1)}${eyeSVG(78, f, 1)}
    <path d="M60 70 Q57 82 54 85 Q60 88 66 85" class="ln"/>
    ${f.mouth}
  </svg>`;
}

const GUIDE_TIPS = ["guide.tip1", "guide.tip2", "guide.tip3", "guide.tip4"];
function renderGuide() {
  $("#guideTips").innerHTML = GUIDE_TIPS.map((k) => `<div class="tip">${t(k)}</div>`).join("");
  $("#guideGrid").innerHTML = CLASSES.map((c) => `
    <article class="gcard" style="--c:${EMOTIONS[c].color}">
      <header><div class="gface">${faceSVG(c, `${t("guide.illus")} ${emoName(c)}`)}</div><h3><span class="gemo">${EMOTIONS[c].emoji}</span> ${emoName(c)}</h3></header>
      <ul>
        <li><b>${t("guide.brows")}</b> ${t(`g.${c}.brows`)}</li>
        <li><b>${t("guide.eyes")}</b> ${t(`g.${c}.eyes`)}</li>
        <li><b>${t("guide.mouth")}</b> ${t(`g.${c}.mouth`)}</li>
      </ul>
      <p class="gtip">💡 ${t(`g.${c}.tip`)}</p>
      <p class="gwarn">⚠️ ${t(`g.${c}.avoid`)}</p>
    </article>`).join("");
}
renderGuide();
langHooks.push(renderGuide);

// =====================================================================
// Consentement et contribution
// =====================================================================
let pendingItems = [];
function openConsent(items) {
  pendingItems = items;
  $("#consentForm").hidden = false; $("#consentOk").hidden = true; $("#cErr").hidden = true;
  $("#cCopy").textContent = t("c.copy");
  ["#cAdult", "#cSelf", "#cMemoire", "#cPub"].forEach((s) => ($(s).checked = false));
  $("#consentThumbs").innerHTML = items.map((it) => `<img src="${it.image}" alt="">`).join("");
  updateConsentBtn();
  $("#consentDlg").showModal();
}
function updateConsentBtn() { $("#cSubmit").disabled = !($("#cAdult").checked && $("#cSelf").checked && $("#cMemoire").checked); }
["#cAdult", "#cSelf", "#cMemoire"].forEach((s) => $(s).addEventListener("change", updateConsentBtn));
$("#cSubmit").addEventListener("click", async () => {
  $("#cSubmit").disabled = true;
  try {
    const r = await api("/api/contribute", {
      consent: { adult: $("#cAdult").checked, self: $("#cSelf").checked, memoire: $("#cMemoire").checked, publications: $("#cPub").checked, lang: getLang() },
      pseudonym: $("#cPseudo").value.trim(), contact: $("#cContact").value.trim(), items: pendingItems,
    });
    $("#cCode").textContent = r.withdrawal_code;
    $("#consentForm").hidden = true; $("#consentOk").hidden = false;
  } catch (e) {
    console.error(e); $("#cErr").hidden = false; $("#cErr").textContent = t("c.err"); updateConsentBtn();
  }
});
$("#cCopy").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText($("#cCode").textContent); $("#cCopy").textContent = t("c.copied"); } catch {}
});
$("#cSave").addEventListener("click", () => {
  const code = $("#cCode").textContent;
  const txt = `EmoVision — ${t("c.receipt")}\n\n${t("c.receipt.code")} : ${code}\n${t("c.receipt.date")} : ${new Date().toLocaleString(getLang())}\n`
    + `${t("c.receipt.n")} : ${pendingItems.length}\n\n${t("c.ok.d")}\n${location.origin}/#about\n`;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([txt], { type: "text/plain;charset=utf-8" }));
  a.download = `emovision-${code}.txt`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});
$("#cClose").addEventListener("click", () => $("#consentDlg").close());

// Retrait d'une contribution
$("#wBtn").addEventListener("click", async () => {
  const code = $("#wCode").value.trim();
  if (!code) return;
  try {
    const r = await api("/api/withdraw", { code });
    $("#wMsg").textContent = r.deleted ? t("w.ok") : t("w.no");
  } catch { $("#wMsg").textContent = t("c.err"); }
});

// =====================================================================
// Démarrage
// =====================================================================
(async function init() {
  apply();
  initChrome();
  try {
    state.cfg = await api("/api/config");
    if (state.cfg.contact_email) {
      $("#contactLine").hidden = false;
      $("#contactMail").textContent = state.cfg.contact_email;
      $("#contactMail").href = `mailto:${state.cfg.contact_email}`;
    }
  } catch (e) { console.warn("config", e); }
  route();
})();
