import { Engine } from "./gpu.js";
import { Tracker } from "./tracker.js";
import { nearRx, taboToScreen, blurInfo } from "./optics.js";
import { drawReading, drawSentence, drawChart, chartRows, PARAGRAPH, SENTENCE } from "./content.js";

// ---------------- persisted state ----------------
const KEY = "visaoclara.v1";
const DEF = { rx: { od: { S: 0, C: 0, A: 0 }, os: { S: 0, C: 0, A: 0 } },
  set: { cardPx: 325, manualCm: 0, calib: 1, accom: 0, robust: true, rgb: false, hdr: true, pre: false, res: 1024,
         pupil: 4, K: 0.01, b: 0.8, font: 17, method: "fista" }, chartLog: [] };
let S = JSON.parse(JSON.stringify(DEF));
try { const j = JSON.parse(localStorage.getItem(KEY)); if (j) S = { ...S, ...j, set: { ...S.set, ...j.set }, rx: { ...S.rx, ...j.rx } }; } catch (e) {}
const save = () => localStorage.setItem(KEY, JSON.stringify(S));
const $ = (id) => document.getElementById(id);
const cssMM = () => 53.98 / S.set.cardPx;     // card SHORT side (fits a phone in portrait)

// ---------------- tracking ----------------
const tracker = new Tracker(); tracker.calib = S.set.calib;
let rawD = null;
tracker.onUpdate = (s) => { rawD = s.d; updateBadge(); };
const distance = () => S.set.manualCm > 0 ? S.set.manualCm / 100 : (tracker.state.ok || rawD ? tracker.state.d : 0.30);
function updateBadge() {
  const t = tracker.state, d = distance();
  $("distBadge").textContent = S.set.manualCm > 0 ? `${(d * 100).toFixed(0)} cm (manual)` : t.ok ? `${(d * 100).toFixed(0)} cm · câmera` : `${(d * 100).toFixed(0)} cm (${t.err ? "padrão, sem câmera" : rawD ? "último" : "padrão"})`;
}
$("camBtn").onclick = async () => {
  $("camInfo").textContent = "Carregando detector de rosto (no aparelho)…";
  const ok = await tracker.start($("cam"));
  $("camInfo").textContent = ok ? "Câmera ativa: distância e posição dos olhos medidas no aparelho." : "Câmera indisponível — usando 30 cm (ajuste manual em Ajustes). " + (tracker.state.err || "");
  updateBadge();
};

// ---------------- engines ----------------
let main = null, small = null, caps = {};
function makeEngines() {
  try {
    const c = document.createElement("canvas"); main = new Engine(c, +S.set.res);
    caps.hdr = S.set.hdr ? main.enableHDR() : false;
    const c2 = document.createElement("canvas"); small = new Engine(c2, 512, { preserve: true });
    caps.webgl = true;
  } catch (e) { caps.webgl = false; caps.err = String(e); }
  caps.dynHigh = matchMedia("(dynamic-range: high)").matches;
  caps.p3 = matchMedia("(color-gamut: p3)").matches;
  caps.cam = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  window.__vc = { main, small, S, caps, tracker };
}
makeEngines();

function applyParams(E) {
  Object.assign(E.params, { K: S.set.K, b: S.set.b, rgb: S.set.rgb, method: S.set.method, hi: E.hdr ? 1.6 : 1.0 });
}
// kernels: list of (eye x distance) PSF models with weights (joint least squares)
function kernels(rxByEye, weights, pupil) {
  const d0 = distance(), t = tracker.state, ks = [];
  const ds = S.set.robust ? [[d0, 0.5], [d0 - 0.03, 0.25], [d0 + 0.03, 0.25]] : [[d0, 1]];
  const eyes = Object.keys(weights).filter(e => weights[e] > 0).sort((a, b) => weights[b] - weights[a]);
  for (const e of eyes) for (const [d, w] of ds) {
    const n = nearRx(rxByEye[e], d, S.set.accom);
    ks.push({ S: n.S, C: n.C, axisScreen: taboToScreen(n.A), d, pupil, yaw: t.ok ? t.yaw : 0, pitch: t.ok ? t.pitch : 0, roll: t.ok ? t.roll : 0, w: w * weights[e], eye: e });
  }
  return ks;
}
function autoWeights() {
  const t = tracker.state;
  if (t.ok && t.rightOpen && !t.leftOpen) return { od: 1, os: 0 };
  if (t.ok && t.leftOpen && !t.rightOpen) return { od: 0, os: 1 };
  return { od: 0.5, os: 0.5 };   // both open: binocular compromise
}
const q = (k) => JSON.stringify(k.map(x => [x.S.toFixed(2), x.C.toFixed(2), Math.round(x.axisScreen + x.roll), (x.d * 100).toFixed(0), x.pupil, Math.round(x.yaw / 5), Math.round(x.pitch / 5), x.w.toFixed(2)]));

function predistortFor(k, fontPx, pixMM) {
  if (!S.set.pre || !k) return null;
  const s = k.S, c = k.C, sc = k.pupil * 1e-3 * k.d * 1000 / pixMM;
  const m1 = Math.abs(s) * sc, m2 = Math.abs(s + c) * sc;
  const ang = ((m1 >= m2 ? k.axisScreen + 90 : k.axisScreen) * Math.PI) / 180;
  return { angle: ang, scale: Math.max(0.8, 1 - 0.25 * Math.abs(m1 - m2) / fontPx) };
}

// A "view" = engine + how to draw its content + its PSF model
class View {
  constructor(E, host) { this.E = E; this.host = host; this.contentKey = ""; this.modelKey = ""; this.lastModel = 0; this.photo = null; }
  attach() { if (this.host && this.E.canvas.parentNode !== this.host) this.host.appendChild(this.E.canvas); }
  pixMM() { const w = this.E.canvas.getBoundingClientRect().width || Math.min(innerWidth, 520); return cssMM() * w / this.E.N; }
  scale() { const w = this.E.canvas.getBoundingClientRect().width || Math.min(innerWidth, 520); return this.E.N / w; }
  tick(spec, now) {
    const E = this.E; applyParams(E);
    const ks = spec.kernels();
    const mk = q(ks) + S.set.rgb;
    if (mk !== this.modelKey && now - this.lastModel > 250) {
      this.modelKey = mk; this.lastModel = now; E.setModel(ks, this.pixMM()); this.dirty = true;
      this.info = ks[0];
    }
    const ck = spec.contentKey() + (S.set.pre ? q(ks.slice(0, 1)) : "");
    if (ck !== this.contentKey) {
      this.contentKey = ck;
      const c = this.c2 || (this.c2 = document.createElement("canvas")); c.width = c.height = E.N;
      const ctx = c.getContext("2d", { willReadFrequently: false });
      spec.draw(ctx, E.N, this.scale(), this.pixMM(), ks[0]);
      E.setTarget(c, !!spec.photo); this.dirty = true;
    }
    if (E.params.b !== this._b || E.params.K !== this._K || E.params.rgb !== this._rgb) { this._b = E.params.b; this._K = E.params.K; this._rgb = E.params.rgb; E.retarget(); this.dirty = true; }
    const P = E.params, flags = [P.filter, P.split, P.retina, P.method].join();
    if (flags !== this._flags) { this._flags = flags; this.dirty = true; }
    if (P.method === "fista" && E.iters < (spec.maxIters || 60)) { E.iterate(spec.itPerFrame || (E.N > 512 ? 2 : 4)); this.dirty = true; }
    if (this.dirty) { E.render(); this.dirty = false; }
  }
}

// ---------------- screens ----------------
let screen = "home";
const views = {};
function route() {
  screen = (location.hash || "#home").slice(1);
  document.querySelectorAll("section").forEach(s => s.classList.toggle("on", s.dataset.screen === screen));
  if (screen === "read" && main) { views.read = views.read || new View(main, $("stageRead")); views.read.host = $("stageRead"); views.read.attach(); views.read.contentKey = ""; views.read.modelKey = ""; }
  if (screen === "read" && main) Object.assign(main.params, { filter: $("tFilter").checked, split: $("tSplit").checked, retina: $("tRetina").checked });
  if (screen === "chart" && main) Object.assign(main.params, { filter: $("cFilter").checked, split: false, retina: $("cRetina").checked });
  if (screen === "chart" && main) { views.chart = views.chart || new View(main, $("stageChart")); views.chart.host = $("stageChart"); views.chart.attach(); views.chart.contentKey = ""; views.chart.modelKey = ""; renderChartChips(); }
  if (screen === "discover") disc.start();
  if (screen === "settings") showCaps();
  if (screen === "rx") fillRx();
  window.scrollTo(0, 0);
}
addEventListener("hashchange", route);

// ----- Rx form -----
const opt = (sel, from, to, step, fmt, val) => { sel.innerHTML = ""; for (let v = from; step > 0 ? v <= to + 1e-9 : v >= to - 1e-9; v += step) { const o = document.createElement("option"); o.value = v.toFixed(2); o.textContent = fmt(v); sel.appendChild(o); } sel.value = (+val).toFixed(2); };
const fD = (v) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(2);
function fillRx() {
  for (const e of ["od", "os"]) {
    opt($(e + "S"), 6, -10, -0.25, fD, S.rx[e].S); opt($(e + "C"), 0, -6, -0.25, fD, S.rx[e].C); opt($(e + "A"), 0, 180, 5, (v) => v.toFixed(0) + "°", S.rx[e].A);
  }
}
function readRx() { for (const e of ["od", "os"]) S.rx[e] = { S: +$(e + "S").value, C: +$(e + "C").value, A: +$(e + "A").value }; save(); }
$("rxApply").onclick = () => { readRx(); location.hash = "#read"; };
$("rxTune").onclick = () => { readRx(); tune.begin(["od", "os"], () => (location.hash = "#read")); };

// ----- Reading -----
const readSpec = {
  kernels: () => kernels(S.rx, autoWeights(), S.set.pupil),
  contentKey: () => [S.set.font, main.params.split, views.read?.photo ? "photo" : "text", innerWidth].join(),
  draw: (ctx, N, scale, pixMM, k0) => {
    const ph = views.read?.photo;
    if (ph) { ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, N, N); const r = Math.max(N / ph.width, N / ph.height); ctx.drawImage(ph, (N - ph.width * r) / 2, (N - ph.height * r) / 2, ph.width * r, ph.height * r); readSpec.photo = true; return; }
    readSpec.photo = false;
    drawReading(ctx, N, { text: PARAGRAPH, fontCss: S.set.font, scale, split: main.params.split, grid: true, predistort: predistortFor(k0, S.set.font * scale, pixMM) });
  },
};
function bindRead() {
  const sync = () => {
    S.set.K = Math.pow(10, +$("sK").value); S.set.b = +$("sB").value; S.set.pupil = +$("sP").value; S.set.font = +$("sF").value; S.set.method = $("method").value;
    $("vK").textContent = S.set.K.toFixed(3); $("vB").textContent = S.set.b.toFixed(2); $("vP").textContent = S.set.pupil.toFixed(2) + " mm"; $("vF").textContent = S.set.font + " pt";
    if (main) Object.assign(main.params, { filter: $("tFilter").checked, split: $("tSplit").checked, retina: $("tRetina").checked });
    save();
  };
  $("sK").value = Math.log10(S.set.K); $("sB").value = S.set.b; $("sP").value = S.set.pupil; $("sF").value = S.set.font; $("method").value = S.set.method;
  for (const id of ["sK", "sB", "sP", "sF", "method", "tFilter", "tSplit", "tRetina"]) $(id).addEventListener("input", sync);
  $("photo").onchange = async (e) => { const f = e.target.files[0]; if (!f) return; views.read.photo = await createImageBitmap(f); views.read.contentKey = ""; };
  sync();
}
bindRead();

// ----- Chart -----
let rows = chartRows(7);
const chartSpec = {
  kernels: () => kernels(S.rx, autoWeights(), S.set.pupil),
  contentKey: () => [Math.round(distance() * 100), S.set.cardPx].join(),
  draw: (ctx, N, scale, pixMM) => { chartSpec.shown = drawChart(ctx, N, distance(), pixMM, rows); setTimeout(renderChartChips, 0); },
};
function renderChartChips() {
  const box = $("chartRows"); box.innerHTML = "";
  for (const r of (chartSpec.shown || [])) { const b = document.createElement("button"); b.textContent = `${r.letters} (${r.lm.toFixed(1)})`;
    b.onclick = () => { S.chartLog.push({ t: Date.now(), lm: r.lm, filter: $("cFilter").checked, d: distance(), rx: S.rx }); save(); $("chartLog").textContent = `Registrado: logMAR ${r.lm.toFixed(1)} com filtro ${$("cFilter").checked ? "ligado" : "desligado"} a ${(distance() * 100).toFixed(0)} cm.`; };
    box.appendChild(b); }
}
for (const id of ["cFilter", "cRetina"]) $(id).addEventListener("input", () => { if (main) Object.assign(main.params, { filter: $("cFilter").checked, retina: $("cRetina").checked }); });

// ----- small-engine renders (discovery preview + fine-tune variants) -----
function renderSmall(rxByEye, eye, drawFn, iters = 40) {
  const E = small, v = new View(E, null); applyParams(E);
  const pix = cssMM() * (Math.min(innerWidth, 520) * 0.48) / E.N, sc = E.N / (Math.min(innerWidth, 520) * 0.48);
  const ks = kernels(rxByEye, { [eye]: 1 }, S.set.pupil);
  const c = document.createElement("canvas"); c.width = c.height = E.N; drawFn(c.getContext("2d"), E.N, sc, pix, ks[0]);
  E.params.split = false; E.params.filter = true; E.params.retina = false;
  E.setTarget(c, false); E.setModel(ks, pix); E.retarget(); if (E.params.method === "fista") E.iterate(iters); E.render();
  const out = document.createElement("canvas"); out.width = out.height = E.N; out.getContext("2d").drawImage(E.canvas, 0, 0); return out;
}

// ----- Fine tune: 4 variants (axis ±10°/±5°, cyl ±0.25), 3 rounds, per eye -----
const tune = {
  begin(eyes, done) { this.eyes = eyes; this.ei = 0; this.round = 1; this.done = done; location.hash = "#tune"; setTimeout(() => this.show(), 50); },
  cands() {
    const e = this.eyes[this.ei], c = S.rx[e], dA = this.round === 1 ? 10 : 5, ax = (a) => ((a % 180) + 180) % 180;
    const list = [{ ...c, A: ax(c.A - dA) }, { ...c, A: ax(c.A + dA) }, { ...c, C: Math.min(0, c.C - 0.25) }, { ...c, C: Math.min(0, c.C + 0.25) }];
    if (this.round === 1 && Math.abs(c.C) >= 0.5) list[1] = { ...c, A: ax(c.A + 90) };   // resolves the fan's 90° ambiguity
    return list;
  },
  show() {
    const e = this.eyes[this.ei];
    $("tuneRound").textContent = `(rodada ${this.round}/3)`;
    $("tuneEye").textContent = `${e === "od" ? "Olho DIREITO" : "Olho ESQUERDO"} — feche ou cubra o outro olho. Atual: ${fD(S.rx[e].S)} ${fD(S.rx[e].C)} × ${S.rx[e].A}°`;
    const box = $("variants"); box.innerHTML = "<p class='small'>Calculando…</p>";
    setTimeout(() => {
      box.innerHTML = "";
      this.cands().forEach((rx, i) => {
        const cv = renderSmall({ [e]: rx }, e, (ctx, N, sc, pix, k0) => drawSentence(ctx, N, SENTENCE, 15, sc, predistortFor(k0, 15 * sc, pix)));
        cv.onclick = () => this.pick(i); cv.setAttribute("aria-label", `versão ${i + 1}`); box.appendChild(cv);
      });
    }, 30);
  },
  pick(i) {
    const e = this.eyes[this.ei];
    if (i !== null) S.rx[e] = this.cands()[i];
    save();
    if (this.round < 3) { this.round++; this.show(); return; }
    this.ei++; this.round = 1;
    if (this.ei < this.eyes.length) this.show(); else this.done();
  },
};
$("tuneNone").onclick = () => tune.pick(null);

// ----- Discovery: 40 cm lock, one eye at a time, sphere slider, fan, cyl, fine tune -----
const disc = {
  start() { this.eye = "od"; this.step = 0; this.work = { od: { S: 0, C: 0, A: 0 }, os: { S: 0, C: 0, A: 0 } }; this.render(); },
  render() {
    const box = $("discStep"), e = this.eye, other = e === "od" ? "esquerdo" : "direito";
    const eyeName = e === "od" ? "DIREITO" : "ESQUERDO";
    const w = this.work[e];
    if (this.step === 0) {
      box.innerHTML = `<p>Segure o celular a <b>40 cm</b> dos olhos (um palmo e meio). ${tracker.state.ok ? "" : "Sem câmera: use uma régua."}</p>
        <p class="big" id="dLive"></p>
        <label class="row">Sua idade <select id="age"><option value="">—</option>${[15,20,25,30,35,40,45,50,55,60,65,70].map(a => `<option>${a}</option>`).join("")}</select></label>
        <p class="small">A idade estima quanto o olho ainda foca de perto (acomodação). Abaixo de ~40 anos o olho compensa sozinho parte do grau.</p>
        <button class="btn primary" id="dGo">Estou a 40 cm</button>`;
      $("age").onchange = () => { const a = +$("age").value; if (a) { S.set.accom = Math.max(0, 0.5 * (15 - 0.25 * a)); save(); } };
      const live = () => { const el = $("dLive"); if (!el) return; const d = distance(); const ok = Math.abs(d - 0.40) <= 0.03;
        el.innerHTML = tracker.state.ok ? `<span class="${ok ? "ok" : "bad"}">${(d * 100).toFixed(0)} cm</span>` : "câmera desligada"; if (screen === "discover" && this.step === 0) requestAnimationFrame(live); };
      live();
      $("dGo").onclick = () => { if (!tracker.state.ok) S.set.manualCm = 40; updateBadge(); this.step = 1; this.render(); };
    } else if (this.step === 1) {
      box.innerHTML = `<p>Olho <b>${eyeName}</b>: feche ou cubra o olho ${other}.</p><p class="small" id="occ"></p><button class="btn primary" id="dGo">Pronto</button>`;
      const chk = () => { const el = $("occ"); if (!el) return; const t = tracker.state; const closed = e === "od" ? !t.leftOpen : !t.rightOpen;
        el.textContent = t.ok ? (closed ? "✓ olho fechado detectado" : "(não detectado — se cobriu com a mão, tudo bem)") : ""; if (this.step === 1 && screen === "discover") setTimeout(chk, 200); };
      chk(); $("dGo").onclick = () => { this.step = 2; this.render(); };
    } else if (this.step === 2) {
      box.innerHTML = `<p>Arraste até a frase ficar <b>o mais nítida possível</b> (filtro ligado).</p><div id="pv"></div>
        <input type="range" id="sph" min="-8" max="2" step="0.25" value="${w.S}" style="width:100%"><p class="big" id="sphV"></p>
        <button class="btn primary" id="dGo">Esta é a mais nítida</button>`;
      const upd = () => { w.S = +$("sph").value; $("sphV").textContent = `esférico ${fD(w.S)} D`; this.preview(); };
      $("sph").oninput = () => { clearTimeout(this._t); this._t = setTimeout(upd, 60); }; upd();
      $("dGo").onclick = () => { this.step = 3; this.render(); };
    } else if (this.step === 3) {
      let svg = `<svg viewBox="-110 -110 220 120" class="fan">`;
      for (let a = 0; a < 180; a += 10) { const r = a * Math.PI / 180, x = Math.cos(r), y = -Math.sin(r);
        svg += `<line x1="${x * 22}" y1="${y * 22}" x2="${x * 95}" y2="${y * 95}" stroke="#000" stroke-width="2.2"/><text x="${x * 104}" y="${y * 104 + 3}" font-size="7" text-anchor="middle">${a}</text>
        <line data-a="${a}" x1="0" y1="0" x2="${x * 110}" y2="${y * 110}" stroke="transparent" stroke-width="12"/>`; }
      box.innerHTML = `<p>Olho ${eyeName}. Toque na linha <b>mais escura/nítida</b>.</p>${svg}</svg><button class="btn" id="dSame">Todas iguais (sem astigmatismo)</button>`;
      box.querySelectorAll("line[data-a]").forEach(l => l.onclick = () => {
        const axisScreen = (+l.dataset.a + 90) % 180; w.A = (180 - axisScreen) % 180; w.C = -0.75; this.step = 4; this.render(); });
      $("dSame").onclick = () => { w.C = 0; this.finishEye(); };
    } else if (this.step === 4) {
      box.innerHTML = `<p>Ajuste o cilíndrico até a frase e a grade ficarem mais nítidas.</p><div id="pv"></div>
        <input type="range" id="cyl" min="-4" max="0" step="0.25" value="${w.C}" style="width:100%"><p class="big" id="cylV"></p>
        <button class="btn primary" id="dGo">Continuar para ajuste fino</button>`;
      const upd = () => { w.C = +$("cyl").value; $("cylV").textContent = `cil ${fD(w.C)} × ${w.A}°`; this.preview(true); };
      $("cyl").oninput = () => { clearTimeout(this._t); this._t = setTimeout(upd, 60); }; upd();
      $("dGo").onclick = () => this.finishEye();
    }
  },
  preview(grid = false) {
    const e = this.eye, cv = renderSmall(this.work, e, (ctx, N, sc, pix, k0) => {
      drawSentence(ctx, N, SENTENCE, 15, sc, predistortFor(k0, 15 * sc, pix));
      if (grid) { ctx.fillStyle = "#000"; for (let x = 40; x < N - 40; x += 28) ctx.fillRect(x, N * 0.72, 2, N * 0.22); for (let y = N * 0.72; y < N * 0.94; y += 28) ctx.fillRect(40, y, N - 80, 2); }
    });
    cv.style.width = "100%"; const pv = $("pv"); if (pv) { pv.innerHTML = ""; pv.appendChild(cv); }
  },
  finishEye() {
    S.rx[this.eye] = { ...this.work[this.eye] }; save();
    const next = this.eye === "od" ? "os" : null;
    tune.begin([this.eye], () => {
      if (next) { this.eye = next; this.step = 1; location.hash = "#discover"; setTimeout(() => { this.step = 1; this.render(); }, 30); }
      else { if (S.set.manualCm === 40 && !tracker.state.ok) S.set.manualCm = 0; save(); location.hash = "#read"; }
    });
  },
};
// discovery start() is called on route; keep the eye/step when coming back from tune
const _start = disc.start.bind(disc);
disc.start = function () { if (this._resume) { this._resume = false; return; } _start(); };
const _finish = disc.finishEye.bind(disc);
disc.finishEye = function () { this._resume = true; _finish(); };

// ----- Settings -----
function bindSettings() {
  const sync = () => {
    S.set.cardPx = +$("sCard").value; $("cardBar").style.width = S.set.cardPx + "px";
    S.set.manualCm = +$("sDist").value; $("vDist").textContent = S.set.manualCm ? S.set.manualCm + " cm" : "automática (câmera ou 30 cm)";
    S.set.accom = +$("sAcc").value; $("vAcc").textContent = S.set.accom.toFixed(2) + " D";
    S.set.robust = $("fRobust").checked; S.set.rgb = $("fRGB").checked; S.set.pre = $("fPre").checked;
    const hdrWas = S.set.hdr, resWas = S.set.res; S.set.hdr = $("fHDR").checked; S.set.res = +$("res").value;
    save(); updateBadge();
    if (hdrWas !== S.set.hdr || resWas !== S.set.res) location.reload();
  };
  $("sCard").value = S.set.cardPx; $("sDist").value = S.set.manualCm; $("sAcc").value = S.set.accom;
  $("fRobust").checked = S.set.robust; $("fRGB").checked = S.set.rgb; $("fHDR").checked = S.set.hdr; $("fPre").checked = S.set.pre; $("res").value = S.set.res;
  for (const id of ["sCard", "sDist", "sAcc", "fRobust", "fRGB", "fHDR", "fPre", "res"]) $(id).addEventListener("change", sync);
  $("sCard").addEventListener("input", () => { $("cardBar").style.width = $("sCard").value + "px"; });
  $("calib40").onclick = () => { if (!tracker.state.ok) { alert("Ative a câmera primeiro (tela inicial)."); return; }
    S.set.calib = S.set.calib * 0.40 / tracker.state.d; tracker.calib = S.set.calib; tracker._dHist = []; save(); alert("Calibrado."); };
  $("reset").onclick = () => { if (confirm("Apagar grau e ajustes salvos neste aparelho?")) { localStorage.removeItem(KEY); location.reload(); } };
  sync();
}
function showCaps() {
  $("caps").innerHTML = `WebGL2 float: <b>${caps.webgl ? "sim" : "não"}</b>${caps.err ? " (" + caps.err + ")" : ""} · HDR/EDR no canvas: <b>${caps.hdr ? "sim (folga 1,6×)" : "não suportado — usando [0,1]"}</b>
   · tela HDR (CSS): ${caps.dynHigh ? "sim" : "não"} · P3: ${caps.p3 ? "sim" : "não"} · câmera: ${caps.cam ? "sim" : "não"} · mm por px CSS: ${cssMM().toFixed(4)}`;
}
bindSettings();

// ---------------- main loop ----------------
function loop(now) {
  try {
    if (main && screen === "read") { views.read.tick(readSpec, now); const k = views.read.info; if (k) { const b = blurInfo({ S: k.S, C: k.C, A: 0 }, k.pupil, k.d, views.read.pixMM());
      $("readInfo").innerHTML = `Borrão residual ≈ ${b.D.toFixed(2)} D → ${b.arcmin.toFixed(0)}′ (${b.px.toFixed(0)} px). ${b.D > 2 ? "<b>Acima de ~2 D a tela não recupera letras pequenas — aumente a fonte/brilho.</b>" : b.D < 0.25 ? "Quase sem borrão nesta distância." : "Faixa onde a pré-compensação ajuda (~1 linha)."} Iterações: ${main.iters}${views.read.photo ? " · <a href='#' id='backText'>voltar ao texto</a>" : ""}`;
      const bt = $("backText"); if (bt) bt.onclick = (e) => { e.preventDefault(); views.read.photo = null; views.read.contentKey = ""; }; } }
    if (main && screen === "chart") views.chart.tick(chartSpec, now);
  } catch (e) { console.error(e); $("readInfo").textContent = "Erro: " + e; }
  requestAnimationFrame(loop);
}
if (!caps.webgl) document.querySelector("main").insertAdjacentHTML("afterbegin", `<div class="warn">Este navegador não suporta WebGL2 com float: ${caps.err}</div>`);
route(); updateBadge(); requestAnimationFrame(loop);
if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
