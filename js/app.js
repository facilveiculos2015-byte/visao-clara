import { Engine } from "./gpu.js";
import { METHODS, methodById } from "./methods.js";
import { WebGPUPresenter } from "./hdr.js";
import { CPUEngine } from "./cpu.js";
import { Tracker } from "./tracker.js";
import { nearRx, taboToScreen, blurInfo } from "./optics.js";
import { drawReading, drawSentence, drawChart, chartRows, PARAGRAPH, SENTENCE, DISC_TEXT } from "./content.js";

// ---------------- persisted state ----------------
const KEY = "visaoclara.v1";
const DEF = { rx: { od: { S: 0, C: 0, A: 0 }, os: { S: 0, C: 0, A: 0 } },
  set: { cardPx: 325, manualCm: 0, calib: 1, accom: 3, robust: false, rgb: false, hdr: true, pre: false, res: 1024,
         pupil: 4, K: 0.01, b: 0.6, font: 17, method: "fista", iters: 30, edrH: 2, bias: 0.10, ageSet: false }, chartLog: [], v: 4 };
let S = JSON.parse(JSON.stringify(DEF));
try { const j = JSON.parse(localStorage.getItem(KEY)); if (j) { const old = (j.v || 1) < 3, old4 = (j.v || 1) < 4; S = { ...S, ...j, set: { ...S.set, ...j.set }, rx: { ...S.rx, ...j.rx }, v: 4 };
  if (old) Object.assign(S.set, { robust: false, iters: 30, edrH: 2, bias: 0.10, hdr: true });
  if (old4 && !S.set.ageSet) S.set.accom = 3; } } catch (e) {}
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
// self-test: light target with dark text, -2 D model, 3 iterations -> output must not be black/NaN
function selfTest(E) {
  const N = E.N, c = document.createElement("canvas"); c.width = c.height = N; const x = c.getContext("2d");
  x.fillStyle = "#fff"; x.fillRect(0, 0, N, N); x.fillStyle = "#000"; x.font = `${N / 10}px sans-serif`; x.fillText("Teste ok", N / 8, N / 2);
  const P = { ...E.params }; Object.assign(E.params, { split: false, filter: true, retina: false });
  E.setTarget(c, false); E.setModel([{ S: -2, C: -0.5, axisScreen: 30, roll: 0, d: 0.4, pupil: 4, yaw: 0, pitch: 0, w: 1 }], 0.08);
  E.iterate(3); E.render(); const r = E.check(); Object.assign(E.params, P); E.hasTarget = false; E.iters = 0; return r;
}
function notice(msg) {
  let n = $("compat"); if (!n) { document.querySelector("main").insertAdjacentHTML("afterbegin", `<div class="warn" id="compat"></div>`); n = $("compat"); }
  n.innerHTML = msg;
}
function useCPU(reason) {
  if (main?.cpu && small?.cpu) return;
  console.warn("GPU path failed → CPU fallback:", reason);
  caps.engine = "cpu"; caps.fallback = reason; caps.hdr = false; caps.hdrMode = "sdr"; caps.hdrReason = "motor CPU (sem HDR)";
  main = new CPUEngine(document.createElement("canvas"), 512); small = new CPUEngine(document.createElement("canvas"), 256);
  for (const v of Object.values(views || {})) { v.E = main; v.attach(); v.modelKey = ""; v.contentKey = ""; v._pk = ""; }
  window.__vc.main = main; window.__vc.small = small;
  setTimeout(() => notice(`Modo de compatibilidade: a GPU deste aparelho falhou (${reason}). Usando cálculo no processador, com resolução menor.`), 0);
}
function cpuFailed(reason) {
  console.warn("CPU fallback failed too:", reason); caps.engine = "off";
  for (const E of [main, small]) if (E) E.params.filter = false;
  notice(`Não foi possível calcular o filtro neste aparelho (${reason}). Mostrando o texto original.`);
}
function verify(E) { // called after renders; swaps engine if output is bad
  if (!E || E._verified === E.iters + ":" + E.ready) return true;
  E._verified = E.iters + ":" + E.ready;
  const r = E.check(); if (r.ok) return true;
  if (E.cpu) cpuFailed(r.reason); else useCPU(r.reason);
  return false;
}
function makeEngines() {
  try {
    const c = document.createElement("canvas"); main = new Engine(c, +S.set.res);
    caps.hdr = false; caps.hdrReason = S.set.hdr ? "verificando…" : "desligado nos ajustes";
    const c2 = document.createElement("canvas"); small = new Engine(c2, 512, { preserve: true });
    if (/[?&]simnan=1/.test(location.search)) main.simNaN = small.simNaN = true;   // QA: simulate Apple-GPU NaN
    caps.webgl = true; caps.engine = "webgl2";
    for (const E of [small, main]) { const r = selfTest(E); if (!r.ok) throw new Error("autoteste da GPU: " + r.reason); }
  } catch (e) { caps.webgl = false; caps.err = String(e); window.__vc = { S, caps, tracker }; useCPU(String(e.message || e)); }
  caps.dynHigh = matchMedia("(dynamic-range: high)").matches;
  caps.p3 = matchMedia("(color-gamut: p3)").matches;
  caps.cam = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  window.__vc = { main, small, S, caps, tracker, get engine() { return main?.cpu ? "cpu" : "webgl2"; } };
}
makeEngines();
if (/[?&]forcecpu=1/.test(location.search)) useCPU("forçado por ?forcecpu=1 (teste)");
async function initHDR() {   // (1) EDR headroom: WebGPU extended canvas first, then WebGL2 extended canvas, else SDR [0,1]
  if (!main || main.cpu || !S.set.hdr) return;
  const r = await WebGPUPresenter.create(main.N);
  if (r.ok) { main.setPresenter(r.presenter); caps.hdr = true; caps.hdrMode = "webgpu"; caps.hdrReason = r.reason; }
  else if (main.enableHDR()) { caps.hdr = true; caps.hdrMode = "webgl-extended"; caps.hdrReason = r.reason + "; usando canvas WebGL estendido"; }
  else { caps.hdr = false; caps.hdrMode = "sdr"; caps.hdrReason = r.reason + "; canvas WebGL estendido também indisponível → SDR [0,1]"; }
  for (const v of Object.values(views)) { v.attach(); v.modelKey = ""; v.contentKey = ""; }
  showCaps();
}

function applyParams(E) {
  Object.assign(E.params, { K: S.set.K, b: S.set.b, rgb: S.set.rgb, method: S.set.method, hi: E.hdr ? S.set.edrH : 1.0, maxIters: S.set.iters });
}
// kernels: list of (eye x distance) PSF models with weights (joint least squares)
// capFontCss: cap the design defocus so the blur disc stays <= CAP_F x font height (text views);
// beyond that pre-compensation only makes halos (sim: ~chance at >=2 D). Chart passes Infinity.
const CAP_F = 0.6;
function kernels(rxByEye, weights, pupil, capFontCss = S.set.font || 17) {
  const d0 = distance(), t = tracker.state, ks = [];
  const ds = S.set.robust ? [[d0, 0.5], [d0 - 0.03, 0.25], [d0 + 0.03, 0.25]] : [[d0, 1]];
  const eyes = Object.keys(weights).filter(e => weights[e] > 0).sort((a, b) => weights[b] - weights[a]);
  for (const e of eyes) for (const [d, w] of ds) {
    const n = nearRx(rxByEye[e], d, S.set.accom);
    // (W5) conservative design: assume slightly LESS defocus than estimated (too much is worse than none)
    const M = n.S + n.C / 2, bias = S.set.bias || 0;
    n.S -= Math.sign(M) * Math.min(Math.abs(M), bias);
    const Mmax = Math.max(Math.abs(n.S), Math.abs(n.S + n.C)), cap = CAP_F * capFontCss * cssMM() / (pupil * d);
    if (Mmax > cap) { const f = cap / Mmax; n.S *= f; n.C *= f; }
    ks.push({ S: n.S, C: n.C, axisScreen: taboToScreen(n.A), d, pupil, yaw: t.ok ? t.yaw : 0, pitch: t.ok ? t.pitch : 0, roll: t.ok ? t.roll : 0, w: w * weights[e], eye: e });
  }
  return ks;
}
// contrast target: full contrast when there is (almost) nothing to correct, S.set.b from 0.75 D up
function bFor(ks) {
  const D = Math.max(...ks.map(k => Math.max(Math.abs(k.S), Math.abs(k.S + k.C))));
  return Math.round((1 - (1 - S.set.b) * Math.min(1, D / 0.75)) * 20) / 20;
}
function autoWeights() {
  const t = tracker.state;
  if (t.ok && t.rightOpen && !t.leftOpen) return { od: 1, os: 0 };
  if (t.ok && t.leftOpen && !t.rightOpen) return { od: 0, os: 1 };
  return { od: 0.5, os: 0.5 };   // both open: binocular compromise
}
// PSF recomputed only when vergence changes by ≥0.1 D (1/d quantised), angles ≥5°, etc.
const q = (k) => JSON.stringify(k.map(x => [x.S.toFixed(2), x.C.toFixed(2), Math.round(x.axisScreen + x.roll), Math.round(10 / x.d), x.pupil, Math.round(x.yaw / 5), Math.round(x.pitch / 5), x.w.toFixed(2)]));

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
  attach() { if (!this.host) return; const c = this.E.displayCanvas; if (c.parentNode !== this.host) { this.host.innerHTML = ""; this.host.appendChild(c); } }
  pixMM() { const w = this.E.displayCanvas.getBoundingClientRect().width || Math.min(innerWidth, 520); return cssMM() * w / this.E.N; }
  scale() { const w = this.E.displayCanvas.getBoundingClientRect().width || Math.min(innerWidth, 520); return this.E.N / w; }
  tick(spec, now) {
    const E = this.E; applyParams(E);
    const ks = spec.kernels(); E.params.b = bFor(ks);
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
    const pk = [E.params.b, E.params.K, E.params.rgb, E.params.hi].join();
    if (pk !== this._pk) { this._pk = pk; E.retarget(); this.dirty = true; }
    if (E.params.maxIters !== this._mi) { this._mi = E.params.maxIters; if (E.iters >= E.maxIters()) E.iters = Math.min(E.iters, E.params.maxIters); }
    const P = E.params, flags = [P.filter, P.split, P.retina, P.method].join();
    if (flags !== this._flags) { this._flags = flags; this.dirty = true; }
    if (P.method !== this._method) { this._method = P.method; E.method.init(E); this.dirty = true; }
    if (E.iters < E.maxIters()) { E.iterate(spec.itPerFrame || (E.N > 512 ? 2 : 4)); this.dirty = true; }
    if (this.dirty) { E.render(); this.dirty = false;
      if (E.iters >= E.maxIters() || E.iters === 0) verify(E); }
  }
}

// ---------------- screens ----------------
let screen = "home";
var views = {};
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
    S.set.iters = +$("sI").value; S.set.edrH = +$("sH").value; $("vI").textContent = S.set.iters + " iterações"; $("vH").textContent = S.set.edrH.toFixed(2) + "× o branco";
    $("vK").textContent = S.set.K.toFixed(3); $("vB").textContent = S.set.b.toFixed(2); $("vP").textContent = S.set.pupil.toFixed(2) + " mm"; $("vF").textContent = S.set.font + " pt";
    if (main) Object.assign(main.params, { filter: $("tFilter").checked, split: $("tSplit").checked, retina: $("tRetina").checked });
    save();
  };
  $("method").innerHTML = METHODS.map(m => `<option value="${m.id}">${m.label}</option>`).join("");
  $("sI").value = S.set.iters; $("sH").value = S.set.edrH;
  $("sK").value = Math.log10(S.set.K); $("sB").value = S.set.b; $("sP").value = S.set.pupil; $("sF").value = S.set.font; $("method").value = S.set.method;
  for (const id of ["sK", "sB", "sP", "sF", "sI", "sH", "method", "tFilter", "tSplit", "tRetina"]) $(id).addEventListener("input", sync);
  $("photo").onchange = async (e) => { const f = e.target.files[0]; if (!f) return; views.read.photo = await createImageBitmap(f); views.read.contentKey = ""; };
  sync();
}
bindRead();

// ----- Chart -----
let rows = chartRows(7);
const chartSpec = {
  kernels: () => kernels(S.rx, autoWeights(), S.set.pupil, Infinity),
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
// dispW = CSS width at which the result canvas will be shown (sets pixel pitch and text scale)
function renderSmall(rxByEye, eye, drawFn, iters = 40, dispW = Math.min(innerWidth, 520) * 0.48) {
  const E = small; applyParams(E);
  const pix = cssMM() * dispW / E.N, sc = E.N / dispW;
  const ks = kernels(rxByEye, { [eye]: 1 }, S.set.pupil); E.params.b = bFor(ks);
  const c = document.createElement("canvas"); c.width = c.height = E.N; drawFn(c.getContext("2d"), E.N, sc, pix, ks[0]);
  E.params.split = false; E.params.filter = caps.engine !== "off"; E.params.retina = false;
  E.setTarget(c, false); E.setModel(ks, pix); E.retarget(); E.iterate(Math.min(iters, E.maxIters())); E.render();
  E._verified = ""; if (!verify(E) && small !== E) return renderSmall(rxByEye, eye, drawFn, iters, dispW);
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
    const cands = this.cands(), token = (this._tok = (this._tok || 0) + 1);
    const one = (i) => {
      if (token !== this._tok || i >= cands.length) return;
      if (i === 0) box.innerHTML = "";
      const cv = renderSmall({ [e]: cands[i] }, e, (ctx, N, sc, pix, k0) => drawSentence(ctx, N, SENTENCE, 15, sc, predistortFor(k0, 15 * sc, pix), true), 25);
      cv.onclick = () => this.pick(i); cv.setAttribute("aria-label", `versão ${i + 1}`); box.appendChild(cv);
      setTimeout(() => one(i + 1), 16);   // yield to the UI between variants
    };
    setTimeout(() => one(0), 30);
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
        <p class="small">A idade estima quanto o olho ainda foca de perto (acomodação). Sem idade, assumimos olho jovem (foca bem de perto). Acima de ~45 anos informe a idade.</p>
        <button class="btn primary" id="dGo">Estou a 40 cm</button>`;
      if (S.set.ageSet && S.set.age) $("age").value = String(S.set.age);
      $("age").onchange = () => { const a = +$("age").value;   // no age -> normal (young) accommodation 3 D
        if (a) Object.assign(S.set, { accom: Math.max(0, 0.5 * (15 - 0.25 * a)), ageSet: true, age: a }); else Object.assign(S.set, { accom: 3, ageSet: false, age: 0 }); save(); };
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
  preview(grid = false) {   // phone-size text (17 px CSS) filling a cropped box; shown at the real width
    const pv = $("pv"); if (!pv) return; pv.className = "pvbox" + (grid ? " tall" : "");
    const dispW = pv.clientWidth || Math.min(innerWidth, 520) - 24, f = S.set.font || 17;
    const e = this.eye, cv = renderSmall(this.work, e, (ctx, N, sc, pix, k0) => {
      const h = drawSentence(ctx, N, DISC_TEXT, f, sc, predistortFor(k0, f * sc, pix), true);
      if (grid) { ctx.fillStyle = "#000"; const st = Math.round(f * sc * 0.9), lw = Math.max(1, Math.round(f * sc / 12)), g0 = Math.round(h + f * sc * 0.6), g1 = Math.round(N * 0.74);
        for (let x = Math.round(N * 0.05); x < N * 0.95; x += st) ctx.fillRect(x, g0, lw, g1 - g0); for (let y = g0; y < g1; y += st) ctx.fillRect(Math.round(N * 0.05), y, Math.round(N * 0.9), lw); }
    }, 40, dispW);
    cv.style.width = "100%"; pv.innerHTML = ""; pv.appendChild(cv);
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
    if (+$("sAcc").value !== S.set.accom) S.set.ageSet = true; S.set.accom = +$("sAcc").value; $("vAcc").textContent = S.set.accom.toFixed(2) + " D";
    S.set.bias = +$("sBias").value; $("vBias").textContent = "−" + S.set.bias.toFixed(2) + " D";
    S.set.robust = $("fRobust").checked; S.set.rgb = $("fRGB").checked; S.set.pre = $("fPre").checked;
    const hdrWas = S.set.hdr, resWas = S.set.res; S.set.hdr = $("fHDR").checked; S.set.res = +$("res").value;
    save(); updateBadge();
    if (hdrWas !== S.set.hdr || resWas !== S.set.res) location.reload();
  };
  $("sCard").value = S.set.cardPx; $("sDist").value = S.set.manualCm; $("sAcc").value = S.set.accom; $("sBias").value = S.set.bias;
  $("fRobust").checked = S.set.robust; $("fRGB").checked = S.set.rgb; $("fHDR").checked = S.set.hdr; $("fPre").checked = S.set.pre; $("res").value = S.set.res;
  for (const id of ["sCard", "sDist", "sAcc", "sBias", "fRobust", "fRGB", "fHDR", "fPre", "res"]) $(id).addEventListener("change", sync);
  $("sCard").addEventListener("input", () => { $("cardBar").style.width = $("sCard").value + "px"; });
  $("calib40").onclick = () => { if (!tracker.state.ok) { alert("Ative a câmera primeiro (tela inicial)."); return; }
    S.set.calib = S.set.calib * 0.40 / tracker.state.d; tracker.calib = S.set.calib; tracker._dHist = []; save(); alert("Calibrado."); };
  $("reset").onclick = () => { if (confirm("Apagar grau e ajustes salvos neste aparelho?")) { localStorage.removeItem(KEY); location.reload(); } };
  sync();
}
function showCaps() {
  $("caps").innerHTML = `Motor: <b>${main?.cpu ? "CPU (compatibilidade)" : "GPU WebGL2"}</b>${caps.fallback ? " — " + caps.fallback : ""} · WebGL2 float: <b>${caps.webgl ? "sim" : "não"}</b>${caps.err ? " (" + caps.err + ")" : ""} · HDR/EDR: <b>${caps.hdr ? `ATIVO (${caps.hdrMode}, limite ${S.set.edrH}× o branco)` : "inativo — usando [0,1]"}</b> (${caps.hdrReason || ""})
   · tela HDR (CSS): ${caps.dynHigh ? "sim" : "não"} · P3: ${caps.p3 ? "sim" : "não"} · câmera: ${caps.cam ? "sim" : "não"} · mm por px CSS: ${cssMM().toFixed(4)}`;
}
bindSettings();

// ---------------- main loop ----------------
function loop(now) {
  try {
    if (main && screen === "read") { views.read.tick(readSpec, now); const k = views.read.info; if (k) { const b = blurInfo({ S: k.S, C: k.C, A: 0 }, k.pupil, k.d, views.read.pixMM());
      $("readTips").innerHTML = readTips();
      $("readInfo").innerHTML = `Borrão residual (projeto, já com viés −${S.set.bias.toFixed(2)} D) ≈ ${b.D.toFixed(2)} D → ${b.arcmin.toFixed(0)}′ (${b.px.toFixed(0)} px). ${b.D > 2 ? "<b>Acima de ~2 D a tela não recupera letras pequenas — aumente a fonte/brilho.</b>" : b.D < 0.25 ? "Quase sem borrão nesta distância." : "Faixa onde a pré-compensação ajuda (~1 linha)."} Iterações: ${main.iters}${views.read.photo ? " · <a href='#' id='backText'>voltar ao texto</a>" : ""}`;
      const bt = $("backText"); if (bt) bt.onclick = (e) => { e.preventDefault(); views.read.photo = null; views.read.contentKey = ""; }; } }
    if (main && screen === "chart") views.chart.tick(chartSpec, now);
  } catch (e) { console.error(e); $("readInfo").textContent = "Erro: " + e; }
  requestAnimationFrame(loop);
}

function readTips() {
  const t = tracker.state, tips = ["☀️ <b>Aumente o brilho da tela ao máximo</b> (pupila menor = imagem mais nítida; o site não consegue fazer isso por você)."];
  if (main?.params.filter) tips.push("Use <b>fundo claro com texto escuro</b> — o filtro funciona melhor assim" + (caps.hdr ? " (e é o que aproveita a folga HDR)." : "."));
  if (S.set.manualCm > 0) tips.push(`Distância fixa em ${S.set.manualCm} cm — mantenha o celular nessa distância (régua).`);
  else if (t.ok && S.set.calib === 1) tips.push("📏 <b>Calibre a distância</b>: em Ajustes, segure a 40 cm (régua) e toque “Estou exatamente a 40 cm”. Erro de distância piora o resultado.");
  else if (!t.ok) tips.push("Sem câmera: assumindo 30 cm. Defina a distância em Ajustes ou ative a câmera.");
  return tips.map(x => `<li>${x}</li>`).join("");
}
route(); updateBadge(); requestAnimationFrame(loop); initHDR();
if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
