import { Engine } from "./gpu.js";
import { METHODS, methodById } from "./methods.js";
import { WebGPUPresenter } from "./hdr.js";
import { CPUEngine } from "./cpu.js";
import { Tracker } from "./tracker.js";
import { nearRx, taboToScreen, blurInfo } from "./optics.js";
import { drawReading, drawChart, chartRows, PARAGRAPH, drawTest, layoutUse, drawUse, hitUse } from "./content.js";

// ---------------- persisted state ----------------
const KEY = "visaoclara.v1";
const DEF = { rx: { od: { S: 0, C: 0, A: 0 }, os: { S: 0, C: 0, A: 0 } },
  set: { cardPx: 325, manualCm: 0, calib: 1, accom: 5, robust: false, rgb: false, hdr: true, pre: false, res: 1024,
         pupil: 4, K: 0.01, b: 0.6, font: 17, method: "fista", iters: 30, edrH: 2, bias: 0.10, ageSet: false, age: 0, useAdj: 0 }, chartLog: [], v: 5 };
const D0 = 0.22;                      // default viewing distance (m): "um palmo" — camera overrides
// available accommodation from age (Hofstetter mean amplitude, ~60% sustainable), unknown age = young (5 D)
function accomFor(a) { return a ? Math.max(0, Math.min(6, 0.6 * (18.5 - 0.3 * a))) : 5; }
let S = JSON.parse(JSON.stringify(DEF));
try { const j = JSON.parse(localStorage.getItem(KEY)); if (j) { const old = (j.v || 1) < 3, old4 = (j.v || 1) < 4; S = { ...S, ...j, set: { ...S.set, ...j.set }, rx: { ...S.rx, ...j.rx }, v: 5 };
  if (old) Object.assign(S.set, { robust: false, iters: 30, edrH: 2, bias: 0.10, hdr: true });
  if ((j.v || 1) < 5) { S.set.accom = accomFor(S.set.ageSet ? S.set.age : 0); if (S.set.manualCm === 40) S.set.manualCm = 0; } } } catch (e) {}
const save = () => localStorage.setItem(KEY, JSON.stringify(S));
const $ = (id) => document.getElementById(id);
const cssMM = () => 53.98 / S.set.cardPx;     // card SHORT side (fits a phone in portrait)

// ---------------- tracking ----------------
const tracker = new Tracker(); tracker.calib = S.set.calib;
let rawD = null;
tracker.onUpdate = (s) => { rawD = s.d; updateBadge(); };
const distance = () => S.set.manualCm > 0 ? S.set.manualCm / 100 : (tracker.state.ok ? tracker.state.d : (rawD && !tracker.state.err ? tracker.state.d : D0));
function updateBadge() {
  const t = tracker.state, d = distance();
  $("distBadge").textContent = `${(d * 100).toFixed(0)} cm${S.set.manualCm > 0 ? " fixo" : t.ok ? " 📷" : ""}`;
}
$("camBtn").onclick = async () => {
  $("camInfo").textContent = "Carregando detector de rosto (no aparelho)…";
  const ok = await tracker.start($("cam"));
  $("camInfo").textContent = ok ? "Câmera ativa: distância e posição dos olhos medidas no aparelho." : "Câmera indisponível — usando 22 cm (um palmo). Ajuste em Ajustes. " + (tracker.state.err || "");
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
  if (main?.cpu) return;
  console.warn("GPU path failed → CPU fallback:", reason);
  caps.engine = "cpu"; caps.fallback = reason; caps.hdr = false; caps.hdrMode = "sdr"; caps.hdrReason = "motor CPU (sem HDR)";
  main = new CPUEngine(document.createElement("canvas"), 512);
  for (const v of Object.values(views || {})) { v.E = main; v.attach(); v.modelKey = ""; v.contentKey = ""; v._pk = ""; }
  window.__vc.main = main;
  setTimeout(() => notice(`Modo de compatibilidade: a GPU deste aparelho falhou (${reason}). Usando cálculo no processador, com resolução menor.`), 0);
}
function cpuFailed(reason) {
  console.warn("CPU fallback failed too:", reason); caps.engine = "off";
  if (main) main.params.filter = false;
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
    if (/[?&]simnan=1/.test(location.search)) main.simNaN = true;   // QA: simulate Apple-GPU NaN
    caps.webgl = true; caps.engine = "webgl2";
    for (const E of [main]) { const r = selfTest(E); if (!r.ok) throw new Error("autoteste da GPU: " + r.reason); }
  } catch (e) { caps.webgl = false; caps.err = String(e); window.__vc = { S, caps, tracker }; useCPU(String(e.message || e)); }
  caps.dynHigh = matchMedia("(dynamic-range: high)").matches;
  caps.p3 = matchMedia("(color-gamut: p3)").matches;
  caps.cam = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  window.__vc = { main, S, caps, tracker, get engine() { return main?.cpu ? "cpu" : "webgl2"; } };
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
  if (screen === "result") showResult();
  if (screen === "use" && main) use.enter();
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
$("rxApply").onclick = () => { readRx(); location.hash = "#use"; };
$("rxTune").onclick = () => { readRx(); disc.fineOnly(); };

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

// ----- big − / + stepper (no sliders in the test flow) -----
const fA = (a) => `${Math.round(a)}°`;
function stepper(host, { label, get, set, step, min, max, fmt, wrap = false }) {
  host.className = "stepper";
  host.innerHTML = `<button class="stp" data-d="-1" aria-label="${label}: menos">−</button><div class="val"><small>${label}</small><b></b></div><button class="stp" data-d="1" aria-label="${label}: mais">+</button>`;
  const show = () => { host.querySelector("b").textContent = fmt(get()); };
  host.querySelectorAll(".stp").forEach(b => b.onclick = () => {
    let v = get() + (+b.dataset.d) * step;
    v = wrap ? ((v % max) + max) % max : Math.max(min, Math.min(max, Math.round(v / step) * step));
    set(+v.toFixed(2)); show();
  });
  show();
}

const TBOX_CROP = 0.56;   // the test box shows the top 56% of the square (where the letters are)
// ----- TEST: small letters at true angular size, live pre-compensation on the main engine -----
const testSpec = {
  kernels: () => kernels(disc.work, { [disc.eye]: 1 }, S.set.pupil),
  contentKey: () => [disc.mode, Math.round(distance() * 100), S.set.cardPx, innerWidth].join(),
  draw: (ctx, N, scale, pixMM) => drawTest(ctx, N, { dM: distance(), pixMM, scale, cols: disc.mode === "cmp" ? 2 : 1, crop: TBOX_CROP - 0.03 }),
};
const AGES = [["Até 39 anos", 30], ["40 a 44", 42], ["45 a 49", 47], ["50 a 54", 52], ["55 a 59", 57], ["60 ou mais", 65], ["Não sei", 0]];
const disc = {
  mode: "on",
  start() {
    if (this._keep) { this._keep = false; this.render(); return; }
    this.only = false; this.eye = "od"; this.step = "intro"; this.mode = "on";
    this.work = { od: { S: 0, C: 0, A: 0 }, os: { S: 0, C: 0, A: 0 } }; this.render();
  },
  fineOnly() {   // "Já sei meu grau" -> fine adjust with − / + starting from the entered Rx
    this.only = true; this.eye = "od"; this.step = "cover"; this.mode = "on"; this.work = JSON.parse(JSON.stringify(S.rx));
    if (screen === "discover") this.render(); else { this._keep = true; location.hash = "#discover"; }
  },
  hasBox() { return !!$("tbox") && !!views.test; },
  go(step) { this.step = step; this.render(); window.scrollTo(0, 0); },
  setMode(m) {
    this.mode = m; if (main) Object.assign(main.params, { filter: m !== "off", split: m === "cmp", retina: false });
    document.querySelectorAll("#seg .btn").forEach(b => b.classList.toggle("sel", b.dataset.m === m));
  },
  render() {
    const box = $("discStep"), e = this.eye, w = this.work[e];
    const EYE = e === "od" ? "DIREITO" : "ESQUERDO", other = e === "od" ? "esquerdo" : "direito";
    if (this.step === "intro") {
      box.innerHTML = `<h2>Teste sem óculos</h2><div class="warn">Estimativa — não substitui exame oftalmológico.</div>
        <p><b>Tire os óculos.</b> Segure o celular a <b>um palmo do rosto (~22 cm)</b>. Coloque o brilho da tela no máximo.</p>
        <p class="big" id="dLive"></p>
        ${tracker.state.ok ? "" : `<button class="btn" id="dCam">📷 Medir distância com a câmera</button>`}
        <h3>Sua idade</h3><p class="note">A idade indica quanto o olho ainda foca de perto.</p>
        <div class="agegrid">${AGES.map(([l, a]) => `<button class="btn age${(S.set.ageSet ? S.set.age : 0) === a ? " sel" : ""}" data-age="${a}">${l}</button>`).join("")}</div>
        <button class="btn primary" id="dGo">Começar o teste</button>`;
      box.querySelectorAll(".age").forEach(b => b.onclick = () => { const a = +b.dataset.age;
        Object.assign(S.set, { age: a, ageSet: a > 0, accom: accomFor(a) }); save(); box.querySelectorAll(".age").forEach(x => x.classList.toggle("sel", x === b)); });
      const cam = $("dCam"); if (cam) cam.onclick = async () => { cam.textContent = "Carregando…"; await tracker.start($("cam")); updateBadge(); if (this.step === "intro") this.render(); };
      const live = () => { const el = $("dLive"); if (!el || this.step !== "intro") return; const d = distance(), ok = Math.abs(d - D0) <= 0.04;
        el.innerHTML = tracker.state.ok ? `<span class="${ok ? "ok" : "bad"}">${(d * 100).toFixed(0)} cm ${ok ? "✓" : d > D0 ? "— aproxime" : "— afaste"}</span>` : `Distância: ${(d * 100).toFixed(0)} cm`;
        if (screen === "discover") setTimeout(live, 250); };
      live();
      $("dGo").onclick = () => this.go("cover");
    } else if (this.step === "cover") {
      box.innerHTML = `<p class="eye">Olho ${EYE}</p><p class="big">Cubra o olho ${other} com a mão.</p><p class="note" id="occ"></p><button class="btn primary" id="dGo">Pronto</button>`;
      const chk = () => { const el = $("occ"); if (!el || this.step !== "cover") return; const t = tracker.state, closed = e === "od" ? !t.leftOpen : !t.rightOpen;
        el.textContent = t.ok ? (closed ? "✓ olho coberto" : "Se cobriu com a mão, tudo bem.") : ""; if (screen === "discover") setTimeout(chk, 300); };
      chk(); $("dGo").onclick = () => this.go(this.only ? "fine" : "sph");
    } else if (this.step === "fan") {
      if (this.fanA === undefined) this.fanA = 90;
      box.innerHTML = `<p class="eye">Olho ${EYE}</p><p>Alguma linha parece <b>mais escura</b>? Use <b>−</b> / <b>+</b> para marcar essa linha.</p>
        <div id="fanBox"></div><div id="stA"></div>
        <button class="btn primary" id="dGo">Esta linha é a mais escura</button><button class="btn" id="dSame">Todas iguais</button>`;
      const drawFan = () => { let svg = `<svg viewBox="-115 -112 230 122" class="fan" role="img" aria-label="leque de linhas">`;
        for (let a = 0; a < 180; a += 10) { const r = a * Math.PI / 180, x = Math.cos(r), y = -Math.sin(r), on = a === this.fanA;
          svg += `<line x1="${x * 20}" y1="${y * 20}" x2="${x * 92}" y2="${y * 92}" stroke="#000" stroke-width="3.2"/>`;
          if (on) svg += `<circle cx="${x * 102}" cy="${y * 102}" r="8" fill="none" stroke="#d00" stroke-width="3"/>`; }
        $("fanBox").innerHTML = svg + "</svg>"; };
      drawFan();
      stepper($("stA"), { label: "Linha marcada", get: () => this.fanA, set: (v) => { this.fanA = v; drawFan(); }, step: 10, min: 0, max: 180, wrap: true, fmt: fA });
      $("dGo").onclick = () => { const axisScreen = (this.fanA + 90) % 180; w.A = (180 - axisScreen) % 180; if (!w.C) w.C = -0.75; this.go("cyl"); };
      $("dSame").onclick = () => { w.C = 0; this.go("fine"); };
    } else {   // sph | cyl | fine : live test box
      const ins = { sph: "Toque <b>−</b> ou <b>+</b> até as letras pequenas ficarem <b>mais nítidas</b>.",
        cyl: "Agora ajuste o <b>cilíndrico</b> até as letras ficarem mais nítidas.", fine: "Ajuste fino: eixo, cilíndrico e esférico." }[this.step];
      box.innerHTML = `<p class="eye">Olho ${EYE}</p><p>${ins}</p><div class="tbox" id="tbox"></div>
        ${this.step === "fine" ? `<div id="stA"></div><div id="stC"></div>` : ""}<div id="${this.step === "cyl" ? "stC" : "stS"}"></div>${this.step === "fine" ? `<div id="stS"></div>` : ""}
        <button class="btn primary" id="dGo">${this.step === "fine" ? "Pronto" : "Esta é a mais nítida"}</button>
        <div class="seg" id="seg"><button class="btn" data-m="off">Sem filtro</button><button class="btn" data-m="on">Com filtro</button><button class="btn" data-m="cmp">Comparar</button></div>`;
      if ($("stS")) stepper($("stS"), { label: "Esférico", get: () => w.S, set: (v) => (w.S = v), step: 0.25, min: -8, max: 2, fmt: (v) => fD(v) + " D" });
      if ($("stC")) stepper($("stC"), { label: "Cilíndrico", get: () => w.C, set: (v) => (w.C = v), step: 0.25, min: -4, max: 0, fmt: (v) => fD(v) + " D" });
      if ($("stA")) stepper($("stA"), { label: "Eixo", get: () => w.A, set: (v) => (w.A = v), step: 5, min: 0, max: 180, wrap: true, fmt: fA });
      document.querySelectorAll("#seg .btn").forEach(b => b.onclick = () => this.setMode(b.dataset.m));
      const v = views.test || (views.test = new View(main, null)); v.E = main; v.host = $("tbox"); v.attach(); v.contentKey = ""; v.modelKey = "";
      this.setMode(this.mode);
      $("dGo").onclick = () => {
        if (this.step === "sph") return this.go("fan");
        if (this.step === "cyl") return this.go("fine");
        S.rx[e] = { ...w }; save();
        if (e === "od") { this.eye = "os"; this.fanA = undefined; this.go("cover"); } else location.hash = "#result";
      };
    }
  },
};

// ----- RESULT -----
function showResult() {
  const r = S.rx, card = (e, name) => `<div class="rcard"><h3>${name}</h3><table class="rtable">
    <tr><th>Esférico</th><td class="v">${fD(r[e].S)}</td></tr><tr><th>Cilíndrico</th><td class="v">${fD(r[e].C)}</td></tr>
    <tr><th>Eixo</th><td class="v">${r[e].C ? Math.round(r[e].A) + "°" : "—"}</td></tr></table></div>`;
  $("resultBox").innerHTML = card("od", "Olho direito (OD)") + card("os", "Olho esquerdo (OE)") +
    `<p class="note">Medido a ${(distance() * 100).toFixed(0)} cm · idade: ${S.set.ageSet ? "~" + S.set.age + " anos" : "não informada (olho jovem)"}.</p>`;
}

// ----- USE mode: interactive feed rendered pre-compensated live (camera keeps tracking distance) -----
const use = {
  st: { likes: 3, more: 0, chat: [{ s: "Oi! Você chega a que horas amanhã?", me: false }, { s: "Lá pelas 18h30, depois do trabalho.", me: true }, { s: "Combinado. Não esqueça a receita do médico!", me: false }] },
  y: 0, ver: 0, filter: true,
  W() { return $("useStage").clientWidth || Math.min(innerWidth, 560); },
  layout() { const c = this._mc || (this._mc = document.createElement("canvas").getContext("2d")); this.L = layoutUse(c, this.W(), this.st); this.clamp(); },
  clamp() { this.y = Math.max(0, Math.min(this.y, Math.max(0, this.L.H - this.W()))); },
  scroll(dy) { this.y += dy; this.clamp(); },
  enter() {
    const v = views.use || (views.use = new View(main, null)); v.E = main; v.host = $("useStage"); v.attach(); v.contentKey = ""; v.modelKey = "";
    this.layout(); Object.assign(main.params, { filter: this.filter, split: false, retina: false }); this.paintToggle();
    stepper($("useStep"), { label: "Ajuste fino (esférico)", get: () => S.set.useAdj || 0, set: (x) => { S.set.useAdj = x; save(); }, step: 0.25, min: -2, max: 2, fmt: (x) => (x ? fD(x) : "0,00") + " D" });
    this._info = "";
  },
  paintToggle() { const b = $("useFilter"); b.textContent = this.filter ? "Filtro LIGADO" : "Filtro DESLIGADO"; b.classList.toggle("primary", this.filter); },
  tap(x, y) {
    const it = hitUse(this.L, x, y + this.y); if (!it) return;
    if (it.id === "like") this.st.likes++; else if (it.id === "more") this.st.more++; else if (it.id === "reply") this.st.chat.push({ s: "Ok, combinado! 👍", me: true });
    this.ver++; this.layout(); if (it.id === "reply") this.y = this.L.H;   // jump to the new message
    this.clamp();
  },
  info() {
    const t = `${(distance() * 100).toFixed(0)} cm · ${tracker.state.ok ? "câmera acompanhando" : "sem câmera (22 cm)"} · arraste o texto para rolar`;
    if (t !== this._info) { this._info = t; $("useInfo").textContent = t; }
  },
};
const useSpec = {
  kernels: () => { const a = S.set.useAdj || 0; return kernels({ od: { ...S.rx.od, S: S.rx.od.S + a }, os: { ...S.rx.os, S: S.rx.os.S + a } }, autoWeights(), S.set.pupil); },
  contentKey: () => [Math.round(use.y), use.ver, use.W()].join(),
  draw: (ctx, N) => drawUse(ctx, N, use.W(), use.y, use.L),
};
(function bindUse() {
  const st = $("useStage"); let y0 = null, moved = 0, last = 0;
  st.addEventListener("pointerdown", (e) => { y0 = e.clientY; last = e.clientY; moved = 0; st.setPointerCapture?.(e.pointerId); });
  st.addEventListener("pointermove", (e) => { if (y0 === null) return; const dy = e.clientY - last; last = e.clientY; moved += Math.abs(dy); use.scroll(-dy); });
  st.addEventListener("pointerup", (e) => { if (y0 !== null && moved < 10) { const r = st.getBoundingClientRect(); use.tap(e.clientX - r.left, e.clientY - r.top); } y0 = null; });
  st.addEventListener("pointercancel", () => { y0 = null; });
  st.addEventListener("wheel", (e) => { e.preventDefault(); use.scroll(e.deltaY); }, { passive: false });
  $("useUp").onclick = () => use.scroll(-use.W() * 0.6); $("useDown").onclick = () => use.scroll(use.W() * 0.6);
  $("useFilter").onclick = () => { use.filter = !use.filter; if (main) main.params.filter = use.filter; use.paintToggle(); };
})();

// ----- Settings -----
function bindSettings() {
  const sync = () => {
    S.set.cardPx = +$("sCard").value; $("cardBar").style.width = S.set.cardPx + "px";
    S.set.manualCm = +$("sDist").value; $("vDist").textContent = S.set.manualCm ? S.set.manualCm + " cm" : "automática (câmera ou 22 cm)";
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
    S.set.calib = S.set.calib * D0 / tracker.state.d; tracker.calib = S.set.calib; tracker._dHist = []; save(); alert("Calibrado."); };
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
    if (main && screen === "discover" && disc.hasBox()) views.test.tick(testSpec, now);
    if (main && screen === "use" && views.use) { views.use.tick(useSpec, now); use.info(); }
  } catch (e) { console.error(e); }
  requestAnimationFrame(loop);
}

function readTips() {
  const t = tracker.state, tips = ["☀️ <b>Aumente o brilho da tela ao máximo</b> (pupila menor = imagem mais nítida; o site não consegue fazer isso por você)."];
  if (main?.params.filter) tips.push("Use <b>fundo claro com texto escuro</b> — o filtro funciona melhor assim" + (caps.hdr ? " (e é o que aproveita a folga HDR)." : "."));
  if (S.set.manualCm > 0) tips.push(`Distância fixa em ${S.set.manualCm} cm — mantenha o celular nessa distância (régua).`);
  else if (t.ok && S.set.calib === 1) tips.push("📏 <b>Calibre a distância</b>: em Ajustes, segure a 22 cm (um palmo, régua) e toque “Calibrar distância do rosto”. Erro de distância piora o resultado.");
  else if (!t.ok) tips.push("Sem câmera: assumindo 22 cm (um palmo). Mantenha essa distância ou ative a câmera.");
  return tips.map(x => `<li>${x}</li>`).join("");
}
Object.assign(window.__vc, { use, disc });
route(); updateBadge(); requestAnimationFrame(loop); initHDR();
if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
