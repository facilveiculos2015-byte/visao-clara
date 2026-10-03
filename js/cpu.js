// CPU (pure JS) fallback engine, same interface as the WebGL2 Engine.
// Used when WebGL2/float is missing or the GPU output fails the sanity check (NaN / black / context lost).
// Model: geometric-optics defocus+astigmatism blur = elliptical pillbox (analytic OTF 2J1(2πρ)/(2πρ)),
// times the pixel box (sinc). Luminance only; joint kernels A=Σw H², B=Σw H (H real). Wiener init + FISTA in [0,1].
import { methodById } from "./methods.js";

function j1(x) { // Numerical Recipes bessj1
  const ax = Math.abs(x);
  if (ax < 8) { const y = x * x;
    const a = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
    const b = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y))));
    return a / b; }
  const z = 8 / ax, y = z * z, xx = ax - 2.356194491;
  const a = 1 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
  const b = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
  const r = Math.sqrt(0.636619772 / ax) * (Math.cos(xx) * a - z * Math.sin(xx) * b);
  return x < 0 ? -r : r;
}
const jinc = (r) => r < 1e-6 ? 1 : 2 * j1(2 * Math.PI * r) / (2 * Math.PI * r);
const sinc = (x) => x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);

// in-place 2D FFT on (re, im) Float32Array of n*n
function fft1(re, im, n, off, stride, inv, rev, cs, sn) {
  for (let i = 0; i < n; i++) { const j = rev[i]; if (j > i) { const a = off + i * stride, b = off + j * stride;
    let t = re[a]; re[a] = re[b]; re[b] = t; t = im[a]; im[a] = im[b]; im[b] = t; } }
  for (let size = 2; size <= n; size <<= 1) { const h = size >> 1, step = n / size;
    for (let i = 0; i < n; i += size) for (let k = 0; k < h; k++) {
      const wr = cs[k * step], wi = inv ? sn[k * step] : -sn[k * step];
      const a = off + (i + k) * stride, b = off + (i + k + h) * stride;
      const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
      re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi; } }
}
function makeFFT(n) {
  const L = Math.log2(n), rev = new Uint32Array(n), cs = new Float32Array(n), sn = new Float32Array(n);
  for (let i = 0; i < n; i++) { let r = 0; for (let b = 0; b < L; b++) r |= ((i >> b) & 1) << (L - 1 - b); rev[i] = r;
    cs[i] = Math.cos(2 * Math.PI * i / n); sn[i] = Math.sin(2 * Math.PI * i / n); }
  return (re, im, inv) => {
    for (let y = 0; y < n; y++) fft1(re, im, n, y * n, 1, inv, rev, cs, sn);
    for (let x = 0; x < n; x++) fft1(re, im, n, x, n, inv, rev, cs, sn);
    if (inv) { const s = 1 / (n * n); for (let i = 0; i < n * n; i++) { re[i] *= s; im[i] *= s; } }
  };
}

export class CPUEngine {
  constructor(canvas, N = 512) {
    this.cpu = true; this.canvas = canvas; this.N = N; canvas.width = canvas.height = N;
    this.ctx = canvas.getContext("2d"); this.fft = makeFFT(N); const n2 = N * N;
    const F = () => new Float32Array(n2);
    Object.assign(this, { T: F(), R: F(), G: F(), B: F(), A: F(), Bk: F(), Hr: F(), TspR: F(), TspI: F(), X: F(), Xo: F(), Y: F(), wr: F(), wi: F() });
    this.hdr = false; this.ready = false; this.hasTarget = false; this.iters = 0; this.fistaT = 1; this.lost = false;
    this.params = { b: 0.6, mid: 0.5, lo: 0.1, maxIters: 30, K: 0.01, hi: 1.0, rgb: false, method: "fista", split: true, filter: true, retina: false, photo: false };
    this.img = this.ctx.createImageData(N, N);
  }
  get displayCanvas() { return this.canvas; }
  get method() { return methodById(this.params.method); }
  maxIters() { const m = this.method.maxIters; return typeof m === "function" ? m(this) : m; }
  enableHDR() { return false; }
  setModel(kernels, pixMM) {
    const N = this.N, A = this.A, Bk = this.Bk; A.fill(0); Bk.fill(0);
    const wsum = kernels.reduce((a, k) => a + k.w, 0);
    kernels.forEach((K, idx) => {
      const pr = pixMM / (K.d * 1000);                           // rad per pixel
      const sx = pr * Math.cos(K.yaw * Math.PI / 180), sy = pr * Math.cos(K.pitch * Math.PI / 180);
      const p = K.pupil * 1e-3, P1 = K.S, P2 = K.S + K.C;          // power along axis / perpendicular
      const r1 = p * Math.abs(P1) / 2, r2 = p * Math.abs(P2) / 2;   // blur-disc semi-axes (rad)
      const a = (K.axisScreen + K.roll) * Math.PI / 180, ca = Math.cos(a), sa = Math.sin(a), w = K.w / wsum;
      for (let y = 0; y < N; y++) { const fy0 = (y < N / 2 ? y : y - N) / N;
        for (let x = 0; x < N; x++) { const fx0 = (x < N / 2 ? x : x - N) / N;
          const fx = fx0 / sx, fy = -fy0 / sy;                       // cycles/rad, y up
          const fa = fx * ca + fy * sa, fb = -fx * sa + fy * ca;
          const H = jinc(Math.hypot(r1 * fa, r2 * fb)) * sinc(fx0) * sinc(fy0);
          const i = y * N + x; A[i] += w * H * H; Bk[i] += w * H; if (idx === 0) this.Hr[i] = H; } }
    });
    this.ready = true; this.iters = 0; this.fistaT = 1; this.modelInfo = { os: 0, S: 0, cpu: true };
    if (this.hasTarget) this.method.init(this);
  }
  setTarget(c2d, photo = false) {
    const N = this.N, d = c2d.getContext("2d").getImageData(0, 0, N, N).data;
    const lin = (v) => { v /= 255; return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    const lut = new Float32Array(256); for (let i = 0; i < 256; i++) lut[i] = lin(i);
    for (let i = 0; i < N * N; i++) { const r = lut[d[4 * i]], g = lut[d[4 * i + 1]], b = lut[d[4 * i + 2]];
      this.R[i] = r; this.G[i] = g; this.B[i] = b; this.T[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b; }
    this.params.photo = photo; this.hasTarget = true; this.retarget();
  }
  retarget() {
    if (!this.hasTarget) return; const P = this.params, n2 = this.N * this.N;
    for (let i = 0; i < n2; i++) { this.TspR[i] = P.mid + P.b * (this.T[i] - P.mid); this.TspI[i] = 0; }
    this.fft(this.TspR, this.TspI, false);
    if (this.ready) this.method.init(this);
  }
  _wienerInit() {
    const n2 = this.N * this.N, K = this.params.K, wr = this.wr, wi = this.wi;
    for (let i = 0; i < n2; i++) { const g = this.Bk[i] / (this.A[i] + K); wr[i] = g * this.TspR[i]; wi[i] = g * this.TspI[i]; }
    this.fft(wr, wi, true);
    for (let i = 0; i < n2; i++) { const v = Math.min(1, Math.max(0, wr[i])); this.X[i] = v; this.Y[i] = v; }
    this.iters = 0; this.fistaT = 1;
  }
  fistaStep() {
    const n2 = this.N * this.N, wr = this.wr, wi = this.wi;
    wr.set(this.Y); wi.fill(0); this.fft(wr, wi, false);
    for (let i = 0; i < n2; i++) { wr[i] = this.A[i] * wr[i] - this.Bk[i] * this.TspR[i]; wi[i] = this.A[i] * wi[i] - this.Bk[i] * this.TspI[i]; }
    this.fft(wr, wi, true);
    const tn = (1 + Math.sqrt(1 + 4 * this.fistaT * this.fistaT)) / 2, beta = (this.fistaT - 1) / tn; this.fistaT = tn;
    for (let i = 0; i < n2; i++) { const xn = Math.min(1, Math.max(0, this.Y[i] - wr[i])), xo = this.X[i];
      this.X[i] = xn; this.Y[i] = xn + beta * (xn - xo); }
    this.iters++;
  }
  iterate(n) { if (!this.ready || !this.hasTarget || this.iters >= this.maxIters()) return 0;
    const k = Math.min(n, 2); for (let i = 0; i < k; i++) this.method.step(this); return k; }   // ≤2 per frame on CPU
  render() {
    if (!this.hasTarget) return;
    const N = this.N, P = this.params, use = P.filter && this.ready, o = this.img.data;
    const enc = (v) => { v = Math.min(1, Math.max(0, v)); return 255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055); };
    let R = this.R, G = this.G, B = this.B, X = this.X;
    const out = [new Float32Array(N * N), new Float32Array(N * N), new Float32Array(N * N)];
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const i = y * N + x, proc = use && !(P.split && x < N / 2);
      if (!proc) { out[0][i] = P.photo ? R[i] : this.T[i]; out[1][i] = P.photo ? G[i] : this.T[i]; out[2][i] = P.photo ? B[i] : this.T[i]; }
      else if (P.photo) { const dlt = X[i] - this.T[i]; out[0][i] = R[i] + dlt; out[1][i] = G[i] + dlt; out[2][i] = B[i] + dlt; }
      else out[0][i] = out[1][i] = out[2][i] = X[i]; }
    if (P.retina && this.ready) for (const c of out) { const wr = this.wr, wi = this.wi; wr.set(c); wi.fill(0); this.fft(wr, wi, false);
      for (let i = 0; i < N * N; i++) { wr[i] *= this.Hr[i]; wi[i] *= this.Hr[i]; } this.fft(wr, wi, true); c.set(wr); }
    for (let i = 0; i < N * N; i++) { o[4 * i] = enc(out[0][i]); o[4 * i + 1] = enc(out[1][i]); o[4 * i + 2] = enc(out[2][i]); o[4 * i + 3] = 255; }
    this.ctx.putImageData(this.img, 0, 0); this._last = out[1];
  }
  check() {
    const a = this._last; if (!a || !this.hasTarget) return { ok: true };
    let bad = 0, mx = 0, tm = 0; const st = Math.max(1, (a.length / 4096) | 0);
    for (let i = 0; i < a.length; i += st) { if (!Number.isFinite(a[i])) bad++; else mx = Math.max(mx, a[i]); tm += this.T[i]; }
    tm /= a.length / st;
    if (bad) return { ok: false, reason: `${bad} NaN (CPU)` };
    if (tm > 0.2 && mx < 0.05) return { ok: false, reason: "saída preta (CPU)" };
    return { ok: true, mx, tmean: tm };
  }
}
