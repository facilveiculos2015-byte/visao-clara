// WebGL2 pre-compensation engine.
//  PSF  : pupil field (OSA Zernike Z2^0, Z2^±2, per display primary + eye LCA) -> FFT -> |.|^2
//         -> box-integrated per screen pixel (pixel aperture), oblique view by anisotropic sampling.
//  Model: a list of kernels (eye x distance) with weights -> A = Σw|H|^2, B = Σw conj(H)
//         (binocular compromise and distance-range robustness = joint least squares, one FFT pair/iter)
//  Solve: FISTA projected gradient, x in [0, hi], target t' = mid + b (t - mid), warm start,
//         Wiener (B T'/(A+K)) as initialiser / fast fallback. Optional per-channel R/G/B.
//  All maths in linear light; sRGB encode only at display.
import { PRIMARIES, lca, zernike } from "./optics.js";
import { methodById } from "./methods.js";

const VS = `#version 300 es
in vec2 p; void main(){ gl_Position = vec4(p,0.,1.); }`;
const HDR = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
#define PI 3.141592653589793
vec2 cmul(vec2 a, vec2 b){ return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
vec2 cconj(vec2 a){ return vec2(a.x, -a.y); }
`;
const FS = {
  fft: `uniform sampler2D u_in; uniform int u_size, u_sub, u_horiz; uniform float u_sign, u_scale;
  out vec4 o; void main(){ ivec2 p = ivec2(gl_FragCoord.xy); int idx = u_horiz==1 ? p.x : p.y;
   int h = u_sub/2; int ev = (idx / u_sub)*h + (idx % h); int od = ev + u_size/2;
   ivec2 pe = u_horiz==1 ? ivec2(ev,p.y) : ivec2(p.x,ev); ivec2 po = u_horiz==1 ? ivec2(od,p.y) : ivec2(p.x,od);
   vec4 e = texelFetch(u_in,pe,0), q = texelFetch(u_in,po,0);
   float a = u_sign*2.0*PI*float(idx % u_sub)/float(u_sub); vec2 tw = vec2(cos(a), sin(a));
   o = u_scale * vec4(e.xy + cmul(tw,q.xy), e.zw + cmul(tw,q.zw)); }`,
  // two wavelengths at once: xy <- lambda1, zw <- lambda2 (pupil grid extents differ per lambda)
  pupil: `uniform int u_G; uniform vec2 u_L1, u_L2; uniform float u_r, u_lam1, u_lam2; uniform vec3 u_z1, u_z2; uniform int u_two;
  out vec4 o;
  vec2 field(vec2 L, float lam, vec3 z){ ivec2 p = ivec2(gl_FragCoord.xy);
    vec2 i = vec2(p.x < u_G/2 ? p.x : p.x-u_G, p.y < u_G/2 ? p.y : p.y-u_G);
    vec2 xy = vec2(i.x, -i.y) * L / float(u_G); /* rows go DOWN the screen */ float rho2 = dot(xy,xy)/(u_r*u_r); if (rho2 > 1.0) return vec2(0);
    float phi = atan(xy.y, xy.x);
    float w = z.x*sqrt(3.)*(2.*rho2-1.) + z.y*sqrt(6.)*rho2*cos(2.*phi) + z.z*sqrt(6.)*rho2*sin(2.*phi);
    float ph = 2.*PI*w/lam; return vec2(cos(ph), sin(ph)); }
  void main(){ o = vec4(field(u_L1,u_lam1,u_z1), u_two==1 ? field(u_L2,u_lam2,u_z2) : vec2(0)); }`,
  bin: `uniform sampler2D u_in; uniform int u_G, u_S, u_os; out vec4 o;
  void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec2 s = vec2(0);
    for (int j=0;j<8;j++){ if (j>=u_os) break; for (int i=0;i<8;i++){ if (i>=u_os) break;
      ivec2 f = (p - u_S/2)*u_os + ivec2(i,j) - u_os/2; f = (f + 4*u_G) % u_G; /* keep % operands positive */
      vec4 v = texelFetch(u_in,f,0); s += vec2(dot(v.xy,v.xy), dot(v.zw,v.zw)); }}
    o = vec4(s, 0, 0); }`,
  // wrap centred SxS PSFs into NxN with centre at (0,0); MRT: (pR,0,pG,0) and (pB,0,0,0)
  wrap: `uniform sampler2D u_rg, u_b; uniform int u_N, u_S; layout(location=0) out vec4 o0; layout(location=1) out vec4 o1;
  void main(){ ivec2 p = ivec2(gl_FragCoord.xy); ivec2 d = ivec2(p.x < u_N/2 ? p.x : p.x-u_N, p.y < u_N/2 ? p.y : p.y-u_N);
    ivec2 s = d + u_S/2; vec2 rg = vec2(0); float b = 0.;
    if (all(greaterThanEqual(s, ivec2(0))) && all(lessThan(s, ivec2(u_S)))) { rg = texelFetch(u_rg,s,0).xy; b = texelFetch(u_b,s,0).x; }
    o0 = vec4(rg.x,0,rg.y,0); o1 = vec4(b,0,0,0); }`,
  // normalise spectra by DC, accumulate A=Σw|H|^2, B=Σw conj H for R,G,B,L ; and export H of the reference kernel
  accum: `uniform sampler2D u_h1, u_h2, u_aR, u_aG, u_aB, u_aL; uniform float u_w; uniform vec3 u_lw;
  layout(location=0) out vec4 oR; layout(location=1) out vec4 oG; layout(location=2) out vec4 oB; layout(location=3) out vec4 oL;
  layout(location=4) out vec4 oH1; layout(location=5) out vec4 oH2;
  void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec4 h1 = texelFetch(u_h1,p,0), h2 = texelFetch(u_h2,p,0);
    vec4 d1 = texelFetch(u_h1,ivec2(0),0), d2 = texelFetch(u_h2,ivec2(0),0);
    vec2 HR = h1.xy/d1.x, HG = h1.zw/d1.z, HB = h2.xy/d2.x; vec2 HL = u_lw.x*HR + u_lw.y*HG + u_lw.z*HB;
    oR = texelFetch(u_aR,p,0) + u_w*vec4(cconj(HR), dot(HR,HR), 0);
    oG = texelFetch(u_aG,p,0) + u_w*vec4(cconj(HG), dot(HG,HG), 0);
    oB = texelFetch(u_aB,p,0) + u_w*vec4(cconj(HB), dot(HB,HB), 0);
    oL = texelFetch(u_aL,p,0) + u_w*vec4(cconj(HL), dot(HL,HL), 0);
    oH1 = vec4(HR, HG); oH2 = vec4(HB, HL); }`,
  // canvas (sRGB 8-bit) -> linear target t (luminance) and linear rgb (photo)
  linearize: `uniform sampler2D u_img; uniform int u_N; layout(location=0) out vec4 oT; layout(location=1) out vec4 oRGB;
  vec3 lin(vec3 c){ return mix(c/12.92, pow((c+0.055)/1.055, vec3(2.4)), step(0.04045, c)); }
  void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 c = lin(texelFetch(u_img, p, 0).rgb);   // row 0 = top of the canvas
    float Y = dot(c, vec3(0.2126,0.7152,0.0722)); oT = vec4(Y); oRGB = vec4(c,1); }`,
  packTp: `uniform sampler2D u_t; uniform float u_mid, u_b; out vec4 o;
  void main(){ float t = texelFetch(u_t, ivec2(gl_FragCoord.xy),0).x; o = vec4(u_mid + u_b*(t-u_mid), 0, t, 0); }`,
  // pack an RGB real texture into two complex textures
  pack: `uniform sampler2D u_x; uniform int u_rgb; layout(location=0) out vec4 o0; layout(location=1) out vec4 o1;
  void main(){ vec4 x = texelFetch(u_x, ivec2(gl_FragCoord.xy),0); if (u_rgb==1){ o0 = vec4(x.r,0,x.g,0); o1 = vec4(x.b,0,0,0);} else { o0 = vec4(x.r,0,0,0); o1 = vec4(0);} }`,
  wiener: `uniform sampler2D u_aR, u_aG, u_aB, u_aL, u_T; uniform float u_K; uniform int u_rgb;
  layout(location=0) out vec4 o0; layout(location=1) out vec4 o1;
  vec2 f(vec4 ab, vec2 T){ return cmul(ab.xy, T) / (ab.z + u_K); }
  void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec2 T = texelFetch(u_T,p,0).xy;
    if (u_rgb==1){ o0 = vec4(f(texelFetch(u_aR,p,0),T), f(texelFetch(u_aG,p,0),T)); o1 = vec4(f(texelFetch(u_aB,p,0),T),0,0); }
    else { o0 = vec4(f(texelFetch(u_aL,p,0),T),0,0); o1 = vec4(0); } }`,
  initX: `uniform sampler2D u_g1, u_g2; uniform int u_rgb; uniform float u_hi; layout(location=0) out vec4 oY; layout(location=1) out vec4 oX;
  void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec4 a = texelFetch(u_g1,p,0), b = texelFetch(u_g2,p,0);
    vec3 x = u_rgb==1 ? vec3(a.x, a.z, b.x) : vec3(a.x); x = clamp(x, 0., u_hi); oY = vec4(x,1); oX = vec4(x,1); }`,
  grad: `uniform sampler2D u_y1, u_y2, u_aR, u_aG, u_aB, u_aL, u_T; uniform int u_rgb;
  layout(location=0) out vec4 o0; layout(location=1) out vec4 o1;
  vec2 g(vec4 ab, vec2 Y, vec2 T){ return ab.z*Y - cmul(ab.xy, T); }
  void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec2 T = texelFetch(u_T,p,0).xy; vec4 y1 = texelFetch(u_y1,p,0), y2 = texelFetch(u_y2,p,0);
    if (u_rgb==1){ o0 = vec4(g(texelFetch(u_aR,p,0),y1.xy,T), g(texelFetch(u_aG,p,0),y1.zw,T)); o1 = vec4(g(texelFetch(u_aB,p,0),y2.xy,T),0,0); }
    else { o0 = vec4(g(texelFetch(u_aL,p,0),y1.xy,T),0,0); o1 = vec4(0); } }`,
  update: `uniform sampler2D u_g1, u_g2, u_y, u_x; uniform float u_step, u_beta, u_hi; uniform int u_rgb;
  layout(location=0) out vec4 oY; layout(location=1) out vec4 oX;
  void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec4 a = texelFetch(u_g1,p,0), b = texelFetch(u_g2,p,0);
    vec3 g = u_rgb==1 ? vec3(a.x, a.z, b.x) : vec3(a.x);
    vec3 y = texelFetch(u_y,p,0).rgb, xo = texelFetch(u_x,p,0).rgb;
    vec3 xn = clamp(y - u_step*g, 0., u_hi); oX = vec4(xn,1); oY = vec4(xn + u_beta*(xn - xo), 1); }`,
  // composite: what the screen shows (rgb linear). split: left half original, right half processed
  composite: `uniform sampler2D u_x, u_t, u_rgb; uniform int u_N, u_split, u_filter, u_photo; out vec4 o;
  void main(){ ivec2 p = ivec2(gl_FragCoord.xy); float t = texelFetch(u_t,p,0).x; vec3 x = texelFetch(u_x,p,0).rgb;
    vec3 orig = u_photo==1 ? texelFetch(u_rgb,p,0).rgb : vec3(t);
    vec3 proc = u_photo==1 ? max(orig + (x.r - t), 0.) : x;      // photo: Y-only change, chroma kept
    bool useProc = u_filter==1 && !(u_split==1 && p.x < u_N/2);
    o = vec4(useProc ? proc : orig, 1); }`,
  retinaMul: `uniform sampler2D u_c1, u_c2, u_h1, u_h2; layout(location=0) out vec4 o0; layout(location=1) out vec4 o1;
  void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec4 c1 = texelFetch(u_c1,p,0), c2 = texelFetch(u_c2,p,0), h1 = texelFetch(u_h1,p,0), h2 = texelFetch(u_h2,p,0);
    o0 = vec4(cmul(h1.xy,c1.xy), cmul(h1.zw,c1.zw)); o1 = vec4(cmul(h2.xy,c2.xy),0,0); }`,
  display: `uniform sampler2D u_c, u_r1, u_r2; uniform int u_N, u_retina, u_hdr; uniform float u_gain; out vec4 o;
  vec3 enc(vec3 c){ vec3 a = abs(c); vec3 e = mix(a*12.92, 1.055*pow(a, vec3(1./2.4)) - 0.055, step(0.0031308, a)); return sign(c)*e; }
  void main(){ ivec2 p = ivec2(int(gl_FragCoord.x), u_N - 1 - int(gl_FragCoord.y));
    vec3 c = u_retina==1 ? vec3(texelFetch(u_r1,p,0).x, texelFetch(u_r1,p,0).z, texelFetch(u_r2,p,0).x) : texelFetch(u_c,p,0).rgb;
    c *= u_gain; if (u_hdr==0) c = clamp(c, 0., 1.); o = vec4(enc(c), 1); }`,
};

export class Engine {
  constructor(canvas, N = 1024, opts = {}) {
    this.canvas = canvas; this.N = N; canvas.width = N; canvas.height = N;
    const gl = canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: false, preserveDrawingBuffer: !!opts.preserve, alpha: false });
    if (!gl) throw new Error("WebGL2 indisponível");
    if (!gl.getExtension("EXT_color_buffer_float")) throw new Error("Sem EXT_color_buffer_float (render em float)");
    this.gl = gl;
    this.hdr = false;
    this.maxDraw = gl.getParameter(gl.MAX_DRAW_BUFFERS);
    if (this.maxDraw < 6) throw new Error("MAX_DRAW_BUFFERS < 6");
    const vb = gl.createBuffer(); this.vb = vb; gl.bindBuffer(gl.ARRAY_BUFFER, vb);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    this.prog = {};
    for (const [k, src] of Object.entries(FS)) this.prog[k] = this._program(HDR + src);
    this.G = 1024;
    const T = (n) => this._tex(n);
    this.t = {};
    for (const k of ["fa", "fb", "T", "RGB", "Tspec", "tmp", "c1", "c2", "s1", "s2", "g1", "g2", "X0", "X1", "Y0", "Y1", "C",
      "aR0", "aG0", "aB0", "aL0", "aR1", "aG1", "aB1", "aL1", "H1", "H2", "Hr1", "Hr2", "Hk1", "Hk2", "w1", "w2", "r1", "r2", "imgSrc"]) this.t[k] = null;
    for (const k of Object.keys(this.t)) if (k !== "imgSrc") this.t[k] = T(N);
    this.t.w1 = T(512); this.t.w2 = T(512);   // binned SxS PSFs (S <= 384)
    this.t.ga = T(this.G); this.t.gb = T(this.G); this.t.gc = T(this.G);
    this.xi = 0; this.ai = 0; this.fistaT = 1; this.iters = 0; this.ready = false;
    this.params = { b: 0.6, mid: 0.5, K: 0.01, hi: 1.0, rgb: false, method: "fista", split: true, filter: true, retina: false, photo: false };
  }

  enableHDR() { // (a) HDR/EDR headroom – only where the browser exposes an extended-range WebGL canvas
    const gl = this.gl, c = this.canvas;
    try {
      if (typeof gl.drawingBufferStorage === "function" && typeof c.configureHighDynamicRange === "function") {
        gl.drawingBufferStorage(gl.RGBA16F, this.N, this.N);
        c.configureHighDynamicRange({ mode: "extended" });
        this.hdr = true;
      }
    } catch (e) { this.hdr = false; }
    return this.hdr;
  }

  _program(fs) {
    const gl = this.gl;
    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) + "\n" + src.slice(0, 400)); return s; };
    const p = gl.createProgram(); gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
    gl.bindAttribLocation(p, 0, "p"); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    p.u = {}; return p;
  }
  _tex(n) {
    const gl = this.gl, t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, n, n);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    t.n = n; const fb = gl.createFramebuffer(); t.fb = fb; return t;
  }
  // run program `name` writing into `outs` (array of textures, MRT), with uniforms (textures auto-bound)
  run(name, outs, uni = {}) {
    const gl = this.gl, p = this.prog[name]; gl.useProgram(p);
    let unit = 0;
    for (const [k, v] of Object.entries(uni)) {
      let loc = p.u[k]; if (loc === undefined) loc = p.u[k] = gl.getUniformLocation(p, k);
      if (loc === null) continue;
      if (v && v.fb !== undefined) { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, v); gl.uniform1i(loc, unit++); }
      else if (typeof v === "object" && v.i !== undefined) gl.uniform1i(loc, v.i);
      else if (Array.isArray(v)) (v.length === 2 ? gl.uniform2fv : gl.uniform3fv).call(gl, loc, v);
      else gl.uniform1f(loc, v);
    }
    if (outs === null) { gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, this.N, this.N); }
    else {
      const fb = outs[0].fb; gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      outs.forEach((t, i) => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t, 0));
      for (let i = outs.length; i < 6; i++) gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, null, 0);
      gl.drawBuffers(outs.map((_, i) => gl.COLOR_ATTACHMENT0 + i));
      gl.viewport(0, 0, outs[0].n, outs[0].n);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vb);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    // unbind sampled textures to avoid feedback loops on the next pass
    for (let i = 0; i < unit; i++) { gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, null); }
  }
  // 2D FFT (radix-2 Stockham), src -> dst, using scratch a/b of the same size
  fft(src, dst, inverse, a = this.t.fa, b = this.t.fb) {
    const n = src.n, L = Math.log2(n), sign = inverse ? 1 : -1, sc = inverse ? 0.5 : 1;
    let cur = src, passes = [];
    for (const h of [1, 0]) for (let s = 1; s <= L; s++) passes.push([h, 1 << s]);
    passes.forEach(([h, sub], i) => {
      const out = i === passes.length - 1 ? dst : (i % 2 === 0 ? a : b);
      this.run("fft", [out], { u_in: cur, u_size: { i: n }, u_sub: { i: sub }, u_horiz: { i: h }, u_sign: sign, u_scale: sc });
      cur = out;
    });
  }

  // ---------- model ----------
  // kernels: [{S,C,axisScreen,d,pupil,yaw,pitch,roll,w}], pixMM = physical size of one canvas pixel
  setModel(kernels, pixMM) {
    const gl = this.gl, N = this.N, G = this.G;
    // zero accumulators
    for (const k of ["aR", "aG", "aB", "aL"]) for (const s of [0, 1]) this._clear(this.t[k + s]);
    let ai = 0; const wsum = kernels.reduce((a, k) => a + k.w, 0);
    let info = null;
    kernels.forEach((K, idx) => {
      const lamMin = 460e-6;
      const dthBase = pixMM / (K.d * 1000);
      const os = Math.min(8, Math.max(2, Math.ceil(2.1 * K.pupil * dthBase / lamMin)));
      const S = Math.min(N / 2, Math.floor(G / os / 2) * 2, 384);
      const dthX = pixMM * Math.cos(K.yaw * Math.PI / 180) / (K.d * 1000) / os;
      const dthY = pixMM * Math.cos(K.pitch * Math.PI / 180) / (K.d * 1000) / os;
      const z = PRIMARIES.map(pr => { const zz = zernike(K.S + lca(pr.lam), K.C, K.axisScreen + K.roll, K.pupil); return [zz.c20, zz.c22, zz.c2m2]; });
      const L = PRIMARIES.map(pr => [pr.lam * 1e-6 / dthX, pr.lam * 1e-6 / dthY]);
      const r = K.pupil / 2;
      // R,G pair
      this.run("pupil", [this.t.gc], { u_G: { i: G }, u_L1: L[0], u_L2: L[1], u_r: r, u_lam1: PRIMARIES[0].lam * 1e-3, u_lam2: PRIMARIES[1].lam * 1e-3, u_z1: z[0], u_z2: z[1], u_two: { i: 1 } });
      this.fft(this.t.gc, this.t.gc, false, this.t.ga, this.t.gb);
      this.run("bin", [this.t.w1], { u_in: this.t.gc, u_G: { i: G }, u_S: { i: S }, u_os: { i: os } });
      this.run("pupil", [this.t.gc], { u_G: { i: G }, u_L1: L[2], u_L2: L[2], u_r: r, u_lam1: PRIMARIES[2].lam * 1e-3, u_lam2: 1, u_z1: z[2], u_z2: z[2], u_two: { i: 0 } });
      this.fft(this.t.gc, this.t.gc, false, this.t.ga, this.t.gb);
      this.run("bin", [this.t.w2], { u_in: this.t.gc, u_G: { i: G }, u_S: { i: S }, u_os: { i: os } });
      this.run("wrap", [this.t.c1, this.t.c2], { u_rg: this.t.w1, u_b: this.t.w2, u_N: { i: N }, u_S: { i: S } });
      this.fft(this.t.c1, this.t.H1, false); this.fft(this.t.c2, this.t.H2, false);
      const src = ai, dst = 1 - ai;
      this.run("accum", [this.t["aR" + dst], this.t["aG" + dst], this.t["aB" + dst], this.t["aL" + dst], idx === 0 ? this.t.Hr1 : this.t.Hk1, idx === 0 ? this.t.Hr2 : this.t.Hk2],
        { u_h1: this.t.H1, u_h2: this.t.H2, u_aR: this.t["aR" + src], u_aG: this.t["aG" + src], u_aB: this.t["aB" + src], u_aL: this.t["aL" + src],
          u_w: K.w / wsum, u_lw: PRIMARIES.map(p => p.w) });
      ai = dst;
      if (idx === 0) info = { os, S };
    });
    this.ai = ai; this.fistaT = 1; this.iters = 0; this.modelInfo = info;
    this.ready = true;
    // warm start: keep the current solution when only the PSF changed (pose/distance), restart momentum
    if (this.hasTarget) { if (this._hasX && this.method.maxIters > 0) { this.fistaT = 1; this.iters = 0; } else this.method.init(this); }
  }
  _clear(t) { const gl = this.gl; gl.bindFramebuffer(gl.FRAMEBUFFER, t.fb); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
    for (let i = 1; i < 6; i++) gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, null, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]); gl.viewport(0, 0, t.n, t.n); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); }

  // ---------- target ----------
  setTarget(canvas2d, photo = false) {
    const gl = this.gl;
    if (!this.t.imgSrc) { this.t.imgSrc = gl.createTexture(); }
    gl.bindTexture(gl.TEXTURE_2D, this.t.imgSrc);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, canvas2d);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    const img = this.t.imgSrc; img.fb = 0; // mark as texture for run()
    this.run("linearize", [this.t.T, this.t.RGB], { u_img: img, u_N: { i: this.N } });
    this.params.photo = photo;
    this.hasTarget = true; this._hasX = false;
    this.retarget();
  }
  retarget() { // contrast target changed (b / mid)
    if (!this.hasTarget) return;
    this.run("packTp", [this.t.tmp], { u_t: this.t.T, u_mid: this.params.mid, u_b: this.params.b });
    this.fft(this.t.tmp, this.t.Tspec, false);
    if (this.ready) this.method.init(this);
  }
  _a(c) { return this.t["a" + c + this.ai]; }
  _wienerInit() {
    const P = this.params, rgb = { i: P.rgb ? 1 : 0 };
    this.run("wiener", [this.t.s1, this.t.s2], { u_aR: this._a("R"), u_aG: this._a("G"), u_aB: this._a("B"), u_aL: this._a("L"), u_T: this.t.Tspec, u_K: P.K, u_rgb: rgb });
    this.fft(this.t.s1, this.t.g1, true); if (P.rgb) this.fft(this.t.s2, this.t.g2, true);
    this.run("initX", [this.t["Y" + this.xi], this.t["X" + this.xi]], { u_g1: this.t.g1, u_g2: this.t.g2, u_rgb: rgb, u_hi: P.hi });
    this.fistaT = 1; this.iters = 0; this._hasX = true;
  }
  get method() { return methodById(this.params.method); }
  iterate(n) {
    if (!this.ready || !this.hasTarget) return 0;
    const m = this.method; if (this.iters >= m.maxIters) return 0;
    for (let k = 0; k < n; k++) m.step(this);
    return n;
  }
  fistaStep() {
    const P = this.params, rgb = { i: P.rgb ? 1 : 0 };
    {
      const Y = this.t["Y" + this.xi], X = this.t["X" + this.xi];
      this.run("pack", [this.t.c1, this.t.c2], { u_x: Y, u_rgb: rgb });
      this.fft(this.t.c1, this.t.s1, false); if (P.rgb) this.fft(this.t.c2, this.t.s2, false);
      this.run("grad", [this.t.c1, this.t.c2], { u_y1: this.t.s1, u_y2: this.t.s2, u_aR: this._a("R"), u_aG: this._a("G"), u_aB: this._a("B"), u_aL: this._a("L"), u_T: this.t.Tspec, u_rgb: rgb });
      this.fft(this.t.c1, this.t.g1, true); if (P.rgb) this.fft(this.t.c2, this.t.g2, true);
      const tn = (1 + Math.sqrt(1 + 4 * this.fistaT * this.fistaT)) / 2, beta = (this.fistaT - 1) / tn; this.fistaT = tn;
      const o = 1 - this.xi;
      this.run("update", [this.t["Y" + o], this.t["X" + o]], { u_g1: this.t.g1, u_g2: this.t.g2, u_y: Y, u_x: X, u_step: 1.0, u_beta: beta, u_hi: P.hi, u_rgb: rgb });
      this.xi = o; this.iters++;
    }
  }
  render() {
    if (!this.hasTarget) return;
    const P = this.params, N = this.N;
    const X = this.t["X" + this.xi];
    this.run("composite", [this.t.C], { u_x: X, u_t: this.t.T, u_rgb: this.t.RGB, u_N: { i: N }, u_split: { i: P.split ? 1 : 0 }, u_filter: { i: P.filter && this.ready ? 1 : 0 }, u_photo: { i: P.photo ? 1 : 0 } });
    if (P.retina && this.ready) {
      this.run("pack", [this.t.c1, this.t.c2], { u_x: this.t.C, u_rgb: { i: 1 } });
      this.fft(this.t.c1, this.t.s1, false); this.fft(this.t.c2, this.t.s2, false);
      this.run("retinaMul", [this.t.c1, this.t.c2], { u_c1: this.t.s1, u_c2: this.t.s2, u_h1: this.t.Hr1, u_h2: this.t.Hr2 });
      this.fft(this.t.c1, this.t.r1, true); this.fft(this.t.c2, this.t.r2, true);
    }
    this.run("display", null, { u_c: this.t.C, u_r1: this.t.r1, u_r2: this.t.r2, u_N: { i: N }, u_retina: { i: P.retina && this.ready ? 1 : 0 }, u_hdr: { i: this.hdr ? 1 : 0 }, u_gain: 1.0 });
  }
  // debug/QA: read back the reference luminance PSF magnitude spectrum DC etc.
  readTex(t, n = 8) { const gl = this.gl; gl.bindFramebuffer(gl.FRAMEBUFFER, t.fb); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
    gl.readBuffer(gl.COLOR_ATTACHMENT0); const a = new Float32Array(n * n * 4); gl.readPixels(0, 0, n, n, gl.RGBA, gl.FLOAT, a); return a; }
}
