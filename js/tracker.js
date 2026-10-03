// Front-camera face tracking, fully on-device (MediaPipe Face Landmarker, WASM/GPU). No frame leaves the phone.
// Distance from the iris diameter (HVID ≈ 11.7 mm): d = f_px * 11.7 / iris_px.  f_px from an assumed
// horizontal FOV (default 60°) times a user calibration factor. Head roll rotates the cyl axis;
// view angles (yaw/pitch of the line of sight w.r.t. screen normal) come from the face position.
const MP_VER = "0.10.14";
const MP = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}`;
const MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

export class Tracker {
  constructor() {
    this.state = { ok: false, source: "padrão 22 cm", d: 0.22, yaw: 0, pitch: 0, roll: 0, rightOpen: true, leftOpen: true, fps: 0 };
    this.hfovDeg = 60; this.calib = 1.0; this.fallback = 0.22; this._dHist = [];
    this.onUpdate = () => {};
  }
  async start(videoEl) {
    try {
      const vision = await import(`${MP}/vision_bundle.mjs`);
      const files = await vision.FilesetResolver.forVisionTasks(`${MP}/wasm`);
      this.lm = await vision.FaceLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: MODEL, delegate: "GPU" }, runningMode: "VIDEO", numFaces: 1,
        outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true });
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
      videoEl.srcObject = stream; videoEl.muted = true; videoEl.playsInline = true; await videoEl.play();
      this.video = videoEl; this.running = true; this._loop();
      return true;
    } catch (e) {
      this.state = { ...this.state, ok: false, source: "padrão 22 cm (câmera indisponível)", d: this.fallback, err: String(e) };
      this.onUpdate(this.state); return false;
    }
  }
  stop() { this.running = false; this.video?.srcObject?.getTracks().forEach(t => t.stop()); }
  _loop() {
    if (!this.running) return;
    const v = this.video; let last = performance.now(), n = 0, t0 = last;
    const step = () => {
      if (!this.running) return;
      if (v.readyState >= 2) {
        const now = performance.now();
        const r = this.lm.detectForVideo(v, now);
        this._process(r, v.videoWidth, v.videoHeight); n++;
        if (now - t0 > 1000) { this.state.fps = n; n = 0; t0 = now; }
      }
      (v.requestVideoFrameCallback ? v.requestVideoFrameCallback(step) : requestAnimationFrame(step));
    };
    step();
  }
  _process(r, W, H) {
    const s = this.state;
    if (!r.faceLandmarks || !r.faceLandmarks.length) { s.ok = false; s.source = "rosto não detectado (mantendo último)"; this.onUpdate(s); return; }
    const L = r.faceLandmarks[0];
    const px = (i) => [L[i].x * W, L[i].y * H];
    const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
    // iris contours: 469/471 (one eye) and 474/476 (other), horizontal diameter
    const irisA = dist(px(469), px(471)), irisB = dist(px(474), px(476));
    const iris = Math.max(irisA, irisB);              // the larger is less foreshortened
    const f = (W / 2) / Math.tan(this.hfovDeg * Math.PI / 360);
    let d = f * 11.7e-3 / Math.max(iris, 1) * this.calib;
    // robust smoothing: median of last 9, then one-pole
    this._dHist.push(d); if (this._dHist.length > 9) this._dHist.shift();
    const med = [...this._dHist].sort((a, b) => a - b)[this._dHist.length >> 1];
    s.d = s.ok ? s.d + 0.15 * (med - s.d) : med;   // heavier smoothing (W5): PSF must not jitter
    // line of sight angles from the face position in the image (camera ~ at the top of the screen)
    const c = [(L[468].x + L[473].x) / 2 * W - W / 2, (L[468].y + L[473].y) / 2 * H - H / 2];
    s.yaw = Math.atan2(c[0], f) * 180 / Math.PI; s.pitch = Math.atan2(c[1], f) * 180 / Math.PI;
    // head roll from the facial transformation matrix (column-major 4x4)
    const m = r.facialTransformationMatrixes?.[0]?.data;
    if (m) s.roll = Math.atan2(m[1], m[0]) * 180 / Math.PI;
    // eye openness (blendshape names follow ARKit; "Left" = the user's left eye)
    const bs = r.faceBlendshapes?.[0]?.categories || [];
    const g = (n) => (bs.find(c => c.categoryName === n) || { score: 0 }).score;
    s.leftOpen = g("eyeBlinkLeft") < 0.5; s.rightOpen = g("eyeBlinkRight") < 0.5;
    s.ok = true; s.source = "câmera (íris)";
    this.onUpdate(s);
  }
}
