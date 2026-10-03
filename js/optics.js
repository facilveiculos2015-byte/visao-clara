// Prescription -> near residual -> OSA Zernike (Z2^0, Z2^±2). Mirrors sim/optics.py (validated there).
export const PRIMARIES = [ // approx. display primaries (nm) and luminance weight of white
  { lam: 615, w: 0.2126 }, { lam: 540, w: 0.7152 }, { lam: 460, w: 0.0722 }];

// Longitudinal chromatic aberration of the eye (Thibos 1992 "chromatic eye"), referenced at 555 nm.
export function lca(lamNm) {
  const f = (l) => 1.68524 - 633.46 / (l - 214.102);
  return f(lamNm) - f(555);
}

// Residual refractive error at a screen at distance d (m): S_near = S + 1/d,
// positive part reduced by available accommodation (choose circle of least confusion).
export function nearRx(rx, dM, accom = 0) {
  let s = rx.S + 1 / dM;
  if (s > 0) { const m = s + rx.C / 2; s -= Math.min(Math.max(m, 0), accom); }
  return { S: s, C: rx.C, A: rx.A };
}

// TABO axis is measured CCW as the examiner faces the patient; the user faces the screen
// from the patient side -> mirrored in x on the screen.
export const taboToScreen = (a) => ((180 - a) % 180 + 180) % 180;

// OSA Zernike coefficients (µm) for pupil diameter (mm). axisScreenDeg already in screen coords.
export function zernike(S, C, axisScreenDeg, pupilMM) {
  const r = pupilMM / 2, a = axisScreenDeg * Math.PI / 180;
  const M = S + C / 2, J0 = -(C / 2) * Math.cos(2 * a), J45 = -(C / 2) * Math.sin(2 * a);
  return { c20: -M * r * r / (4 * Math.sqrt(3)), c22: -J0 * r * r / (2 * Math.sqrt(6)), c2m2: -J45 * r * r / (2 * Math.sqrt(6)), M, J: Math.hypot(J0, J45) };
}

// Geometric blur (major axis) in arcmin and screen px — for UI honesty messages.
export function blurInfo(near, pupilMM, dM, pixMM) {
  const z = zernike(near.S, near.C, 0, pupilMM);
  const mrad = pupilMM * (Math.abs(z.M) + z.J);
  return { arcmin: mrad * 1e-3 * 180 / Math.PI * 60, px: mrad * 1e-3 * dM * 1000 / pixMM, D: Math.abs(z.M) + z.J };
}

// Letter (cap) height in px for logMAR at distance d.
export const letterPx = (logmar, dM, pixMM) => (5 * Math.pow(10, logmar) / 60) * Math.PI / 180 * dM * 1000 / pixMM;
