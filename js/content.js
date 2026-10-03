// Content rendered from vector glyphs by the browser's text rasteriser, directly at the
// processing resolution (never an upscaled bitmap). Black on white, linearised in the shader.
import { letterPx } from "./optics.js";

export const PARAGRAPH = "Quando a gente lê no celular sem os óculos, as letras pequenas ficam borradas e cansam a vista. " +
  "Este texto é desenhado a partir dos contornos vetoriais das letras e pré-compensado para o seu grau, " +
  "a sua distância e o seu ângulo de visão. Compare os dois lados.";
export const SENTENCE = "O rato roeu a roupa do rei de Roma.";
const FONT = `-apple-system, "SF Pro Text", system-ui, Roboto, "Helvetica Neue", Arial, sans-serif`;

function wrap(ctx, text, maxW) {
  const words = text.split(" "), lines = []; let cur = "";
  for (const w of words) { const t = cur ? cur + " " + w : w; if (ctx.measureText(t).width > maxW && cur) { lines.push(cur); cur = w; } else cur = t; }
  if (cur) lines.push(cur); return lines;
}

// (d) experimental "vector pre-distortion": squeeze glyph outlines along the blur major axis
function drawLine(ctx, text, x, y, pd) {
  if (!pd) { ctx.fillText(text, x, y); return; }
  const w = ctx.measureText(text).width, cx = x + w / 2;
  ctx.save(); ctx.translate(cx, y); ctx.rotate(-pd.angle); ctx.scale(pd.scale, 1); ctx.rotate(pd.angle);
  ctx.fillText(text, -w / 2, 0); ctx.restore();
}

export function drawReading(ctx, N, o) {
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, N, N); ctx.fillStyle = "#000"; ctx.textBaseline = "alphabetic";
  const cols = o.split ? 2 : 1, colW = N / cols, margin = Math.round(N * 0.045);
  const fpx = o.fontCss * o.scale;   // CSS px -> canvas px
  ctx.font = `${o.weight || 400} ${fpx}px ${FONT}`;
  for (let c = 0; c < cols; c++) {
    const x0 = c * colW + margin, lines = wrap(ctx, o.text, colW - 2 * margin);
    let y = margin + fpx;
    for (const l of lines) { if (y > N * 0.66) break; drawLine(ctx, l, x0, y, o.predistort); y += fpx * 1.35; }
    if (o.grid) { // grid in the lower third
      const g0 = Math.round(N * 0.70), step = Math.max(8, Math.round(fpx * 0.9)), lw = Math.max(1, Math.round(fpx / 12));
      for (let yy = g0; yy < N - margin; yy += step) ctx.fillRect(c * colW + margin, yy, colW - 2 * margin, lw);
      for (let xx = c * colW + margin; xx < (c + 1) * colW - margin; xx += step) ctx.fillRect(xx, g0, lw, N - margin - g0);
    }
  }
  if (o.split) { ctx.fillStyle = "#fff"; ctx.fillRect(N / 2 - 2, 0, 4, N); }
}

export const DISC_TEXT = "O rato roeu a roupa do rei de Roma. Quando a gente lê no celular sem os óculos, as letras pequenas ficam borradas e cansam a vista. Deslize até estas linhas ficarem o mais nítidas possível.";
// top=true: text starts at the top margin (the preview box crops the lower part); returns y after last line
export function drawSentence(ctx, N, text, fontCss, scale, pd, top = false) {
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, N, N); ctx.fillStyle = "#000";
  const fpx = fontCss * scale; ctx.font = `400 ${fpx}px ${FONT}`;
  const lines = wrap(ctx, text, N * 0.9);
  let y = top ? Math.round(N * 0.05) + fpx : N / 2 - (lines.length - 1) * fpx * 0.67 + fpx * 0.35;
  for (const l of lines) { if (top && y > N * 0.92) break; drawLine(ctx, l, N * 0.05, y, pd); y += fpx * 1.35; }
  return y - fpx * 1.35;
}

const SLOAN = "CDHKNORSVZ";
export function chartRows(seed = 1) {
  let s = seed; const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  return [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0].map(lm => {
    let r = ""; while (r.length < 5) { const c = SLOAN[Math.floor(rnd() * 10)]; if (!r.includes(c)) r += c; } return { lm, letters: r };
  });
}
// ETDRS-style chart: sizes are TRUE angular sizes for the current distance and pixel pitch.
export function drawChart(ctx, N, dM, pixMM, rows) {
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, N, N); ctx.fillStyle = "#000"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  let y = N * 0.04; const shown = [];
  for (const r of rows) {
    const h = letterPx(r.lm, dM, pixMM); if (10 * h > N * 0.96) continue;   // too big for the screen
    if (y + 2 * h > N * 0.98) break;
    ctx.font = `700 ${h / 0.72}px ${FONT}`;                                  // cap height ≈ 0.72 em
    for (let i = 0; i < 5; i++) ctx.fillText(r.letters[i], N / 2 + (i - 2) * 2 * h, y + h);
    shown.push({ ...r, y: y + h, h }); y += 2 * h + Math.max(4, h * 0.3);
  }
  ctx.textAlign = "start"; ctx.textBaseline = "alphabetic"; return shown;
}
