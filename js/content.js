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

// ---------- TEST box: small letters (true angular size) + small phone text ----------
// cols=2 -> the same test in each half (compare mode: left original | right filtered)
export function drawTest(ctx, N, o) {
  const { dM, pixMM, scale, cols = 1, seed = 3 } = o;
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, N, N); ctx.fillStyle = "#000";
  const colW = N / cols, m = Math.round(colW * 0.05); let yEnd = 0;
  let s = seed; const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  const rowsLm = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2], yMax = N * (o.crop || 0.98);
  const rows = rowsLm.map(lm => { let r = ""; while (r.length < 8) { const c = SLOAN[Math.floor(rnd() * 10)]; if (r.slice(-1) !== c) r += c; } return { lm, r }; });
  for (let c = 0; c < cols; c++) {
    const x0 = c * colW + m, w = colW - 2 * m; let y = m;
    // small phone text lines (13 / 11 CSS px)
    for (const [px, txt] of [[14, "Chego às 18h30. Traga o documento e a receita do médico."], [12, "Saldo disponível: R$ 1.284,50 · Pix recebido às 09:41"], [11, "Sua entrega chega amanhã entre 8h e 12h."]]) {
      const f = px * scale; ctx.font = `400 ${f}px ${FONT}`; ctx.textAlign = "start"; ctx.textBaseline = "alphabetic";
      for (const l of wrap(ctx, txt, w)) { y += f * 1.25; if (y > yMax) break; ctx.fillText(l, x0, y); }
      y += f * 0.5;
    }
    // ETDRS-like rows, true size for the distance; as many letters as fit
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (const r of rows) {
      const h = letterPx(r.lm, dM, pixMM); if (h < 3) continue;
      if (y + 2.2 * h > yMax) break;
      ctx.font = `700 ${h / 0.72}px ${FONT}`;
      const n = Math.max(3, Math.min(8, Math.floor(w / (2 * h))));
      for (let i = 0; i < n; i++) ctx.fillText(r.r[i], x0 + w / 2 + (i - (n - 1) / 2) * 2 * h, y + 1.1 * h);
      y += 2.2 * h;
    }
    ctx.textAlign = "start"; ctx.textBaseline = "alphabetic"; yEnd = y;
  }
  if (cols === 2) { ctx.fillStyle = "#fff"; ctx.fillRect(N / 2 - 2, 0, 4, N); }
  return yEnd;
}

// ---------- USE mode: scrollable feed (article + chat + buttons), laid out in CSS px ----------
export function layoutUse(ctx, W, st) {
  const items = []; let y = 14; const m = 14;
  const font = (px, wgt = 400) => `${wgt} ${px}px ${FONT}`;
  const para = (txt, px = 17, wgt = 400) => { ctx.font = font(px, wgt); for (const l of wrap(ctx, txt, W - 2 * m)) { y += px * 1.35; items.push({ t: "text", x: m, y, s: l, px, wgt }); } y += px * 0.6; };
  const btn = (label, id, x, w) => { items.push({ t: "btn", x, y, w, h: 48, s: label, id }); };
  para("Notícias de hoje", 24, 700);
  para("Cidade inaugura praça com área verde e wi-fi gratuito", 19, 700);
  para("A nova praça tem bancos com sombra, brinquedos acessíveis e pontos de recarga para celular. A prefeitura diz que a manutenção será feita por moradores do bairro em parceria com a escola.");
  btn(`👍 Curtir (${st.likes})`, "like", m, (W - 3 * m) / 2); btn("Ler mais", "more", m * 2 + (W - 3 * m) / 2, (W - 3 * m) / 2); y += 62;
  for (let i = 0; i < st.more; i++) para("O projeto também prevê uma feira de artesanato aos sábados e aulas de alongamento gratuitas pela manhã, abertas a todas as idades.");
  para("Mensagens", 22, 700);
  for (const msg of st.chat) {
    ctx.font = font(16); const lines = wrap(ctx, msg.s, W * 0.68); const bw = Math.min(W * 0.72, Math.max(...lines.map(l => ctx.measureText(l).width)) + 24);
    const bh = lines.length * 16 * 1.35 + 16, x = msg.me ? W - m - bw : m;
    items.push({ t: "bubble", x, y, w: bw, h: bh, me: msg.me, lines, from: msg.from }); y += bh + 10;
  }
  btn("Responder “Ok, combinado!”", "reply", m, W - 2 * m); y += 62;
  para("Dica: aumente o brilho da tela ao máximo e mantenha o celular a um palmo do rosto.", 16);
  y += 40; return { items, H: y };
}
export function drawUse(ctx, N, W, scrollY, L) {
  const sc = N / W; ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, N, N);
  ctx.setTransform(sc, 0, 0, sc, 0, -scrollY * sc); ctx.textBaseline = "alphabetic";
  const top = scrollY - 60, bot = scrollY + W + 60;
  for (const it of L.items) {
    if ((it.y + (it.h || 0)) < top || it.y - 40 > bot) continue;
    if (it.t === "text") { ctx.fillStyle = "#000"; ctx.font = `${it.wgt} ${it.px}px ${FONT}`; ctx.fillText(it.s, it.x, it.y); }
    else if (it.t === "btn") { ctx.fillStyle = "#000"; rr(ctx, it.x, it.y, it.w, it.h, 12); ctx.fill(); ctx.fillStyle = "#fff"; rr(ctx, it.x + 3, it.y + 3, it.w - 6, it.h - 6, 10); ctx.fill();
      ctx.fillStyle = "#000"; ctx.font = `700 17px ${FONT}`; ctx.textAlign = "center"; ctx.fillText(it.s, it.x + it.w / 2, it.y + 30); ctx.textAlign = "start"; }
    else if (it.t === "bubble") { ctx.fillStyle = it.me ? "#d8d8d8" : "#ececec"; rr(ctx, it.x, it.y, it.w, it.h, 14); ctx.fill(); ctx.fillStyle = "#000"; ctx.font = `400 16px ${FONT}`;
      it.lines.forEach((l, i) => ctx.fillText(l, it.x + 12, it.y + 8 + (i + 1) * 16 * 1.35 - 4)); }
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}
function rr(ctx, x, y, w, h, r) { ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); }
export function hitUse(L, x, y) { return L.items.find(it => it.t === "btn" && x >= it.x && x <= it.x + it.w && y >= it.y && y <= it.y + it.h); }
