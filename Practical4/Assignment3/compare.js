'use strict';

// ── Models ────────────────────────────────────────────────────────────────────
// All 3 loaded from the same @tensorflow-models/mobilenet CDN — no TF Hub, no large downloads
let model1 = null;  // MobileNet v1  alpha=1.0  (~4.2M params)
let model2 = null;  // MobileNet v2  alpha=1.0  (~3.4M params)
let model3 = null;  // MobileNet v1  alpha=0.25 (~0.5M params) — ultra-fast "Lite"

let modelsReady = false;
let currentImg  = null;

const compareBtn    = document.getElementById('compareBtn');
const statsStrip    = document.getElementById('statsStrip');
const analysisPanel = document.getElementById('analysisPanel');
const loadProgress  = document.getElementById('loadProgress');
const previewImg    = document.getElementById('previewImg');

// ── Animated loading bar helper ───────────────────────────────────────────────
function animateBar(fillId, pctId, subId, steps, intervalMs) {
  let step = 0;
  const timer = setInterval(() => {
    if (step >= steps.length) { clearInterval(timer); return; }
    const { pct, msg } = steps[step++];
    document.getElementById(fillId).style.width = pct + '%';
    document.getElementById(pctId).textContent  = Math.round(pct) + '%';
    document.getElementById(subId).textContent  = msg;
  }, intervalMs);
  return timer;
}

function startBar(dotId, lblId, pctId, subId, activeCls, label) {
  document.getElementById(dotId).className    = 'load-dot ' + activeCls;
  document.getElementById(lblId).textContent  = label;
  document.getElementById(pctId).textContent  = '0%';
  document.getElementById(subId).textContent  = 'Fetching weights from CDN…';
}

function finishBar(fillId, pctId, subId, dotId, lblId, timeId, modelName, ms) {
  document.getElementById(fillId).style.width = '100%';
  document.getElementById(pctId).textContent  = '100%';
  document.getElementById(subId).textContent  = `Loaded in ${(ms / 1000).toFixed(1)}s`;
  document.getElementById(dotId).className    = 'load-dot done';
  const lbl = document.getElementById(lblId);
  lbl.textContent = modelName + ' ✓';
  lbl.classList.add('done');
  document.getElementById(timeId).textContent = ms + ' ms';
}

// ── Init all 3 models (all via mobilenet CDN — loads in seconds) ──────────────
async function init() {
  try {
    await tf.setBackend('webgl');
    await tf.ready();
    setStatus('loading', 'WebGL ready · loading 3 MobileNet variants…');

    // ── Model 1: MobileNet v1 alpha=1.0 ──────────────────────────────────────
    startBar('dot1','lbl1','pct1','sub1','a1','Loading MobileNet v1 (α1.0)…');
    const t1bar = animateBar('fill1','pct1','sub1',[
      {pct:20, msg:'Downloading weights…'},
      {pct:55, msg:'Parsing layer config…'},
      {pct:80, msg:'Compiling WebGL kernels…'},
      {pct:92, msg:'Almost ready…'},
    ], 300);
    const t1 = performance.now();
    model1 = await mobilenet.load({ version: 1, alpha: 1.0 });
    clearInterval(t1bar);
    const ms1 = Math.round(performance.now() - t1);
    finishBar('fill1','pct1','sub1','dot1','lbl1','time1','MobileNet v1 (α1.0)', ms1);

    // ── Model 2: MobileNet v2 alpha=1.0 ──────────────────────────────────────
    startBar('dot2','lbl2','pct2','sub2','a2','Loading MobileNet v2 (α1.0)…');
    const t2bar = animateBar('fill2','pct2','sub2',[
      {pct:20, msg:'Downloading weights…'},
      {pct:55, msg:'Parsing inverted residuals…'},
      {pct:80, msg:'Compiling WebGL kernels…'},
      {pct:92, msg:'Almost ready…'},
    ], 300);
    const t2 = performance.now();
    model2 = await mobilenet.load({ version: 2, alpha: 1.0 });
    clearInterval(t2bar);
    const ms2 = Math.round(performance.now() - t2);
    finishBar('fill2','pct2','sub2','dot2','lbl2','time2','MobileNet v2 (α1.0)', ms2);

    // ── Model 3: MobileNet v1 alpha=0.25 — tiny, ultra-fast ──────────────────
    startBar('dot3','lbl3','pct3','sub3','a3','Loading MobileNet v1-Lite (α0.25)…');
    const t3bar = animateBar('fill3','pct3','sub3',[
      {pct:30, msg:'Downloading slim weights…'},
      {pct:65, msg:'Compiling WebGL kernels…'},
      {pct:90, msg:'Almost ready…'},
    ], 200);
    const t3 = performance.now();
    model3 = await mobilenet.load({ version: 1, alpha: 0.25 });
    clearInterval(t3bar);
    const ms3 = Math.round(performance.now() - t3);
    finishBar('fill3','pct3','sub3','dot3','lbl3','time3','MobileNet v1-Lite (α0.25)', ms3);

    modelsReady = true;
    setStatus('ready',
      `All 3 models ready · v1 ${ms1}ms · v2 ${ms2}ms · v1-Lite ${ms3}ms`);
    setTimeout(() => { loadProgress.style.display = 'none'; }, 1500);
    if (currentImg) compareBtn.disabled = false;

  } catch (err) {
    setStatus('error', 'Load failed: ' + err.message);
    console.error(err);
  }
}

// ── Image helpers ─────────────────────────────────────────────────────────────
function loadLocal(filename, btn) {
  document.querySelectorAll('.quick-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  setPreview(`../assignment1/images/${filename}`);
}

function handleUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  document.querySelectorAll('.quick-btn').forEach(b => b.classList.remove('active'));
  const reader = new FileReader();
  reader.onload = ev => setPreview(ev.target.result);
  reader.readAsDataURL(file);
}

function setPreview(src) {
  previewImg.src = src;
  previewImg.style.display = 'block';
  document.getElementById('previewPh').style.display = 'none';
  previewImg.onload = () => {
    currentImg = previewImg;
    if (modelsReady) compareBtn.disabled = false;
  };
  ['results1','results2','results3'].forEach(id => {
    document.getElementById(id).innerHTML =
      '<div class="pred-placeholder"><span>✅</span><p>Image ready · click Compare</p></div>';
  });
  statsStrip.style.display   = 'none';
  analysisPanel.style.display = 'none';
}

// ── Run 3-way comparison ──────────────────────────────────────────────────────
async function runComparison() {
  if (!modelsReady || !currentImg) return;
  compareBtn.disabled = true;

  showSpinner('results1','sp1','MobileNet v1 (α1.0)…');
  showSpinner('results2','sp2','MobileNet v2 (α1.0)…');
  showSpinner('results3','sp3','MobileNet v1-Lite (α0.25)…');
  setStatus('loading', 'Running inference on all 3 models…');

  try {
    const ta = performance.now();
    const preds1 = await model1.classify(currentImg, 5);
    const ms1 = Math.round(performance.now() - ta);

    const tb = performance.now();
    const preds2 = await model2.classify(currentImg, 5);
    const ms2 = Math.round(performance.now() - tb);

    const tc = performance.now();
    const preds3 = await model3.classify(currentImg, 5);
    const ms3 = Math.round(performance.now() - tc);

    renderResults('results1', preds1, 'bar1', 'pct1');
    renderResults('results2', preds2, 'bar2', 'pct2');
    renderResults('results3', preds3, 'bar3', 'pct3');
    renderStats(ms1, ms2, ms3, preds1, preds2, preds3);
    renderAnalysis(ms1, ms2, ms3, preds1, preds2, preds3);

    setStatus('ready', `Done · v1 ${ms1}ms · v2 ${ms2}ms · v1-Lite ${ms3}ms`);
  } catch (err) {
    setStatus('error', err.message);
    console.error(err);
    ['results1','results2','results3'].forEach(id => {
      document.getElementById(id).innerHTML =
        `<div class="pred-placeholder"><span>❌</span><p>${err.message}</p></div>`;
    });
  }
  compareBtn.disabled = false;
}

// ── Render prediction bars ────────────────────────────────────────────────────
function renderResults(containerId, predictions, barCls, pctCls) {
  let html = '';
  predictions.forEach((pred, i) => {
    const pct   = (pred.probability * 100).toFixed(1);
    const label = pred.className.split(',')[0];
    html += `
      <div class="pred-item" style="animation-delay:${i * .1}s">
        <div class="pred-header">
          <div class="pred-label"><span class="rank-chip">${i+1}</span>${label}</div>
          <div class="${pctCls}">${pct}%</div>
        </div>
        <div class="bar-track">
          <div class="${barCls}" id="${containerId}_b${i}" style="width:0%"></div>
        </div>
      </div>`;
  });
  document.getElementById(containerId).innerHTML = html;
  requestAnimationFrame(() => {
    predictions.forEach((pred, i) => {
      setTimeout(() => {
        const b = document.getElementById(`${containerId}_b${i}`);
        if (b) b.style.width = (pred.probability * 100) + '%';
      }, i * 90);
    });
  });
}

// ── Stats strip ───────────────────────────────────────────────────────────────
function renderStats(ms1, ms2, ms3, p1, p2, p3) {
  document.getElementById('statT1').textContent = ms1 + 'ms';
  document.getElementById('statT2').textContent = ms2 + 'ms';
  document.getElementById('statT3').textContent = ms3 + 'ms';

  const top = [p1,p2,p3].map(p => p[0].className.split(',')[0].toLowerCase().trim());
  const allAgree  = top[0] === top[1] && top[1] === top[2];
  const anyAgree  = top[0]===top[1] || top[1]===top[2] || top[0]===top[2];
  document.getElementById('statAgree').textContent = allAgree ? '✓ All' : anyAgree ? '~ Partial' : '✗ None';
  document.getElementById('statAgree').style.color =
    allAgree ? 'var(--green)' : anyAgree ? '#f59e0b' : '#ef4444';

  const ranked = [{n:'v1',ms:ms1},{n:'v2',ms:ms2},{n:'v1-Lite',ms:ms3}].sort((a,b)=>a.ms-b.ms);
  document.getElementById('statFastest').textContent = ranked[0].n;

  statsStrip.style.display = 'grid';
}

// ── Analysis panel ────────────────────────────────────────────────────────────
function renderAnalysis(ms1, ms2, ms3, p1, p2, p3) {
  const ranked = [
    {n:'MobileNet v1 (α1.0)', ms:ms1},
    {n:'MobileNet v2 (α1.0)', ms:ms2},
    {n:'MobileNet v1-Lite (α0.25)', ms:ms3},
  ].sort((a,b) => a.ms - b.ms);

  document.getElementById('analysisSA').innerHTML =
    `Fastest: <span class="hi">${ranked[0].n}</span> (${ranked[0].ms}ms) → ` +
    `${ranked[1].n} (${ranked[1].ms}ms) → ${ranked[2].n} (${ranked[2].ms}ms). ` +
    `v1-Lite uses only α=0.25 of the channel width, making it ~8× smaller but less accurate.`;

  const labels = [p1,p2,p3].map(p =>
    p.slice(0,3).map(x => x.className.split(',')[0].toLowerCase().trim())
  );
  const allShared = labels[0].filter(l => labels[1].includes(l) && labels[2].includes(l));
  const shared12  = labels[0].filter(l => labels[1].includes(l));

  document.getElementById('analysisAG').innerHTML = allShared.length > 0
    ? `All 3 models agree on <span class="hi">"${allShared[0]}"</span> — strong classification confidence.`
    : shared12.length > 0
      ? `v1 &amp; v2 agree on <span class="hi">"${shared12[0]}"</span>. v1-Lite differs — expected with its reduced capacity.`
      : `All 3 give <span class="hi">different top labels</span> — the image is ambiguous or v1-Lite lacks capacity for it.`;

  const entropies = [p1,p2,p3].map(p =>
    p.reduce((s,x) => s - (x.probability > 0 ? x.probability * Math.log2(x.probability) : 0), 0).toFixed(2)
  );
  const minIdx = entropies.map(Number).indexOf(Math.min(...entropies.map(Number)));
  const names  = ['MobileNet v1','MobileNet v2','v1-Lite'];

  document.getElementById('analysisCP').innerHTML =
    `Entropy — v1: <span class="hi">${entropies[0]}</span> · ` +
    `v2: <span class="hi">${entropies[1]}</span> · ` +
    `v1-Lite: <span class="hi">${entropies[2]}</span>. ` +
    `<span class="hi">${names[minIdx]}</span> gives the most peaked (confident) distribution.`;

  analysisPanel.style.display = 'block';
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function showSpinner(containerId, spinCls, label) {
  document.getElementById(containerId).innerHTML = `
    <div class="spinner-wrap">
      <div class="spinner ${spinCls}"></div>
      <div class="spinner-text">${label}</div>
    </div>`;
}

function setStatus(type, msg) {
  document.getElementById('statusDot').className = 'status-dot ' + type;
  document.getElementById('statusText').textContent = msg;
}

// ── Boot ──────────────────────────────────────────────────────────────────────
init();
