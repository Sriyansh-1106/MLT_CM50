// Assignment 3 – Model Comparison
// Compares MobileNet v2 alpha=1.0 (full) vs alpha=0.5 (lite) predictions

let modelFull = null;   // alpha 1.0
let modelLite = null;   // alpha 0.5
let modelsReady = false;
let currentImg = null;

const previewImg = document.getElementById('previewImg');
const previewPh = document.getElementById('previewPh');
const compareBtn = document.getElementById('compareBtn');

// ─── Init both models ─────────────────────────────────────────────────────────
async function init() {
  try {
    await tf.setBackend('webgl');
    await tf.ready();
    setStatus('loading', 'Loading MobileNet v2 (full)…');

    modelFull = await mobilenet.load({ version: 2, alpha: 1.0 });
    setStatus('loading', 'Loading MobileNet v2 (lite)…');

    modelLite = await mobilenet.load({ version: 2, alpha: 0.5 });

    modelsReady = true;
    setStatus('ready', 'Both MobileNet variants loaded · Select an image to compare');
    console.log('Both models ready');
  } catch (err) {
    setStatus('error', 'Model load failed: ' + err.message);
    console.error(err);
  }
}

// ─── Load sample from assignment1/images ─────────────────────────────────────
function loadLocal(filename) {
  const src = `../assignment1/images/${filename}`;
  setPreview(src);
}

// ─── File upload ──────────────────────────────────────────────────────────────
function handleUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => setPreview(e.target.result);
  reader.readAsDataURL(file);
}

function setPreview(src) {
  previewImg.src = src;
  previewImg.style.display = 'block';
  previewPh.style.display = 'none';
  previewImg.onload = () => {
    currentImg = previewImg;
    if (modelsReady) compareBtn.disabled = false;
  };
  // Reset results
  document.getElementById('resultsA').innerHTML = '<div class="pred-placeholder"><span>✅</span><p>Image ready · Click Run Comparison</p></div>';
  document.getElementById('resultsB').innerHTML = '<div class="pred-placeholder"><span>✅</span><p>Image ready · Click Run Comparison</p></div>';
  document.getElementById('statsPanel').style.display = 'none';
}

// ─── Run classification on both models ───────────────────────────────────────
async function runComparison() {
  if (!modelsReady || !currentImg) return;

  compareBtn.disabled = true;
  setStatus('loading', 'Running inference on both models…');

  showSpinner('resultsA', 'MobileNet Full…');
  showSpinner('resultsB', 'MobileNet Lite…');

  try {
    // Run full model
    const t0 = performance.now();
    const predsA = await modelFull.classify(currentImg, 5);
    const timeA = (performance.now() - t0).toFixed(0);

    // Run lite model
    const t1 = performance.now();
    const predsB = await modelLite.classify(currentImg, 5);
    const timeB = (performance.now() - t1).toFixed(0);

    renderResults('resultsA', predsA, 'a');
    renderResults('resultsB', predsB, 'b');
    renderStats(timeA, timeB, predsA, predsB);

    setStatus('ready', `Comparison complete · Full: ${timeA}ms · Lite: ${timeB}ms`);
  } catch (err) {
    setStatus('error', 'Comparison failed: ' + err.message);
    console.error(err);
  }

  compareBtn.disabled = false;
}

// ─── Render prediction bars ───────────────────────────────────────────────────
function renderResults(containerId, predictions, variant) {
  const pctClass = `pred-pct-${variant}`;
  const barClass = `bar-${variant}`;

  let html = '';
  predictions.forEach((pred, i) => {
    const pct = (pred.probability * 100).toFixed(1);
    const label = pred.className.split(',')[0];
    html += `
      <div class="pred-item" style="animation-delay:${i * 0.1}s">
        <div class="pred-header">
          <div class="pred-label">
            <span class="rank-chip">${i + 1}</span>${label}
          </div>
          <div class="${pctClass}">${pct}%</div>
        </div>
        <div class="bar-track">
          <div class="${barClass}" id="${containerId}_bar${i}" style="width:0%"></div>
        </div>
      </div>`;
  });

  document.getElementById(containerId).innerHTML = html;

  // Animate bars
  requestAnimationFrame(() => {
    predictions.forEach((pred, i) => {
      setTimeout(() => {
        const bar = document.getElementById(`${containerId}_bar${i}`);
        if (bar) bar.style.width = (pred.probability * 100) + '%';
      }, i * 100);
    });
  });
}

// ─── Render stat chips ────────────────────────────────────────────────────────
function renderStats(timeA, timeB, predsA, predsB) {
  document.getElementById('timeA').textContent = timeA + 'ms';
  document.getElementById('timeB').textContent = timeB + 'ms';
  document.getElementById('topA').textContent = (predsA[0].probability * 100).toFixed(0) + '%';
  document.getElementById('topB').textContent = (predsB[0].probability * 100).toFixed(0) + '%';
  document.getElementById('statsPanel').style.display = 'grid';
}

// ─── Show spinner in panel ────────────────────────────────────────────────────
function showSpinner(containerId, label) {
  document.getElementById(containerId).innerHTML = `
    <div class="spinner-wrap">
      <div class="spinner"></div>
      <div class="spinner-text">${label}</div>
    </div>`;
}

// ─── Status helpers ───────────────────────────────────────────────────────────
function setStatus(type, msg) {
  document.getElementById('statusDot').className = 'status-dot ' + type;
  document.getElementById('statusText').textContent = msg;
}

// ─── Boot ─────────────────────────────────────────────────────────────────────
init();
