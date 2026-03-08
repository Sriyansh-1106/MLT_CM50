// Assignment 1 – MobileNet Image Classification
// Uses TensorFlow.js + MobileNet CDN (loaded in index.html)

let model = null;
let currentImageSrc = null;

const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const previewImg = document.getElementById('previewImg');
const previewPlaceholder = document.getElementById('previewPlaceholder');
const classifyBtn = document.getElementById('classifyBtn');
const resultsWrap = document.getElementById('resultsWrap');

// ─── Initialise TF backend + load MobileNet ───────────────────────────────────
async function init() {
  try {
    await tf.setBackend('webgl');
    await tf.ready();
    setStatus('loading', 'Loading MobileNet model…');

    model = await mobilenet.load({ version: 2, alpha: 1.0 });

    setStatus('ready', 'MobileNet ready · WebGL backend active');
    console.log('MobileNet loaded:', model);
  } catch (err) {
    setStatus('error', 'Failed to load model: ' + err.message);
    console.error(err);
  }
}

// ─── Load sample image from /images folder ────────────────────────────────────
function loadSampleImage(filename, btn) {
  // Deactivate all buttons
  document.querySelectorAll('.img-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');

  currentImageSrc = `images/${filename}`;
  setPreview(currentImageSrc);
}

// ─── Handle file upload ───────────────────────────────────────────────────────
function handleUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  // Deactivate sample buttons
  document.querySelectorAll('.img-btn').forEach(b => b.classList.remove('active'));

  const reader = new FileReader();
  reader.onload = e => {
    currentImageSrc = e.target.result;
    setPreview(currentImageSrc);
  };
  reader.readAsDataURL(file);
}

// ─── Show image preview ───────────────────────────────────────────────────────
function setPreview(src) {
  previewImg.src = src;
  previewImg.style.display = 'block';
  previewPlaceholder.style.display = 'none';

  previewImg.onload = () => {
    if (model) {
      classifyBtn.disabled = false;
    }
  };

  // Reset results
  resultsWrap.innerHTML = `
    <div class="empty-state">
      <span>✅</span>
      <p>Image loaded! Click<br/>"Classify Image" to analyse</p>
    </div>`;
}

// ─── Classify the current image ───────────────────────────────────────────────
async function classify() {
  if (!model || !currentImageSrc) return;

  classifyBtn.disabled = true;

  // Show spinner
  resultsWrap.innerHTML = `
    <div class="spinner-wrap">
      <div class="spinner"></div>
      <div class="spinner-text">Running inference…</div>
    </div>`;

  try {
    const img = previewImg;

    // Warm up if needed
    const predictions = await model.classify(img, 3);

    renderPredictions(predictions);
    setStatus('ready', `Top-3 predictions retrieved · ${predictions.length} classes matched`);
  } catch (err) {
    resultsWrap.innerHTML = `
      <div class="empty-state">
        <span>❌</span>
        <p>Classification failed:<br/>${err.message}</p>
      </div>`;
    setStatus('error', 'Classification error: ' + err.message);
    console.error(err);
  }

  classifyBtn.disabled = false;
}

// ─── Render top-3 prediction bars ────────────────────────────────────────────
function renderPredictions(predictions) {
  const rankClasses = ['', 'rank2', 'rank3'];
  const rankEmoji = ['🥇', '🥈', '🥉'];

  let html = '';

  predictions.forEach((pred, i) => {
    const pct = (pred.probability * 100).toFixed(1);
    const label = pred.className.split(',')[0]; // take first label if multiple

    html += `
      <div class="prediction-item" style="animation: slideIn 0.4s ${i * 0.12}s ease both; opacity:0;">
        <div class="prediction-header">
          <div class="prediction-label">
            <span class="rank-badge">${i + 1}</span>${label}
          </div>
          <div class="prediction-pct">${pct}%</div>
        </div>
        <div class="bar-track">
          <div class="bar-fill ${rankClasses[i]}" id="bar${i}" style="width:0%"></div>
        </div>
      </div>`;
  });

  resultsWrap.innerHTML = html;

  // Animate bars after render
  requestAnimationFrame(() => {
    predictions.forEach((pred, i) => {
      setTimeout(() => {
        const bar = document.getElementById(`bar${i}`);
        if (bar) bar.style.width = (pred.probability * 100) + '%';
      }, i * 120);
    });
  });
}

// ─── Status helpers ───────────────────────────────────────────────────────────
function setStatus(type, msg) {
  statusDot.className = 'status-dot ' + type;
  statusText.textContent = msg;
}

// ─── Start ────────────────────────────────────────────────────────────────────
init();
