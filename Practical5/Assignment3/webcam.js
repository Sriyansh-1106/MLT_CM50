'use strict';
// Assignment 3 — Webcam Finger Digit Recognition
// Strategy: Load MobileNetV2 via tf.loadLayersModel from a stable URL,
// strip the top classification layer, use the remaining network as a
// frozen feature extractor, then train a small Dense head on webcam samples.

// ── Globals ───────────────────────────────────────────────────────────────────
let featureModel = null;   // frozen feature extractor (MobileNetV2 truncated)
let headModel    = null;   // trainable Dense classifier head
let fullModel    = null;   // combined model used for prediction
let stream       = null;
let camActive    = false;
let predicting   = false;
let predLoopId   = null;
let embSize      = 0;

const samples      = {};   // digit string → Float32Array[] of embeddings
const sampleCounts = {};
let classLabels    = [];   // sorted digit strings used in last training

const video = document.getElementById('video');

// ── Helpers ───────────────────────────────────────────────────────────────────
function setStatus(type, msg) {
  document.getElementById('sdot').className = 'sdot ' + type;
  document.getElementById('stxt').textContent = msg;
}

function markStep(n) {
  for (let i = 1; i <= 5; i++) {
    const el = document.getElementById('stp' + i);
    el.className = i < n ? 'step done' : i === n ? 'step active' : 'step';
  }
}

function markLoadDone(dotId, txtId, timeId, txt, time) {
  document.getElementById(dotId).className = 'load-dot done';
  document.getElementById(txtId).textContent = txt;
  if (timeId) document.getElementById(timeId).textContent = time;
}

function addLog(msg, color) {
  const log = document.getElementById('trainLog');
  const div = document.createElement('div');
  div.textContent = msg;
  if (color) div.style.color = color;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

// ── Step 1: Load MobileNetV2 as feature extractor ────────────────────────────
// We use the @tensorflow-models/mobilenet package but call it correctly.
// The key insight: mobilenet.load() returns an object with an .infer() method
// that accepts an image tensor and a boolean `embedding` flag.
// We test multiple approaches to find what works with the loaded CDN version.

(async () => {
  try {
    await tf.setBackend('webgl');
    await tf.ready();
    setStatus('work', 'Loading MobileNet V2…');

    const t0 = Date.now();

    // Load using the @tensorflow-models/mobilenet package
    const mn = await mobilenet.load({ version: 2, alpha: 1.0 });

    // Probe embedding size with a dummy image tensor
    const dummyImg = tf.zeros([224, 224, 3], 'int32');
    let testEmb;
    try {
      // Try embedding=true (returns penultimate layer activations)
      testEmb = mn.infer(dummyImg, true);
    } catch {
      testEmb = mn.infer(dummyImg);
    }
    embSize = testEmb.shape[testEmb.shape.length - 1];
    testEmb.dispose();
    dummyImg.dispose();

    featureModel = mn;
    const ms = Date.now() - t0;

    markLoadDone('ld1','lt1','ltime1',
      `MobileNet V2 ready — embedding: ${embSize}d`, ms + 'ms');
    setStatus('ready', 'Model loaded! Enable webcam to start collecting samples.');
    markStep(2);
    document.getElementById('camBtn').disabled = false;
    buildDigitGrid();

  } catch (e) {
    setStatus('err', 'Failed to load MobileNet: ' + e.message);
    console.error(e);
  }
})();

// ── Extract embedding from current video frame ────────────────────────────────
// Returns a plain Float32Array — no lingering tensors.
function extractEmbedding() {
  // We capture synchronously and dispose everything immediately.
  let img, resized, expanded, emb, flat;
  try {
    img      = tf.browser.fromPixels(video);                        // [H,W,3]
    resized  = tf.image.resizeBilinear(img, [224, 224]);            // [224,224,3]
    expanded = resized.toFloat().div(127.5).sub(1).expandDims(0);   // [1,224,224,3] in [-1,1]

    // infer(tensor, embedding=true) → [1, embSize]
    try {
      emb = featureModel.infer(expanded, true);
    } catch {
      emb = featureModel.infer(expanded);
    }

    flat = emb.flatten();
    const data = flat.dataSync();          // synchronous GPU→CPU copy
    return new Float32Array(data);         // detached copy
  } finally {
    // Always dispose, even if an error occurred
    if (img)      img.dispose();
    if (resized)  resized.dispose();
    if (expanded) expanded.dispose();
    if (emb)      emb.dispose();
    if (flat)     flat.dispose();
  }
}

// ── Digit collection grid ─────────────────────────────────────────────────────
function buildDigitGrid() {
  const grid = document.getElementById('digitGrid');
  grid.innerHTML = '';
  for (let d = 0; d <= 9; d++) {
    const btn = document.createElement('button');
    btn.className = 'dgt';
    btn.id = 'dgt' + d;
    btn.innerHTML = `${d}<span class="dgt-cnt" id="cnt${d}">0</span>`;
    btn.disabled = true;
    btn.addEventListener('mousedown',  () => startCapture(d));
    btn.addEventListener('mouseup',    stopCapture);
    btn.addEventListener('mouseleave', stopCapture);
    btn.addEventListener('touchstart', e => { e.preventDefault(); startCapture(d); }, { passive: false });
    btn.addEventListener('touchend',   e => { e.preventDefault(); stopCapture(); },   { passive: false });
    grid.appendChild(btn);
  }
}

let captureTimer = null;
let captureDigit = null;
let capturing    = false;

function startCapture(d) {
  if (!camActive || !featureModel || capturing) return;
  captureDigit = d;
  document.getElementById('dgt' + d).classList.add('active');
  captureTimer = setInterval(() => captureFrame(d), 250);
}

function stopCapture() {
  clearInterval(captureTimer); captureTimer = null;
  capturing = false;
  if (captureDigit !== null) {
    const el = document.getElementById('dgt' + captureDigit);
    if (el) el.classList.remove('active');
    captureDigit = null;
  }
  updateTrainBtn();
}

function captureFrame(d) {
  if (!camActive || !featureModel || video.readyState < 3) return;
  if (capturing) return;   // prevent re-entrant calls
  capturing = true;
  try {
    const emb = extractEmbedding();
    const key = String(d);
    if (!samples[key]) samples[key] = [];
    samples[key].push(emb);
    sampleCounts[key] = (sampleCounts[key] || 0) + 1;

    const cnt = document.getElementById('cnt' + d);
    if (cnt) cnt.textContent = sampleCounts[key];
    if (sampleCounts[key] >= 10)
      document.getElementById('dgt' + d).classList.add('has-data');

  } catch (e) {
    console.warn('Capture frame error:', e);
  } finally {
    capturing = false;
  }
}

// ── Webcam ────────────────────────────────────────────────────────────────────
async function toggleCam() {
  if (camActive) { stopCam(); return; }
  try {
    setStatus('work', 'Requesting webcam…');
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false
    });
    video.srcObject = stream;
    await new Promise(r => { video.onloadedmetadata = r; });
    video.play();
    // Wait for first real frame
    await new Promise(r => { video.onplaying = r; });

    document.getElementById('camPh').style.display = 'none';
    video.style.display = 'block';
    camActive = true;
    document.getElementById('camBtn').textContent = 'Disable Webcam';
    for (let d = 0; d <= 9; d++) document.getElementById('dgt' + d).disabled = false;

    markStep(3);
    setStatus('ready', 'Webcam active! Hold a digit button for 2–3s to collect samples.');
  } catch (e) {
    setStatus('err', 'Webcam error: ' + e.message);
    console.error(e);
  }
}

function stopCam() {
  stopCapture();
  stopLive();
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  video.style.display = 'none';
  document.getElementById('camPh').style.display = 'flex';
  camActive = false;
  document.getElementById('camBtn').textContent = 'Enable Webcam';
}

// ── Train button state ────────────────────────────────────────────────────────
function updateTrainBtn() {
  const eligible = Object.entries(samples).filter(([, v]) => v.length >= 5).length;
  document.getElementById('trainBtn').disabled = eligible < 2;

  const parts = Object.entries(sampleCounts)
    .filter(([, c]) => c > 0)
    .map(([d, c]) => `${d}:${c}`)
    .join('  ');
  if (parts) setStatus('ready', `Samples collected — ${parts}  (need ≥2 classes with ≥5 each)`);
}

// ── Train classifier ──────────────────────────────────────────────────────────
async function doTrain() {
  const btn    = document.getElementById('trainBtn');
  const epochs = Math.max(5, parseInt(document.getElementById('epIn').value) || 30);
  const lr     = parseFloat(document.getElementById('lrIn').value) || 0.0005;
  btn.disabled = true;
  stopLive();

  // Collect eligible classes
  classLabels = Object.entries(samples)
    .filter(([, v]) => v.length >= 5)
    .map(([k]) => k)
    .sort((a, b) => parseInt(a) - parseInt(b));

  const numClasses = classLabels.length;
  if (numClasses < 2) {
    setStatus('err', 'Need at least 2 classes with ≥5 samples each');
    btn.disabled = false; return;
  }

  const log = document.getElementById('trainLog');
  log.innerHTML = '';
  addLog(`Classes: [${classLabels.join(', ')}]`);

  try {
    // Build training arrays
    const allEmbs   = [];
    const allLabels = [];

    for (let ci = 0; ci < numClasses; ci++) {
      const key  = classLabels[ci];
      const embs = samples[key];
      addLog(`  Digit ${key}: ${embs.length} samples`);
      for (const emb of embs) {
        allEmbs.push(Array.from(emb));
        const label = new Array(numClasses).fill(0);
        label[ci] = 1;
        allLabels.push(label);
      }
    }

    addLog(`Total: ${allEmbs.length} samples, ${embSize}d features`);

    // Shuffle
    const order = allEmbs.map((_, i) => i).sort(() => Math.random() - 0.5);
    const xArr  = order.map(i => allEmbs[i]);
    const yArr  = order.map(i => allLabels[i]);

    const xs = tf.tensor2d(xArr);   // [N, embSize]
    const ys = tf.tensor2d(yArr);   // [N, numClasses]

    // Build a strong classifier head
    if (headModel) { headModel.dispose(); headModel = null; }
    headModel = buildHead(embSize, numClasses, lr);

    const params = headModel.countParams();
    addLog(`Head: ${params.toLocaleString()} params`);
    markLoadDone('ld2','lt2','ltime2',
      `Classifier (${numClasses} classes)`, params.toLocaleString() + ' params');

    document.getElementById('trProg').style.width = '0%';
    setStatus('work', `Training ${epochs} epochs…`);

    const valSplit = allEmbs.length >= 30 ? 0.15 : 0;

    await headModel.fit(xs, ys, {
      epochs,
      batchSize: Math.max(4, Math.min(16, Math.floor(allEmbs.length * 0.5))),
      shuffle: true,
      validationSplit: valSplit,
      callbacks: {
        onEpochEnd: async (ep, logs) => {
          const pct = ((ep + 1) / epochs * 100).toFixed(0);
          document.getElementById('trProg').style.width = pct + '%';
          document.getElementById('trEpLbl').textContent = `${ep + 1}/${epochs}`;
          const acc  = (logs.acc * 100).toFixed(1);
          const vacc = logs.val_acc != null ? `  val:${(logs.val_acc * 100).toFixed(1)}%` : '';
          addLog(`Ep ${String(ep+1).padStart(2)} │ loss:${logs.loss.toFixed(4)}  acc:${acc}%${vacc}`);
          await tf.nextFrame();
        }
      }
    });

    xs.dispose(); ys.dispose();

    addLog('✓ Training complete!', 'var(--sage)');
    setStatus('ready', `Classifier trained on [${classLabels.join(', ')}] — start live prediction!`);
    document.getElementById('liveBtn').disabled = false;
    document.getElementById('trEpLbl').textContent = 'Done ✓';
    markStep(5);

  } catch (e) {
    setStatus('err', 'Training failed: ' + e.message);
    addLog('Error: ' + e.message, 'var(--rust)');
    console.error(e);
  }

  btn.disabled = false;
}

// ── Classifier head architecture ──────────────────────────────────────────────
// Strong but fast: L2-regularised Dense layers + BN + Dropout
// Works well even with small datasets (20–50 samples per class)
function buildHead(inputSize, nClasses, lr) {
  const reg = tf.regularizers.l2({ l2: 1e-4 });
  const m = tf.sequential();

  // Layer 1: wide with L2 reg to prevent overfitting on small data
  m.add(tf.layers.dense({
    inputShape: [inputSize],
    units: 512,
    activation: 'relu',
    kernelInitializer: 'heNormal',
    kernelRegularizer: reg
  }));
  m.add(tf.layers.batchNormalization());
  m.add(tf.layers.dropout({ rate: 0.5 }));

  // Layer 2
  m.add(tf.layers.dense({
    units: 256,
    activation: 'relu',
    kernelInitializer: 'heNormal',
    kernelRegularizer: reg
  }));
  m.add(tf.layers.batchNormalization());
  m.add(tf.layers.dropout({ rate: 0.4 }));

  // Layer 3
  m.add(tf.layers.dense({
    units: 128,
    activation: 'relu',
    kernelInitializer: 'heNormal'
  }));
  m.add(tf.layers.dropout({ rate: 0.3 }));

  // Output
  m.add(tf.layers.dense({ units: nClasses, activation: 'softmax' }));

  m.compile({
    optimizer: tf.train.adam(lr),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  return m;
}

// ── Live prediction ───────────────────────────────────────────────────────────
function startLive() {
  if (!headModel || !camActive) return;
  predicting = true;
  document.getElementById('liveBtn').style.display = 'none';
  document.getElementById('stopBtn').style.display = 'block';
  setStatus('ready', 'Live prediction running…');
  requestAnimationFrame(predictLoop);
}

function stopLive() {
  predicting = false;
  if (predLoopId) { cancelAnimationFrame(predLoopId); predLoopId = null; }
  const lb = document.getElementById('liveBtn');
  const sb = document.getElementById('stopBtn');
  if (lb) lb.style.display = 'block';
  if (sb) sb.style.display = 'none';
}

let lastPredTime = 0;
const PRED_INTERVAL = 120; // ms between predictions (~8 FPS)

async function predictLoop(ts) {
  if (!predicting) return;

  if (ts - lastPredTime >= PRED_INTERVAL && video.readyState >= 3) {
    lastPredTime = ts;
    let embTensor, probsTensor;
    try {
      const emb  = extractEmbedding();           // Float32Array
      embTensor  = tf.tensor2d([Array.from(emb)]);  // [1, embSize]
      probsTensor = headModel.predict(embTensor);
      const probs = Array.from(probsTensor.dataSync());

      const topI  = probs.indexOf(Math.max(...probs));
      renderLive(probs, topI);
    } catch (e) {
      console.warn('Predict error:', e);
    } finally {
      if (embTensor)   embTensor.dispose();
      if (probsTensor) probsTensor.dispose();
    }
  }

  predLoopId = requestAnimationFrame(predictLoop);
}

function renderLive(probs, topI) {
  const digit = classLabels[topI];
  const conf  = (probs[topI] * 100).toFixed(1);

  const bars = classLabels.map((d, i) => {
    const p = (probs[i] * 100).toFixed(1);
    return `<div class="lb">
      <div class="lb-d">${d}</div>
      <div class="lb-track">
        <div class="lb-fill${i === topI ? ' top' : ''}" style="width:${p}%"></div>
      </div>
      <div class="lb-pct">${p}%</div>
    </div>`;
  }).join('');

  document.getElementById('liveArea').innerHTML = `
    <div class="live-big">${digit}</div>
    <div class="live-conf">${conf}% confidence</div>
    <div class="live-bars">${bars}</div>`;
}
