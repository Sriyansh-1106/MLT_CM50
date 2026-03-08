// Assignment 2 – Webcam Transfer Learning
// MobileNet feature extractor + custom dense classifier

let mobileNet = null;
let classifier = null;
let webcamStream = null;
let webcamActive = false;
let predicting = false;
let samplesA = [], samplesB = [];
let sampleCountA = 0, sampleCountB = 0;
let trained = false;

const webcamEl = document.getElementById('webcam');
const webcamPlaceholder = document.getElementById('webcamPlaceholder');
const webcamBtn = document.getElementById('webcamBtn');
const collectABtn = document.getElementById('collectA');
const collectBBtn = document.getElementById('collectB');
const trainBtn = document.getElementById('trainBtn');
const predictBtn = document.getElementById('predictBtn');
const logBox = document.getElementById('logBox');
const progressFill = document.getElementById('progressFill');
const epochLabel = document.getElementById('epochLabel');
const livePred = document.getElementById('livePred');

// ─── Initialise ───────────────────────────────────────────────────────────────
async function init() {
  try {
    await tf.setBackend('webgl');
    await tf.ready();
    setStatus('loading', 'Loading MobileNet feature extractor…');

    mobileNet = await mobilenet.load({ version: 2, alpha: 1.0 });
    setStatus('ready', 'MobileNet loaded · Enable webcam to begin');
    console.log('MobileNet loaded');
  } catch (err) {
    setStatus('error', 'Failed to load: ' + err.message);
    console.error(err);
  }
}

// ─── Webcam toggle ────────────────────────────────────────────────────────────
async function toggleWebcam() {
  if (webcamActive) {
    stopWebcam();
    return;
  }
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 224, height: 224, facingMode: 'user' }
    });
    webcamEl.srcObject = webcamStream;
    webcamEl.style.display = 'block';
    webcamPlaceholder.style.display = 'none';
    webcamBtn.textContent = 'Disable Webcam';
    webcamActive = true;

    collectABtn.disabled = false;
    collectBBtn.disabled = false;

    setStep(2);
    setStatus('ready', 'Webcam active · Collect samples for Class A and B');
  } catch (err) {
    setStatus('error', 'Webcam error: ' + err.message);
    alert('Could not access webcam: ' + err.message);
  }
}

function stopWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
  }
  webcamEl.style.display = 'none';
  webcamPlaceholder.style.display = 'flex';
  webcamBtn.textContent = 'Enable Webcam';
  webcamActive = false;
  collectABtn.disabled = true;
  collectBBtn.disabled = true;
}

// ─── Collect a sample ─────────────────────────────────────────────────────────
function collectSample(classIdx) {
  if (!webcamActive || !mobileNet) return;

  const features = tf.tidy(() => {
    const imgTensor = tf.browser.fromPixels(webcamEl)
      .resizeBilinear([224, 224])
      .expandDims(0)
      .toFloat()
      .div(127.5)
      .sub(1);
    return mobileNet.infer(imgTensor, true); // get embedding
  });

  if (classIdx === 0) {
    samplesA.push(features);
    sampleCountA++;
    document.getElementById('countA').textContent = `${sampleCountA} sample${sampleCountA !== 1 ? 's' : ''}`;
  } else {
    samplesB.push(features);
    sampleCountB++;
    document.getElementById('countB').textContent = `${sampleCountB} sample${sampleCountB !== 1 ? 's' : ''}`;
  }

  const total = sampleCountA + sampleCountB;
  if (total >= 4 && sampleCountA >= 2 && sampleCountB >= 2) {
    trainBtn.disabled = false;
    setStep(3);
  }

  setStatus('ready', `Collected: Class A=${sampleCountA}, Class B=${sampleCountB} · Need ≥2 each to train`);
}

// ─── Build classifier head ────────────────────────────────────────────────────
function buildClassifier(inputSize) {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [inputSize],
    units: 64,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({
    units: 2,
    activation: 'softmax',
    kernelInitializer: 'varianceScaling'
  }));
  return model;
}

// ─── Start training ───────────────────────────────────────────────────────────
async function startTraining() {
  if (samplesA.length < 2 || samplesB.length < 2) {
    alert('Please collect at least 2 samples per class.');
    return;
  }

  trainBtn.disabled = true;
  predictBtn.disabled = true;

  const epochs = parseInt(document.getElementById('epochsInput').value) || 20;
  const batchSize = parseInt(document.getElementById('batchInput').value) || 16;
  const lr = parseFloat(document.getElementById('lrInput').value) || 0.001;

  clearLog();
  addLog('Building dataset…');

  try {
    // Concatenate features + labels
    const allFeatures = tf.concat([...samplesA, ...samplesB]);
    const shape = allFeatures.shape[1];

    const labelsA = tf.ones([samplesA.length]).mul(0);
    const labelsB = tf.ones([samplesB.length]);
    const allLabels = tf.concat([labelsA, labelsB]);
    const labelsOneHot = tf.oneHot(allLabels.cast('int32'), 2).toFloat();

    addLog(`Features: ${allFeatures.shape} · Labels: ${allLabels.shape}`);

    // Build model
    classifier = buildClassifier(shape);
    classifier.compile({
      optimizer: tf.train.adam(lr),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    addLog(`Model built · ${epochs} epochs, batch=${batchSize}, lr=${lr}`);
    setStatus('loading', 'Training…');

    await classifier.fit(allFeatures, labelsOneHot, {
      epochs,
      batchSize,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          const pct = ((epoch + 1) / epochs * 100).toFixed(0);
          progressFill.style.width = pct + '%';
          epochLabel.textContent = `${epoch + 1} / ${epochs}`;
          addLog(`Epoch ${epoch + 1}/${epochs} · loss: ${logs.loss.toFixed(4)} · acc: ${(logs.acc * 100).toFixed(1)}%`);
        }
      }
    });

    addLog('✅ Training complete!');
    setStatus('ready', 'Model trained · Start live prediction below');
    setStep(4);
    trained = true;
    predictBtn.disabled = false;

    // Cleanup intermediate tensors
    allFeatures.dispose();
    allLabels.dispose();
    labelsOneHot.dispose();

  } catch (err) {
    addLog('❌ Error: ' + err.message);
    setStatus('error', 'Training failed: ' + err.message);
    console.error(err);
    trainBtn.disabled = false;
  }
}

// ─── Live prediction loop ─────────────────────────────────────────────────────
let predictLoopId = null;

function togglePredicting() {
  if (predicting) {
    predicting = false;
    predictBtn.textContent = 'Start Live Prediction';
    return;
  }
  if (!trained || !webcamActive) return;
  predicting = true;
  predictBtn.textContent = 'Stop Prediction';
  predictLoop();
}

async function predictLoop() {
  if (!predicting) return;

  try {
    const probs = tf.tidy(() => {
      const imgTensor = tf.browser.fromPixels(webcamEl)
        .resizeBilinear([224, 224])
        .expandDims(0)
        .toFloat()
        .div(127.5)
        .sub(1);
      const features = mobileNet.infer(imgTensor, true);
      return classifier.predict(features);
    });

    const data = await probs.data();
    probs.dispose();

    const classIdx = data[0] > data[1] ? 0 : 1;
    const conf = Math.max(data[0], data[1]);
    const className = classIdx === 0 ? 'Class A' : 'Class B';

    livePred.innerHTML = `
      <div class="live-pred-title">Prediction</div>
      <div class="live-class">${className}</div>
      <div class="live-conf">${(conf * 100).toFixed(1)}% confidence</div>
      <div class="live-bar-track">
        <div class="live-bar-fill" style="width:${conf * 100}%"></div>
      </div>
      <div style="margin-top:0.6rem;font-size:0.65rem;color:var(--muted)">
        A: ${(data[0]*100).toFixed(1)}% · B: ${(data[1]*100).toFixed(1)}%
      </div>`;
  } catch (err) {
    console.error('Prediction error:', err);
  }

  requestAnimationFrame(predictLoop);
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function addLog(msg) {
  const div = document.createElement('div');
  div.textContent = '› ' + msg;
  logBox.appendChild(div);
  logBox.scrollTop = logBox.scrollHeight;
}

function clearLog() {
  logBox.innerHTML = '';
}

function setStep(n) {
  for (let i = 1; i <= 4; i++) {
    const el = document.getElementById(`step${i}`);
    if (!el) continue;
    el.classList.remove('active', 'done');
    if (i < n) el.classList.add('done');
    if (i === n) el.classList.add('active');
  }
}

function setStatus(type, msg) {
  document.getElementById('statusDot').className = 'status-dot ' + type;
  document.getElementById('statusText').textContent = msg;
}

// ─── Boot ─────────────────────────────────────────────────────────────────────
init();
