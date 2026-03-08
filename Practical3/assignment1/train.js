'use strict';
// Assignment 1 — Sentiment Classifier Training
// Architecture: Embedding → GlobalAveragePooling → Dense(32,relu) → Dense(1,sigmoid)

// ── Dataset ───────────────────────────────────────────────────────────────────
const DATASET = [
  // Positive (label = 1)
  { text: "this movie was absolutely fantastic and i loved every moment", label: 1 },
  { text: "the food here is delicious and the service is excellent", label: 1 },
  { text: "i had a wonderful experience at this beautiful place", label: 1 },
  { text: "the performance was outstanding and truly impressive", label: 1 },
  { text: "this product is amazing and works perfectly", label: 1 },
  { text: "what a great day full of joy and happiness", label: 1 },
  { text: "the staff were very helpful and kind to everyone", label: 1 },
  { text: "i really enjoyed the show it was superb", label: 1 },
  { text: "the weather is beautiful and i feel wonderful today", label: 1 },
  { text: "excellent quality and very good value for money", label: 1 },
  { text: "this is a brilliant book that i highly recommend", label: 1 },
  { text: "the trip was unforgettable and absolutely perfect", label: 1 },
  { text: "i am so happy with my purchase it is great", label: 1 },
  { text: "the game was thrilling and we won brilliantly", label: 1 },
  { text: "such a lovely atmosphere and everyone was so friendly", label: 1 },
  { text: "the concert was spectacular and the music was amazing", label: 1 },
  { text: "i feel so positive and energized today everything is great", label: 1 },
  { text: "the design is beautiful and the functionality is perfect", label: 1 },
  { text: "best experience of my life truly remarkable and special", label: 1 },
  { text: "the teacher was wonderful and explained everything clearly", label: 1 },
  // Negative (label = 0)
  { text: "this movie was terrible and i wasted my time watching it", label: 0 },
  { text: "the food was disgusting and the service was awful", label: 0 },
  { text: "i had a horrible experience and will never return", label: 0 },
  { text: "the performance was disappointing and very poor quality", label: 0 },
  { text: "this product is broken and completely useless", label: 0 },
  { text: "what a horrible day everything went wrong and i feel sad", label: 0 },
  { text: "the staff were rude and unhelpful to customers", label: 0 },
  { text: "i did not enjoy the show at all it was boring", label: 0 },
  { text: "the weather is dreadful and i feel miserable today", label: 0 },
  { text: "very poor quality and a terrible waste of money", label: 0 },
  { text: "this is a dreadful book that i cannot recommend", label: 0 },
  { text: "the trip was a disaster and completely ruined our holiday", label: 0 },
  { text: "i am so disappointed with my purchase it is broken", label: 0 },
  { text: "the game was terrible and we lost embarrassingly", label: 0 },
  { text: "such an unpleasant atmosphere and everyone was unfriendly", label: 0 },
  { text: "the concert was awful and the sound quality was terrible", label: 0 },
  { text: "i feel so negative and drained today everything is wrong", label: 0 },
  { text: "the design is ugly and the functionality is broken", label: 0 },
  { text: "worst experience of my life truly terrible and awful", label: 0 },
  { text: "the teacher was unhelpful and explained nothing clearly", label: 0 },
];

// ── Tokeniser ─────────────────────────────────────────────────────────────────
const MAX_LEN   = 20;
const VOCAB_SIZE = 500;
let vocab = {};
let vocabSize = 0;

function buildVocab(texts) {
  const freq = {};
  texts.forEach(t => t.toLowerCase().replace(/[^a-z\s]/g,'').split(/\s+/).forEach(w => {
    if (w) freq[w] = (freq[w] || 0) + 1;
  }));
  // Sort by frequency
  const words = Object.keys(freq).sort((a,b) => freq[b]-freq[a]).slice(0, VOCAB_SIZE-2);
  vocab = { '<PAD>': 0, '<UNK>': 1 };
  words.forEach((w, i) => { vocab[w] = i + 2; });
  vocabSize = Object.keys(vocab).length;
  // Save to window for Assignment 2
  window.SENTIMENT_VOCAB = vocab;
  window.SENTIMENT_MAXLEN = MAX_LEN;
}

function tokenise(text) {
  const words = text.toLowerCase().replace(/[^a-z\s]/g,'').split(/\s+/).filter(Boolean);
  const ids = words.map(w => vocab[w] !== undefined ? vocab[w] : 1);
  // Pad / truncate to MAX_LEN
  while (ids.length < MAX_LEN) ids.push(0);
  return ids.slice(0, MAX_LEN);
}

// ── Model ─────────────────────────────────────────────────────────────────────
let model = null;

function buildModel(embDims) {
  const m = tf.sequential();
  m.add(tf.layers.embedding({
    inputDim: vocabSize,
    outputDim: embDims,
    inputLength: MAX_LEN,
    embeddingsInitializer: 'glorotUniform'
  }));
  m.add(tf.layers.globalAveragePooling1d());
  m.add(tf.layers.dense({ units: 32, activation: 'relu', kernelInitializer: 'heNormal' }));
  m.add(tf.layers.dropout({ rate: 0.3 }));
  m.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  m.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  return m;
}

// ── Chart ─────────────────────────────────────────────────────────────────────
const canvas  = document.getElementById('chartCanvas');
const ctx     = canvas.getContext('2d');
const chartW  = () => canvas.parentElement.clientWidth;
const chartH  = 220;
let chartData = { acc: [], valAcc: [], loss: [] };

function drawChart() {
  const W = chartW(), H = chartH;
  canvas.width  = W * devicePixelRatio;
  canvas.height = H * devicePixelRatio;
  canvas.style.width  = W + 'px';
  canvas.style.height = H + 'px';
  ctx.scale(devicePixelRatio, devicePixelRatio);

  const pad = { t:10, r:20, b:30, l:45 };
  const iW  = W - pad.l - pad.r;
  const iH  = H - pad.t - pad.b;
  const n   = chartData.acc.length;
  if (n < 2) return;

  ctx.clearRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = 'rgba(33,38,45,.8)';
  ctx.lineWidth = 1;
  for (let i=0; i<=4; i++) {
    const y = pad.t + (iH / 4) * i;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + iW, y); ctx.stroke();
    ctx.fillStyle = 'rgba(139,148,158,.5)';
    ctx.font = `${9 / devicePixelRatio}px Space Mono`;
    ctx.fillText((1 - i/4).toFixed(2), 2, y + 4);
  }

  const xStep = iW / (n - 1);

  function drawLine(data, color, dashed) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    if (dashed) ctx.setLineDash([4,3]); else ctx.setLineDash([]);
    data.forEach((v, i) => {
      const x = pad.l + i * xStep;
      const y = pad.t + iH * (1 - Math.min(1, Math.max(0, v)));
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  drawLine(chartData.loss,   'rgba(248,81,73,.7)', true);
  drawLine(chartData.acc,    '#39d353', false);
  drawLine(chartData.valAcc, '#58a6ff', false);

  // X-axis labels
  ctx.fillStyle = 'rgba(139,148,158,.6)';
  ctx.font = `${9/devicePixelRatio}px Space Mono`;
  [0, Math.floor(n/2), n-1].forEach(i => {
    ctx.fillText(i+1, pad.l + i*xStep - 5, H - 8);
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function setStatus(type, msg) {
  document.getElementById('sdot').className = 'sdot ' + type;
  document.getElementById('stxt').textContent = msg;
}

function log(msg, type='') {
  const box  = document.getElementById('logBox');
  const now  = new Date().toLocaleTimeString('en',{hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'});
  const line = document.createElement('div');
  line.className = 'log-line ' + type;
  line.innerHTML = `<span class="log-time">${now}</span><span class="log-msg">${msg}</span>`;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}

function addEpochRow(ep, loss, acc, valAcc) {
  const tbody = document.getElementById('epochBody');
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${ep}</td><td>${loss.toFixed(4)}</td><td>${(acc*100).toFixed(1)}%</td><td class="val-acc">${(valAcc*100).toFixed(1)}%</td>`;
  tbody.appendChild(tr);
  tr.scrollIntoView({ block: 'nearest' });
}

// ── Dataset display ───────────────────────────────────────────────────────────
function renderDataset() {
  const list = document.getElementById('datasetList');
  list.innerHTML = '';
  DATASET.forEach(d => {
    const row = document.createElement('div');
    row.className = 'ds-row';
    row.innerHTML = `<span class="ds-label ${d.label?'pos':'neg'}">${d.label?'POS':'NEG'}</span><span class="ds-text">${d.text}</span>`;
    list.appendChild(row);
  });
  document.getElementById('dsCount').textContent = `${DATASET.length} samples`;
}

// ── Training ──────────────────────────────────────────────────────────────────
async function startTraining() {
  const btn     = document.getElementById('trainBtn');
  const epochs  = Math.max(5, parseInt(document.getElementById('cfgEpochs').value) || 40);
  const batch   = Math.max(2, parseInt(document.getElementById('cfgBatch').value) || 8);
  const lr      = parseFloat(document.getElementById('cfgLR').value) || 0.005;
  const embDims = parseInt(document.getElementById('cfgEmb').value) || 16;

  btn.disabled = true;
  document.getElementById('resetBtn').disabled = true;
  chartData = { acc: [], valAcc: [], loss: [] };
  document.getElementById('epochBody').innerHTML = '';
  document.getElementById('logBox').innerHTML = '';
  setStatus('work', 'Initialising…');

  try {
    await tf.setBackend('webgl');
    await tf.ready();
    log(`Backend: ${tf.getBackend()}`, 'info');

    // Build vocab
    buildVocab(DATASET.map(d => d.text));
    log(`Vocabulary: ${vocabSize} tokens`, 'info');

    // Prepare data
    const shuffled = [...DATASET].sort(() => Math.random() - 0.5);
    const xs = shuffled.map(d => tokenise(d.text));
    const ys = shuffled.map(d => d.label);

    const xTensor = tf.tensor2d(xs, [xs.length, MAX_LEN], 'int32');
    const yTensor = tf.tensor1d(ys, 'float32');

    // Build model
    if (model) { model.dispose(); model = null; }
    model = buildModel(embDims);
    model.compile({
      optimizer: tf.train.adam(lr),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy']
    });

    const params = model.countParams();
    document.getElementById('statParams').textContent = params.toLocaleString();
    log(`Model: ${params.toLocaleString()} parameters`, 'info');
    log(`Training ${epochs} epochs, batch ${batch}, lr ${lr}`, 'info');
    setStatus('work', 'Training…');

    let bestValAcc = 0;

    await model.fit(xTensor, yTensor, {
      epochs,
      batchSize: batch,
      validationSplit: 0.2,
      shuffle: true,
      callbacks: {
        onEpochEnd: async (ep, logs) => {
          const pct = ((ep+1)/epochs*100).toFixed(0);
          document.getElementById('progFill').style.width = pct + '%';
          document.getElementById('progPct').textContent  = pct + '%';
          document.getElementById('epochLbl').textContent = `${ep+1}/${epochs}`;

          chartData.acc.push(logs.acc);
          chartData.valAcc.push(logs.val_acc);
          chartData.loss.push(logs.loss);
          drawChart();
          addEpochRow(ep+1, logs.loss, logs.acc, logs.val_acc);

          if (logs.val_acc > bestValAcc) {
            bestValAcc = logs.val_acc;
            // Save best model weights to window for Assignment 2
            window.SENTIMENT_MODEL = model;
          }
          setStatus('work', `Epoch ${ep+1}/${epochs} — val_acc: ${(logs.val_acc*100).toFixed(1)}%`);
          await tf.nextFrame();
        }
      }
    });

    xTensor.dispose(); yTensor.dispose();

    document.getElementById('statAcc').textContent   = (bestValAcc*100).toFixed(1) + '%';
    document.getElementById('statLoss').textContent  = chartData.loss[chartData.loss.length-1].toFixed(4);
    document.getElementById('statEpoch').textContent = epochs;

    window.SENTIMENT_MODEL = model;

    log(`Training complete! Best val_acc: ${(bestValAcc*100).toFixed(1)}%`, 'ok');
    setStatus('ready', `Training complete — val accuracy: ${(bestValAcc*100).toFixed(1)}%`);

  } catch(e) {
    log('Error: ' + e.message, 'err');
    setStatus('err', e.message);
    console.error(e);
  }

  btn.disabled = false;
  document.getElementById('resetBtn').disabled = false;
}

function resetAll() {
  if (model) { model.dispose(); model = null; }
  window.SENTIMENT_MODEL = null;
  chartData = { acc: [], valAcc: [], loss: [] };
  drawChart();
  document.getElementById('epochBody').innerHTML = '';
  document.getElementById('logBox').innerHTML    = '';
  document.getElementById('progFill').style.width = '0%';
  document.getElementById('progPct').textContent  = '0%';
  document.getElementById('epochLbl').textContent = '—';
  ['statAcc','statLoss','statEpoch'].forEach(id => document.getElementById(id).textContent = '—');
  setStatus('', 'Reset — ready to train again.');
  document.getElementById('trainBtn').disabled  = false;
  document.getElementById('resetBtn').disabled  = true;
  log('Model reset.', 'info');
}

// ── Init ──────────────────────────────────────────────────────────────────────
(async () => {
  renderDataset();
  await tf.setBackend('webgl');
  await tf.ready();
  setStatus('ready', `TF.js ${tf.version.tfjs} · Backend: ${tf.getBackend()}`);
  log(`TF.js ${tf.version.tfjs} ready. Backend: ${tf.getBackend()}`, 'info');
  drawChart();
})();
