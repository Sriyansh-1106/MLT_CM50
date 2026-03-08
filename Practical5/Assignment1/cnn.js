'use strict';
// Assignment 1 — MNIST CNN vs Dense Training

const SPRITE_URL = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const LABELS_URL = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';
const IMG_SIZE   = 784;   // 28×28
const NUM_CLASSES= 10;

let cnnFinal = null, dnsFinal = null;
// Expose trained CNN globally for Assignment 2
window.TRAINED_CNN = null;

// ── Init ──────────────────────────────────────────────────────────────────────
(async () => {
  try {
    await tf.setBackend('webgl');
    await tf.ready();
    setStatus('ready', 'TF.js WebGL ready — set params and click Train');
    document.getElementById('trainBtn').disabled = false;
  } catch(e) {
    setStatus('err', 'Backend init failed: ' + e.message);
  }
})();

// ── LOAD MNIST ────────────────────────────────────────────────────────────────
async function loadMNIST(n) {
  setStatus('work', 'Fetching MNIST labels…');
  const labRes  = await fetch(LABELS_URL);
  const labBuf  = await labRes.arrayBuffer();
  const labels  = new Uint8Array(labBuf).slice(0, n * NUM_CLASSES);

  setStatus('work', 'Fetching MNIST image sprite…');
  return new Promise((res, rej) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const c   = document.createElement('canvas');
      c.width   = img.width; c.height = img.height;
      const ctx = c.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const px  = ctx.getImageData(0, 0, img.width, img.height).data;

      const total = Math.min(n, 65000);
      const flat  = new Float32Array(total * IMG_SIZE);
      for (let i = 0; i < total * IMG_SIZE; i++) flat[i] = px[i * 4] / 255;

      res({ images: flat, labels: labels.slice(0, total * NUM_CLASSES), total });
    };
    img.onerror = rej;
    img.src = SPRITE_URL;
  });
}

// ── BUILD MODELS ──────────────────────────────────────────────────────────────
function buildCNN() {
  const m = tf.sequential();
  m.add(tf.layers.conv2d({ inputShape:[28,28,1], filters:32, kernelSize:3, activation:'relu', padding:'same' }));
  m.add(tf.layers.maxPooling2d({ poolSize:2 }));
  m.add(tf.layers.conv2d({ filters:64, kernelSize:3, activation:'relu', padding:'same' }));
  m.add(tf.layers.maxPooling2d({ poolSize:2 }));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({ units:128, activation:'relu' }));
  m.add(tf.layers.dropout({ rate:.25 }));
  m.add(tf.layers.dense({ units:10, activation:'softmax' }));
  m.compile({ optimizer:tf.train.adam(1e-3), loss:'categoricalCrossentropy', metrics:['accuracy'] });
  return m;
}

function buildDense() {
  const m = tf.sequential();
  m.add(tf.layers.flatten({ inputShape:[28,28,1] }));
  m.add(tf.layers.dense({ units:256, activation:'relu' }));
  m.add(tf.layers.dense({ units:128, activation:'relu' }));
  m.add(tf.layers.dense({ units:64,  activation:'relu' }));
  m.add(tf.layers.dense({ units:10,  activation:'softmax' }));
  m.compile({ optimizer:tf.train.adam(1e-3), loss:'categoricalCrossentropy', metrics:['accuracy'] });
  return m;
}

// ── TRAIN ─────────────────────────────────────────────────────────────────────
async function startTraining() {
  const btn     = document.getElementById('trainBtn');
  const epochs  = Math.max(1, parseInt(document.getElementById('inEpochs').value)  || 5);
  const batchSz = Math.max(64, parseInt(document.getElementById('inBatch').value)  || 512);
  const samples = Math.max(1000, parseInt(document.getElementById('inSamples').value) || 6000);

  btn.disabled = true;
  document.getElementById('compareBox').style.display = 'none';
  resetCard('cnn', epochs); resetCard('dns', epochs);

  try {
    const { images, labels, total } = await loadMNIST(samples);
    const split = Math.floor(total * .8);

    const xs = tf.tensor4d(images, [total, 28, 28, 1]);
    const ys = tf.tensor2d(labels, [total, NUM_CLASSES]);

    const xTr = xs.slice([0,0,0,0],[split,-1,-1,-1]);
    const yTr = ys.slice([0,0],[split,-1]);
    const xV  = xs.slice([split,0,0,0],[-1,-1,-1,-1]);
    const yV  = ys.slice([split,0],[-1,-1]);

    setStatus('work', `Training CNN (${split} train / ${total-split} val)…`);
    const cnn = buildCNN();
    cnnFinal  = await trainModel(cnn, xTr, yTr, xV, yV, epochs, batchSz, 'cnn');
    window.TRAINED_CNN = cnn;

    setStatus('work', 'Training Dense network…');
    const dns = buildDense();
    dnsFinal  = await trainModel(dns, xTr, yTr, xV, yV, epochs, batchSz, 'dns');

    [xs,ys,xTr,yTr,xV,yV].forEach(t => t.dispose());

    showComparison(epochs);
    setStatus('ready', `Training complete ✓  CNN: ${cnnFinal.acc.toFixed(2)}%  Dense: ${dnsFinal.acc.toFixed(2)}%`);
  } catch(e) {
    setStatus('err', 'Error: ' + e.message);
    console.error(e);
  }

  btn.disabled = false;
}

async function trainModel(model, xTr, yTr, xV, yV, epochs, batchSize, id) {
  const result = { acc:0, loss:0 };
  document.getElementById(`${id}-log`).innerHTML = '';

  await model.fit(xTr, yTr, {
    epochs, batchSize,
    validationData: [xV, yV],
    callbacks: {
      onEpochEnd: async (ep, logs) => {
        const acc  = (logs.val_acc  * 100).toFixed(2);
        const loss = logs.val_loss.toFixed(4);
        const tacc = (logs.acc     * 100).toFixed(2);
        result.acc  = parseFloat(acc);
        result.loss = parseFloat(loss);

        document.getElementById(`${id}-acc`).textContent  = acc + '%';
        document.getElementById(`${id}-loss`).textContent = loss;
        document.getElementById(`${id}-prog`).style.width = ((ep+1)/epochs*100)+'%';
        document.getElementById(`${id}-ep-lbl`).textContent= `${ep+1} / ${epochs}`;
        appendLog(id, ep+1, tacc, acc, loss);
        await tf.nextFrame();
      }
    }
  });
  return result;
}

// ── COMPARISON TABLE ──────────────────────────────────────────────────────────
function showComparison(epochs) {
  document.getElementById('compareBox').style.display = 'block';
  const rows = [
    { metric:'Val Accuracy ↑', cnn: cnnFinal.acc.toFixed(2)+'%', dns: dnsFinal.acc.toFixed(2)+'%', higherWins:true,  cnnVal: cnnFinal.acc, dnsVal: dnsFinal.acc },
    { metric:'Val Loss ↓',     cnn: cnnFinal.loss.toFixed(4),    dns: dnsFinal.loss.toFixed(4),    higherWins:false, cnnVal: cnnFinal.loss, dnsVal: dnsFinal.loss },
  ];

  document.getElementById('compareBody').innerHTML = rows.map(r => {
    const cnnBetter = r.higherWins ? r.cnnVal >= r.dnsVal : r.cnnVal <= r.dnsVal;
    return `<tr>
      <td>${r.metric}</td>
      <td class="cell-a1 ${cnnBetter?'winner':''}">${r.cnn}</td>
      <td class="cell-a2 ${!cnnBetter?'winner':''}">${r.dns}</td>
      <td class="${cnnBetter?'cell-a1':'cell-a2'}">${cnnBetter?'CNN ✓':'Dense ✓'}</td>
    </tr>`;
  }).join('');

  const cnnWins = cnnFinal.acc > dnsFinal.acc;
  const diff    = Math.abs(cnnFinal.acc - dnsFinal.acc).toFixed(2);
  document.getElementById('verdict').textContent =
    `${cnnWins ? 'CNN' : 'Dense'} outperformed by ${diff}% accuracy. ` +
    `CNNs excel at spatial patterns; Dense networks treat each pixel independently. ` +
    `With more epochs and data the gap typically widens further.`;
}

// ── HELPERS ───────────────────────────────────────────────────────────────────
function resetCard(id, epochs) {
  document.getElementById(`${id}-acc`).textContent  = '—';
  document.getElementById(`${id}-loss`).textContent = '—';
  document.getElementById(`${id}-prog`).style.width = '0%';
  document.getElementById(`${id}-ep-lbl`).textContent = `0 / ${epochs}`;
  document.getElementById(`${id}-log`).innerHTML = '<span style="color:var(--muted)">Starting…</span>';
}

function appendLog(id, ep, tacc, vacc, loss) {
  const log = document.getElementById(`${id}-log`);
  if (ep === 1) log.innerHTML = '';
  const div = document.createElement('div');
  div.className = 'log-line';
  div.innerHTML = `<span class="log-ep">Ep${ep}</span>train <span class="log-good">${tacc}%</span> · val <span class="log-good">${vacc}%</span> · loss ${loss}`;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function setStatus(type, msg) {
  document.getElementById('sdot').className = 'sdot ' + type;
  document.getElementById('stxt').textContent = msg;
}
