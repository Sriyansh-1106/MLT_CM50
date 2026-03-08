'use strict';
// Assignment 2 — Custom Sentence Testing

// ── Dataset & tokeniser (duplicated from train.js so this page is self-contained) ─
const DATASET = [
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

const MAX_LEN    = 20;
const VOCAB_SIZE = 500;
let vocab = {};
let vocabSize = 0;
let model = null;
const history = [];

function buildVocab(texts) {
  const freq = {};
  texts.forEach(t => t.toLowerCase().replace(/[^a-z\s]/g,'').split(/\s+/).forEach(w => {
    if (w) freq[w] = (freq[w]||0)+1;
  }));
  const words = Object.keys(freq).sort((a,b)=>freq[b]-freq[a]).slice(0,VOCAB_SIZE-2);
  vocab = { '<PAD>':0, '<UNK>':1 };
  words.forEach((w,i) => { vocab[w]=i+2; });
  vocabSize = Object.keys(vocab).length;
}

function tokenise(text) {
  const words = text.toLowerCase().replace(/[^a-z\s]/g,'').split(/\s+/).filter(Boolean);
  const ids   = words.map(w => vocab[w]!==undefined ? vocab[w] : 1);
  while (ids.length < MAX_LEN) ids.push(0);
  return ids.slice(0, MAX_LEN);
}

function buildModel(embDims=16) {
  buildVocab(DATASET.map(d=>d.text));
  const m = tf.sequential();
  m.add(tf.layers.embedding({ inputDim:vocabSize, outputDim:embDims, inputLength:MAX_LEN }));
  m.add(tf.layers.globalAveragePooling1d());
  m.add(tf.layers.dense({ units:32, activation:'relu', kernelInitializer:'heNormal' }));
  m.add(tf.layers.dropout({ rate:0.3 }));
  m.add(tf.layers.dense({ units:16, activation:'relu' }));
  m.add(tf.layers.dense({ units:1, activation:'sigmoid' }));
  return m;
}

// ── Status ────────────────────────────────────────────────────────────────────
function setStatus(type, msg) {
  document.getElementById('sdot').className = 'sdot '+type;
  document.getElementById('stxt').textContent = msg;
}

// ── Check for model from Assignment 1 ────────────────────────────────────────
function checkModel() {
  // Check if Assignment 1 trained a model in the same window context
  // (Only works if opened from same tab via iframe or same window object)
  if (window.opener && window.opener.SENTIMENT_MODEL) {
    model  = window.opener.SENTIMENT_MODEL;
    vocab  = window.opener.SENTIMENT_VOCAB  || vocab;
    buildVocabIfNeeded();
    onModelReady('Loaded from Assignment 1 (opener window)');
    return;
  }
  if (window.parent && window.parent.SENTIMENT_MODEL) {
    model  = window.parent.SENTIMENT_MODEL;
    vocab  = window.parent.SENTIMENT_VOCAB  || vocab;
    buildVocabIfNeeded();
    onModelReady('Loaded from Assignment 1 (parent window)');
    return;
  }
  // Fallback: try localStorage flag
  setStatus('err', 'No model found — train one in Assignment 1 or click "Train Fresh Model"');
  document.getElementById('modelInfo').innerHTML =
    'No model detected from Assignment 1. <strong>Tip:</strong> Open Assignment 1 in a <em>new tab</em> from this same page, train the model, then come back and click Check again. Or use <strong>Train Fresh Model</strong> to train instantly here.';
}

function buildVocabIfNeeded() {
  if (Object.keys(vocab).length < 2) {
    buildVocab(DATASET.map(d=>d.text));
  }
  vocabSize = Object.keys(vocab).length;
}

function onModelReady(source) {
  document.getElementById('modelInfo').innerHTML =
    `<span style="color:var(--green)">✓ Model ready</span> — ${source}. Vocabulary: ${vocabSize} tokens.`;
  document.getElementById('classifyBtn').disabled = false;
  setStatus('ready', 'Model ready — type a sentence and classify!');
}

// ── Train fresh model on this page ───────────────────────────────────────────
async function trainFresh() {
  const btn = document.getElementById('trainFreshBtn');
  btn.disabled = true;
  btn.textContent = 'Training…';
  setStatus('work', 'Training fresh model (40 epochs)…');

  try {
    await tf.setBackend('webgl'); await tf.ready();

    buildVocab(DATASET.map(d=>d.text));
    const shuffled = [...DATASET].sort(()=>Math.random()-.5);
    const xs = tf.tensor2d(shuffled.map(d=>tokenise(d.text)), [shuffled.length,MAX_LEN], 'int32');
    const ys = tf.tensor1d(shuffled.map(d=>d.label), 'float32');

    if (model) { model.dispose(); model=null; }
    model = buildModel(16);
    model.compile({ optimizer:tf.train.adam(0.005), loss:'binaryCrossentropy', metrics:['accuracy'] });

    let bestAcc = 0;
    await model.fit(xs, ys, {
      epochs:40, batchSize:8, validationSplit:.2, shuffle:true,
      callbacks:{ onEpochEnd: async(ep,logs)=>{
        if(logs.val_acc>bestAcc) bestAcc=logs.val_acc;
        btn.textContent = `Training… ${ep+1}/40`;
        setStatus('work', `Epoch ${ep+1}/40 — acc: ${(logs.acc*100).toFixed(1)}%`);
        await tf.nextFrame();
      }}
    });
    xs.dispose(); ys.dispose();

    onModelReady(`Fresh model trained (val acc: ${(bestAcc*100).toFixed(1)}%)`);
    btn.textContent = '✓ Trained!';
  } catch(e) {
    setStatus('err', e.message); console.error(e);
    btn.disabled=false; btn.textContent='⚡ Train Fresh Model';
  }
}

// ── Classify ──────────────────────────────────────────────────────────────────
async function classify() {
  if (!model) { setStatus('err','No model — train one first!'); return; }
  const text = document.getElementById('sentInput').value.trim();
  if (!text) { setStatus('err','Enter a sentence first.'); return; }

  setStatus('work','Classifying…');
  const t0 = performance.now();

  try {
    const ids = tokenise(text);
    const xt  = tf.tensor2d([ids],[1,MAX_LEN],'int32');
    const out  = model.predict(xt);
    const score= (await out.data())[0];
    xt.dispose(); out.dispose();

    const ms   = (performance.now()-t0).toFixed(1);
    const isPos= score >= 0.5;
    const conf = isPos ? score : 1-score;

    renderResult(text, isPos, score, conf, ids, ms);
    addHistory(text, isPos, conf);
    setStatus('ready', `Classified in ${ms}ms — ${isPos?'POSITIVE':'NEGATIVE'} (${(conf*100).toFixed(1)}%)`);
  } catch(e) {
    setStatus('err', e.message); console.error(e);
  }
}

function renderResult(text, isPos, score, conf, ids, ms) {
  const panel = document.getElementById('resultPanel');
  panel.className = 'result-panel ' + (isPos?'pos':'neg');

  // Show which tokens were known
  const words = text.toLowerCase().replace(/[^a-z\s]/g,'').split(/\s+/).filter(Boolean);
  const tokenHtml = words.slice(0,MAX_LEN).map(w => {
    const known = vocab[w]!==undefined && vocab[w]!==1;
    return `<span class="token ${known?'known':'unk'}" title="${known?'id:'+vocab[w]:'unknown'}">${w}</span>`;
  }).join('');

  panel.innerHTML = `
    <div class="result-verdict ${isPos?'pos':'neg'}">${isPos?'POSITIVE':'NEGATIVE'}</div>
    <div class="result-score">Confidence: <span>${(conf*100).toFixed(1)}%</span> &nbsp;·&nbsp; Raw score: <span>${score.toFixed(4)}</span> &nbsp;·&nbsp; ${ms}ms</div>
    <div class="conf-bar-wrap">
      <div class="conf-labels"><span style="color:var(--red)">Negative</span><span style="color:var(--green)">Positive</span></div>
      <div class="conf-track">
        <div class="conf-fill" id="confFill" style="width:${score*100}%;background:${isPos?'var(--green)':'var(--red)'}"></div>
        <div class="conf-pointer" style="left:${score*100}%"></div>
      </div>
    </div>
    <div style="font-size:.55rem;color:var(--muted);margin-top:.4rem;margin-bottom:.5rem">
      Token analysis — <span style="color:var(--green)">■ known</span> &nbsp; <span style="color:var(--red)">■ unknown</span>
    </div>
    <div class="token-wrap">${tokenHtml}</div>`;
}

function addHistory(text, isPos, conf) {
  history.unshift({ text, isPos, conf });
  if (history.length > 20) history.pop();
  renderHistory();
}

function renderHistory() {
  const list = document.getElementById('historyList');
  if (!history.length) { list.innerHTML='<div class="hist-empty">No classifications yet.</div>'; return; }
  list.innerHTML = history.map(h => `
    <div class="hist-row">
      <span class="hist-verdict ${h.isPos?'pos':'neg'}">${h.isPos?'POS':'NEG'}</span>
      <span class="hist-text" title="${h.text}">${h.text}</span>
      <span class="hist-conf">${(h.conf*100).toFixed(1)}%</span>
    </div>`).join('');
}

function clearHistory() { history.length=0; renderHistory(); }

function clearInput() {
  document.getElementById('sentInput').value='';
  document.getElementById('resultPanel').className='result-panel';
  document.getElementById('resultPanel').innerHTML='<div style="color:var(--muted);font-size:.7rem">Enter a sentence above and click Classify.</div>';
}

function setSample(btn) {
  document.getElementById('sentInput').value = btn.textContent;
  if (model) classify();
}

// ── Enter key shortcut ────────────────────────────────────────────────────────
document.getElementById('sentInput').addEventListener('keydown', e => {
  if (e.key==='Enter' && !e.shiftKey) { e.preventDefault(); classify(); }
});

// ── Init ──────────────────────────────────────────────────────────────────────
(async()=>{
  await tf.setBackend('webgl'); await tf.ready();
  buildVocab(DATASET.map(d=>d.text));
  setStatus('ready', `TF.js ${tf.version.tfjs} ready. Load or train a model to begin.`);
  renderHistory();
})();
