'use strict';
// Assignment 3 — Dense vs LSTM Model Comparison

// ── Dataset ───────────────────────────────────────────────────────────────────
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

// ── Tokeniser ─────────────────────────────────────────────────────────────────
const MAX_LEN    = 20;
const VOCAB_SIZE = 500;
let vocab = {}, vocabSize = 0;

function buildVocab(texts) {
  const freq = {};
  texts.forEach(t => t.toLowerCase().replace(/[^a-z\s]/g,'').split(/\s+/).forEach(w => {
    if (w) freq[w]=(freq[w]||0)+1;
  }));
  const words = Object.keys(freq).sort((a,b)=>freq[b]-freq[a]).slice(0,VOCAB_SIZE-2);
  vocab = {'<PAD>':0,'<UNK>':1};
  words.forEach((w,i)=>{ vocab[w]=i+2; });
  vocabSize = Object.keys(vocab).length;
}

function tokenise(text) {
  const words = text.toLowerCase().replace(/[^a-z\s]/g,'').split(/\s+/).filter(Boolean);
  const ids   = words.map(w => vocab[w]!==undefined ? vocab[w] : 1);
  while (ids.length < MAX_LEN) ids.push(0);
  return ids.slice(0, MAX_LEN);
}

// ── Models ────────────────────────────────────────────────────────────────────
let denseModel = null, lstmModel = null;

function buildDenseModel(embDims) {
  const m = tf.sequential();
  m.add(tf.layers.embedding({ inputDim:vocabSize, outputDim:embDims, inputLength:MAX_LEN }));
  m.add(tf.layers.globalAveragePooling1d());
  m.add(tf.layers.dense({ units:32, activation:'relu', kernelInitializer:'heNormal' }));
  m.add(tf.layers.dropout({ rate:0.3 }));
  m.add(tf.layers.dense({ units:16, activation:'relu' }));
  m.add(tf.layers.dense({ units:1,  activation:'sigmoid' }));
  return m;
}

function buildLSTMModel(embDims) {
  const m = tf.sequential();
  m.add(tf.layers.embedding({ inputDim:vocabSize, outputDim:embDims, inputLength:MAX_LEN }));
  m.add(tf.layers.lstm({ units:32, returnSequences:false }));
  m.add(tf.layers.dropout({ rate:0.3 }));
  m.add(tf.layers.dense({ units:16, activation:'relu' }));
  m.add(tf.layers.dense({ units:1,  activation:'sigmoid' }));
  return m;
}

// ── Chart ─────────────────────────────────────────────────────────────────────
const canvas   = document.getElementById('accChart');
const ctx      = canvas.getContext('2d');
let chartData  = { dAcc:[], lAcc:[], dLoss:[] };

function drawChart() {
  const W = canvas.parentElement.clientWidth, H = 200;
  canvas.width  = W * devicePixelRatio; canvas.height = H * devicePixelRatio;
  canvas.style.width = W+'px'; canvas.style.height = H+'px';
  ctx.scale(devicePixelRatio, devicePixelRatio);

  const pad = {t:10,r:20,b:28,l:42};
  const iW=W-pad.l-pad.r, iH=H-pad.t-pad.b;
  const n = Math.max(chartData.dAcc.length, chartData.lAcc.length);
  ctx.clearRect(0,0,W,H);

  // grid
  ctx.strokeStyle='rgba(33,38,45,.8)'; ctx.lineWidth=1;
  for(let i=0;i<=4;i++){
    const y=pad.t+(iH/4)*i;
    ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(pad.l+iW,y);ctx.stroke();
    ctx.fillStyle='rgba(139,148,158,.5)';
    ctx.font=`${9/devicePixelRatio}px Space Mono`;
    ctx.fillText((1-i/4).toFixed(2), 2, y+4);
  }
  if (n < 2) return;
  const xStep = iW / (n-1);

  function line(data, color, dashed) {
    if (!data.length) return;
    ctx.beginPath(); ctx.strokeStyle=color; ctx.lineWidth=1.5;
    if(dashed) ctx.setLineDash([4,3]); else ctx.setLineDash([]);
    data.forEach((v,i)=>{
      const x=pad.l+i*xStep, y=pad.t+iH*(1-Math.min(1,Math.max(0,v)));
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    });
    ctx.stroke(); ctx.setLineDash([]);
  }

  line(chartData.dLoss, 'rgba(57,211,83,.3)', true);
  line(chartData.dAcc,  '#39d353', false);
  line(chartData.lAcc,  '#e3b341', false);

  // epoch labels
  ctx.fillStyle='rgba(139,148,158,.5)'; ctx.font=`${9/devicePixelRatio}px Space Mono`;
  [0,Math.floor(n/2),n-1].forEach(i=>{
    ctx.fillText(i+1, pad.l+i*xStep-5, H-8);
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function setStatus(type, msg) {
  document.getElementById('sdot').className='sdot '+type;
  document.getElementById('stxt').textContent=msg;
}

function miniLog(id, msg, cls='') {
  const el=document.getElementById(id);
  const d=document.createElement('div');
  if(cls) d.className=cls;
  d.textContent=msg;
  el.appendChild(d);
  el.scrollTop=el.scrollHeight;
}

function setMetric(id, val) { document.getElementById(id).textContent=val; }

// ── Training ──────────────────────────────────────────────────────────────────
async function trainBoth() {
  const btn     = document.getElementById('trainBtn');
  const epochs  = Math.max(5,  parseInt(document.getElementById('cfgEpochs').value)||40);
  const batch   = Math.max(2,  parseInt(document.getElementById('cfgBatch').value)||8);
  const lr      = parseFloat(document.getElementById('cfgLR').value)||0.005;
  const embDims = parseInt(document.getElementById('cfgEmb').value)||16;

  btn.disabled=true; document.getElementById('resetBtn').disabled=true;
  chartData={dAcc:[],lAcc:[],dLoss:[]};
  document.getElementById('d-log').innerHTML='';
  document.getElementById('l-log').innerHTML='';
  setStatus('work','Initialising…');

  await tf.setBackend('webgl'); await tf.ready();
  buildVocab(DATASET.map(d=>d.text));

  // Shared shuffled data
  const shuffled = [...DATASET].sort(()=>Math.random()-.5);
  const xArr = shuffled.map(d=>tokenise(d.text));
  const yArr = shuffled.map(d=>d.label);

  const xsTrain = tf.tensor2d(xArr,[xArr.length,MAX_LEN],'int32');
  const ysTrain = tf.tensor1d(yArr,'float32');

  // ── Train Dense ─────────────────────────────────────────────────────────────
  if (denseModel) { denseModel.dispose(); denseModel=null; }
  denseModel = buildDenseModel(embDims);
  denseModel.compile({ optimizer:tf.train.adam(lr), loss:'binaryCrossentropy', metrics:['accuracy'] });
  setMetric('d-params', denseModel.countParams().toLocaleString());
  miniLog('d-log', `Params: ${denseModel.countParams().toLocaleString()}`, 'info');

  setStatus('work','Training Dense model…');
  let dBestAcc=0, dFinalLoss=0;

  await denseModel.fit(xsTrain, ysTrain, {
    epochs, batchSize:batch, validationSplit:.2, shuffle:true,
    callbacks:{ onEpochEnd:async(ep,logs)=>{
      const pct=((ep+1)/epochs*100).toFixed(0);
      document.getElementById('d-prog').style.width=pct+'%';
      document.getElementById('d-ep').textContent=`${ep+1}/${epochs}`;
      chartData.dAcc.push(logs.val_acc);
      chartData.dLoss.push(logs.loss);
      drawChart();
      dBestAcc=Math.max(dBestAcc,logs.val_acc);
      dFinalLoss=logs.loss;
      if((ep+1)%5===0||ep===epochs-1)
        miniLog('d-log',`Ep${ep+1} loss:${logs.loss.toFixed(4)} acc:${(logs.acc*100).toFixed(1)}% val:${(logs.val_acc*100).toFixed(1)}%`,'yellow');
      setStatus('work',`Dense — Ep ${ep+1}/${epochs} val_acc:${(logs.val_acc*100).toFixed(1)}%`);
      await tf.nextFrame();
    }}
  });

  setMetric('d-acc',  (dBestAcc*100).toFixed(1)+'%');
  setMetric('d-loss', dFinalLoss.toFixed(4));
  miniLog('d-log','✓ Done!','ok');

  // Measure Dense inference speed (average 20 runs)
  const dSpeed = await measureSpeed(denseModel);
  setMetric('d-speed', dSpeed.toFixed(2));

  // ── Train LSTM ──────────────────────────────────────────────────────────────
  if (lstmModel) { lstmModel.dispose(); lstmModel=null; }
  lstmModel = buildLSTMModel(embDims);
  lstmModel.compile({ optimizer:tf.train.adam(lr), loss:'binaryCrossentropy', metrics:['accuracy'] });
  setMetric('l-params', lstmModel.countParams().toLocaleString());
  miniLog('l-log',`Params: ${lstmModel.countParams().toLocaleString()}`,'info');

  setStatus('work','Training LSTM model…');
  let lBestAcc=0, lFinalLoss=0;

  await lstmModel.fit(xsTrain, ysTrain, {
    epochs, batchSize:batch, validationSplit:.2, shuffle:true,
    callbacks:{ onEpochEnd:async(ep,logs)=>{
      const pct=((ep+1)/epochs*100).toFixed(0);
      document.getElementById('l-prog').style.width=pct+'%';
      document.getElementById('l-ep').textContent=`${ep+1}/${epochs}`;
      chartData.lAcc.push(logs.val_acc);
      drawChart();
      lBestAcc=Math.max(lBestAcc,logs.val_acc);
      lFinalLoss=logs.loss;
      if((ep+1)%5===0||ep===epochs-1)
        miniLog('l-log',`Ep${ep+1} loss:${logs.loss.toFixed(4)} acc:${(logs.acc*100).toFixed(1)}% val:${(logs.val_acc*100).toFixed(1)}%`,'yellow');
      setStatus('work',`LSTM — Ep ${ep+1}/${epochs} val_acc:${(logs.val_acc*100).toFixed(1)}%`);
      await tf.nextFrame();
    }}
  });

  xsTrain.dispose(); ysTrain.dispose();

  setMetric('l-acc',  (lBestAcc*100).toFixed(1)+'%');
  setMetric('l-loss', lFinalLoss.toFixed(4));
  miniLog('l-log','✓ Done!','ok');

  const lSpeed = await measureSpeed(lstmModel);
  setMetric('l-speed', lSpeed.toFixed(2));

  // ── Populate comparison table ───────────────────────────────────────────────
  buildComparisonTable(dBestAcc, dFinalLoss, dSpeed, denseModel.countParams(),
                       lBestAcc, lFinalLoss, lSpeed,  lstmModel.countParams());

  document.getElementById('predBtn').disabled=false;
  document.getElementById('resetBtn').disabled=false;
  setStatus('ready',`Training complete! Dense val:${(dBestAcc*100).toFixed(1)}% | LSTM val:${(lBestAcc*100).toFixed(1)}%`);
  miniLog('d-log',`Best val_acc: ${(dBestAcc*100).toFixed(1)}%`,'ok');
  miniLog('l-log',`Best val_acc: ${(lBestAcc*100).toFixed(1)}%`,'ok');
}

async function measureSpeed(model) {
  // Warm up
  const dummy = tf.zeros([1,MAX_LEN],'int32');
  model.predict(dummy).dispose(); dummy.dispose();
  // Time 20 inferences
  const runs = 20;
  const t0 = performance.now();
  for (let i=0;i<runs;i++) {
    const t = tf.zeros([1,MAX_LEN],'int32');
    const out = model.predict(t);
    out.dataSync(); // force sync
    t.dispose(); out.dispose();
  }
  return (performance.now()-t0)/runs;
}

function buildComparisonTable(dAcc, dLoss, dSpeed, dParams, lAcc, lLoss, lSpeed, lParams) {
  const tbody = document.getElementById('cmpBody');
  const rows = [
    {
      metric: 'Val Accuracy',
      d: (dAcc*100).toFixed(1)+'%',
      l: (lAcc*100).toFixed(1)+'%',
      winner: dAcc >= lAcc ? 'dense' : 'lstm',
      higher: true
    },
    {
      metric: 'Final Loss',
      d: dLoss.toFixed(4),
      l: lLoss.toFixed(4),
      winner: dLoss <= lLoss ? 'dense' : 'lstm',
      higher: false
    },
    {
      metric: 'Inference Speed',
      d: dSpeed.toFixed(2)+'ms',
      l: lSpeed.toFixed(2)+'ms',
      winner: dSpeed <= lSpeed ? 'dense' : 'lstm',
      higher: false
    },
    {
      metric: 'Parameters',
      d: dParams.toLocaleString(),
      l: lParams.toLocaleString(),
      winner: dParams <= lParams ? 'dense' : 'lstm',
      higher: false
    }
  ];

  tbody.innerHTML = rows.map(r=>`
    <tr>
      <td>${r.metric}</td>
      <td class="${r.winner==='dense'?'winner':''}">${r.d}${r.winner==='dense'?`<span class="badge dense">winner</span>`:''}</td>
      <td class="${r.winner==='lstm'?'winner':''}">${r.l}${r.winner==='lstm'?`<span class="badge lstm">winner</span>`:''}</td>
      <td>${r.winner==='dense'?'<span class="badge dense">Dense</span>':'<span class="badge lstm">LSTM</span>'}</td>
    </tr>`).join('');
}

// ── Side-by-side prediction ───────────────────────────────────────────────────
async function predictBoth() {
  if (!denseModel || !lstmModel) { setStatus('err','Train models first'); return; }
  const text = document.getElementById('predInput').value.trim();
  if (!text) { setStatus('err','Enter a sentence'); return; }

  const ids = tokenise(text);
  const xt  = tf.tensor2d([ids],[1,MAX_LEN],'int32');

  // Dense
  const t0d   = performance.now();
  const dOut  = denseModel.predict(xt);
  const dScore= (await dOut.data())[0];
  const dMs   = (performance.now()-t0d).toFixed(2);
  dOut.dispose();

  // LSTM
  const t0l   = performance.now();
  const lOut  = lstmModel.predict(xt);
  const lScore= (await lOut.data())[0];
  const lMs   = (performance.now()-t0l).toFixed(2);
  lOut.dispose();

  xt.dispose();

  function renderBox(boxId, verdictId, confId, speedId, barId, score, ms) {
    const isPos = score >= 0.5;
    const conf  = isPos ? score : 1-score;
    document.getElementById(boxId).className = 'pred-box '+(isPos?'pos':'neg');
    const vEl = document.getElementById(verdictId);
    vEl.className = 'pred-verdict '+(isPos?'pos':'neg');
    vEl.textContent = isPos?'POSITIVE':'NEGATIVE';
    document.getElementById(confId).textContent  = `Confidence: ${(conf*100).toFixed(1)}%  ·  Score: ${score.toFixed(4)}`;
    document.getElementById(speedId).textContent = `${ms}ms inference`;
    const bar = document.getElementById(barId);
    bar.style.width      = (conf*100)+'%';
    bar.style.background = isPos ? 'var(--green)' : 'var(--red)';
  }

  renderBox('dPredBox','dVerdict','dConf','dSpeed','dBar', dScore, dMs);
  renderBox('lPredBox','lVerdict','lConf','lSpeed','lBar', lScore, lMs);

  const agree = (dScore>=.5) === (lScore>=.5);
  setStatus('ready', agree
    ? `Both models agree: ${dScore>=.5?'POSITIVE':'NEGATIVE'}`
    : `Models disagree — Dense: ${dScore>=.5?'POS':'NEG'}, LSTM: ${lScore>=.5?'POS':'NEG'}`);
}

function resetAll() {
  if (denseModel) { denseModel.dispose(); denseModel=null; }
  if (lstmModel)  { lstmModel.dispose();  lstmModel=null; }
  chartData={dAcc:[],lAcc:[],dLoss:[]};
  drawChart();
  ['d-acc','d-loss','d-speed','d-params','l-acc','l-loss','l-speed','l-params',
   'd-ep','l-ep'].forEach(id=>setMetric(id,'—'));
  document.getElementById('d-prog').style.width='0%';
  document.getElementById('l-prog').style.width='0%';
  document.getElementById('d-log').innerHTML='Waiting to train…';
  document.getElementById('l-log').innerHTML='Waiting to train…';
  document.getElementById('cmpBody').innerHTML='<tr><td colspan="4" style="color:var(--muted);text-align:center;padding:1.5rem">Train both models to see comparison.</td></tr>';
  document.getElementById('predBtn').disabled=true;
  document.getElementById('trainBtn').disabled=false;
  document.getElementById('resetBtn').disabled=true;
  setStatus('','Reset — ready to train again.');
}

// ── Enter key ─────────────────────────────────────────────────────────────────
document.getElementById('predInput').addEventListener('keydown', e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();predictBoth();}
});

// ── Init ──────────────────────────────────────────────────────────────────────
(async()=>{
  await tf.setBackend('webgl'); await tf.ready();
  buildVocab(DATASET.map(d=>d.text));
  setStatus('ready',`TF.js ${tf.version.tfjs} · ${tf.getBackend()} backend ready`);
  drawChart();
})();
