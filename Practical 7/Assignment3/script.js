/**
 * Assignment 3 — FPS & Performance Analysis + Model Save/Reload
 *
 * Measures real-time inference performance (FPS, per-frame time, average).
 * Demonstrates deploying a model in the browser:
 *   1. Save the MobileNet model to browser IndexedDB
 *   2. Reload the saved model from IndexedDB
 *   3. Run predictions using the locally-saved model
 * Logs all performance data to console for analysis.
 */

// ─── DOM Elements ───────────────────────────────────────────
const video         = document.getElementById('video');
const statusEl      = document.getElementById('status');
const fpsValEl      = document.getElementById('fps-val');
const inferValEl    = document.getElementById('infer-val');
const avgValEl      = document.getElementById('avg-val');
const framesValEl   = document.getElementById('frames-val');
const currentPredEl = document.getElementById('current-pred');
const btnSave       = document.getElementById('btn-save');
const btnLoad       = document.getElementById('btn-load');
const modelStatusEl = document.getElementById('model-status');

// ─── State ──────────────────────────────────────────────────
let totalFrames = 0;
let totalInferenceTime = 0;
let inferenceTimes = [];        // Last 60 frame times for rolling stats
let currentModel = null;        // Currently active model
let modelSource = 'CDN';        // Track which model is active
let isRunning = false;

// ─── FPS Tracking ───────────────────────────────────────────
let fpsFrameCount = 0;
let fpsLastTime = performance.now();

// Path to save model in browser IndexedDB
const MODEL_SAVE_PATH = 'indexeddb://mobilenet-local';

/**
 * Set up the laptop webcam.
 */
async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false
    });
    video.srcObject = stream;
    return new Promise((resolve) => {
        video.onloadedmetadata = () => { video.play(); resolve(video); };
    });
}

/**
 * Load MobileNet from CDN.
 */
async function loadModelFromCDN() {
    return await mobilenet.load({ version: 2, alpha: 1.0 });
}

/**
 * Update FPS display every 500ms.
 */
function updateFPS() {
    fpsFrameCount++;
    const now = performance.now();
    if (now - fpsLastTime >= 500) {
        const fps = Math.round((fpsFrameCount * 1000) / (now - fpsLastTime));
        fpsValEl.textContent = fps;
        fpsFrameCount = 0;
        fpsLastTime = now;
    }
}

/**
 * Track inference performance and log to console every 60 frames.
 *
 * @param {number} ms - Inference time for this frame in milliseconds
 */
function trackPerformance(ms) {
    totalFrames++;
    totalInferenceTime += ms;
    inferenceTimes.push(ms);
    if (inferenceTimes.length > 60) inferenceTimes.shift();

    // Update on-screen metrics
    inferValEl.textContent = `${ms.toFixed(0)} ms`;
    framesValEl.textContent = totalFrames;

    // Rolling average of last 60 frames
    const avg = inferenceTimes.reduce((a, b) => a + b, 0) / inferenceTimes.length;
    avgValEl.textContent = `${avg.toFixed(1)} ms`;

    // Console log every 60 frames
    if (totalFrames % 60 === 0) {
        const min = Math.min(...inferenceTimes).toFixed(1);
        const max = Math.max(...inferenceTimes).toFixed(1);
        console.log(
            `%c[${modelSource}] Frame ${totalFrames} — avg: ${avg.toFixed(1)}ms | min: ${min}ms | max: ${max}ms`,
            'color: #7c3aed; font-weight: bold'
        );
    }
}

/**
 * Main classification loop using whichever model is currently active.
 */
async function classifyLoop() {
    if (!currentModel || !isRunning) return;

    const startTime = performance.now();

    // Classify the current video frame
    const predictions = await currentModel.classify(video, 3);

    const inferenceMs = performance.now() - startTime;

    // Update prediction display
    if (predictions.length > 0) {
        const top = predictions[0];
        const name = top.className.split(',')[0].trim();
        const score = (top.probability * 100).toFixed(1);
        currentPredEl.innerHTML = `${name} <span>${score}%</span>`;
    }

    // Track performance
    trackPerformance(inferenceMs);
    updateFPS();

    // Continue loop
    requestAnimationFrame(classifyLoop);
}

// ─── Save Model to IndexedDB ────────────────────────────────
btnSave.addEventListener('click', async () => {
    if (!currentModel) return;

    btnSave.disabled = true;
    modelStatusEl.textContent = 'Saving model to IndexedDB…';
    modelStatusEl.className = '';

    try {
        // Access the underlying tf.GraphModel from MobileNet wrapper
        const underlyingModel = currentModel.model || currentModel;

        if (underlyingModel.save) {
            await underlyingModel.save(MODEL_SAVE_PATH);
            modelStatusEl.textContent = '✓ Model saved to IndexedDB successfully!';
            modelStatusEl.className = 'success';
            btnLoad.disabled = false;

            console.log('%c✓ Model saved to IndexedDB', 'color: #059669; font-weight: bold');
            console.log(`  Path: ${MODEL_SAVE_PATH}`);
        } else {
            throw new Error('Cannot access internal model for saving');
        }
    } catch (error) {
        modelStatusEl.textContent = `Save failed: ${error.message}`;
        modelStatusEl.className = 'error';
        console.error('Model save error:', error);
    }

    btnSave.disabled = false;
});

// ─── Load Model from IndexedDB ──────────────────────────────
btnLoad.addEventListener('click', async () => {
    btnLoad.disabled = true;
    modelStatusEl.textContent = 'Loading model from IndexedDB…';
    modelStatusEl.className = '';

    try {
        const loadStart = performance.now();

        // Load the raw tf.GraphModel from IndexedDB
        const savedModel = await tf.loadGraphModel(MODEL_SAVE_PATH);

        const loadTime = (performance.now() - loadStart).toFixed(0);

        // Create a lightweight wrapper that mimics mobilenet.classify()
        const savedClassifier = {
            model: savedModel,
            classify: async (input, topK = 3) => {
                // MobileNet v2 expects 224x224 input, normalized to [-1, 1]
                const tensor = tf.browser.fromPixels(input)
                    .resizeBilinear([224, 224])
                    .expandDims(0)
                    .toFloat()
                    .div(127.5)
                    .sub(1.0);

                const logits = savedModel.predict(tensor);
                const probabilities = tf.softmax(logits);
                const values = await probabilities.data();

                // Clean up GPU memory
                tensor.dispose();
                logits.dispose();
                probabilities.dispose();

                // Get top-K predictions
                const topIndices = Array.from(values)
                    .map((val, idx) => ({ idx, val }))
                    .sort((a, b) => b.val - a.val)
                    .slice(0, topK);

                return topIndices.map(({ idx, val }) => ({
                    className: IMAGENET_LABELS[idx] || `Class ${idx}`,
                    probability: val
                }));
            }
        };

        // Stop current loop, switch model, restart
        isRunning = false;
        await new Promise(r => setTimeout(r, 100));

        currentModel = savedClassifier;
        modelSource = 'IndexedDB';

        // Reset counters for fair comparison
        totalFrames = 0;
        totalInferenceTime = 0;
        inferenceTimes = [];

        isRunning = true;
        classifyLoop();

        modelStatusEl.textContent = `✓ Loaded from IndexedDB in ${loadTime}ms — predicting!`;
        modelStatusEl.className = 'success';

        console.log('%c✓ Model loaded from IndexedDB', 'color: #059669; font-weight: bold');
        console.log(`  Load time: ${loadTime}ms`);
        console.log('  Performance counters reset for comparison.');

    } catch (error) {
        modelStatusEl.textContent = `Load failed: ${error.message}. Save the model first.`;
        modelStatusEl.className = 'error';
        console.error('Model load error:', error);
    }

    btnLoad.disabled = false;
});

// ─── ImageNet Labels ────────────────────────────────────────
// Full 1000-class ImageNet labels fetched at startup
const IMAGENET_LABELS = {};

(async function loadLabels() {
    try {
        const res = await fetch('https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json');
        const labels = await res.json();
        labels.forEach((label, i) => { IMAGENET_LABELS[i] = label; });
        console.log('✓ ImageNet labels loaded (1000 classes)');
    } catch (e) {
        console.warn('Could not load ImageNet labels, using class indices instead');
    }
})();

/**
 * Entry point.
 */
async function main() {
    try {
        statusEl.textContent = 'Accessing webcam…';
        await setupCamera();

        statusEl.textContent = 'Loading MobileNet from CDN…';
        currentModel = await loadModelFromCDN();
        modelSource = 'CDN';

        statusEl.textContent = '✓ MobileNet loaded — measuring performance';
        statusEl.classList.add('ready');

        console.log('%c══════════════════════════════════════════', 'color: #7c3aed');
        console.log('%c  MobileNet Performance Analysis Logger   ', 'color: #7c3aed; font-weight: bold');
        console.log('%c══════════════════════════════════════════', 'color: #7c3aed');
        console.log('Stats logged every 60 frames.');
        console.log('Save → Load model to compare CDN vs IndexedDB performance.\n');

        isRunning = true;
        classifyLoop();

    } catch (error) {
        statusEl.textContent = `Error: ${error.message}`;
        console.error('Initialization failed:', error);
    }
}

main();
