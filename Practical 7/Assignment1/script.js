/**
 * Assignment 1 — Webcam Object Classification using MobileNet
 *
 * Captures frames from the laptop webcam and classifies objects
 * using the pre-trained MobileNet model (1000 ImageNet classes).
 * Displays top-3 predictions with confidence bars in real time.
 */

// ─── DOM Elements ───────────────────────────────────────────
const video       = document.getElementById('video');
const statusEl    = document.getElementById('status');
const predListEl  = document.getElementById('pred-list');
const fpsEl       = document.getElementById('fps-display');
const inferenceEl = document.getElementById('inference-display');

// ─── FPS Tracking ───────────────────────────────────────────
let lastTime = performance.now();
let frameCount = 0;

/**
 * Set up the laptop webcam stream.
 * facingMode: 'user' requests the front camera.
 */
async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false
    });
    video.srcObject = stream;

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            video.play();
            resolve(video);
        };
    });
}

/**
 * Load the MobileNet model (v2, alpha 1.0).
 * Pre-trained on ImageNet with 1000 classes.
 */
async function loadModel() {
    const model = await mobilenet.load({ version: 2, alpha: 1.0 });
    return model;
}

/**
 * Render the top-3 predictions as styled bars in the panel.
 *
 * @param {Array} predictions - Array of {className, probability}
 */
function renderPredictions(predictions) {
    const top3 = predictions.slice(0, 3);

    let html = '';
    top3.forEach((pred) => {
        const percent = (pred.probability * 100).toFixed(1);
        // Take just the first name for cleaner display
        const name = pred.className.split(',')[0].trim();

        html += `
            <div class="pred-item">
                <div class="pred-label">
                    <span class="name">${name}</span>
                    <span class="score">${percent}%</span>
                </div>
                <div class="pred-bar-bg">
                    <div class="pred-bar" style="width: ${percent}%"></div>
                </div>
            </div>
        `;
    });

    predListEl.innerHTML = html;
}

/**
 * Update the FPS counter every 500ms.
 */
function updateFPS() {
    frameCount++;
    const now = performance.now();
    if (now - lastTime >= 500) {
        const fps = Math.round((frameCount * 1000) / (now - lastTime));
        fpsEl.textContent = `FPS: ${fps}`;
        frameCount = 0;
        lastTime = now;
    }
}

/**
 * Main classification loop.
 * Classifies webcam feed continuously and updates predictions.
 *
 * @param {Object} model - Loaded MobileNet model
 */
async function classifyLoop(model) {
    // Measure inference time
    const startTime = performance.now();

    // Classify current video frame — returns top 3
    const predictions = await model.classify(video, 3);

    const inferenceMs = performance.now() - startTime;
    inferenceEl.textContent = `Inference: ${inferenceMs.toFixed(0)} ms`;

    // Render in side panel
    renderPredictions(predictions);

    // Update FPS
    updateFPS();

    // Continue loop
    requestAnimationFrame(() => classifyLoop(model));
}

/**
 * Application entry point.
 */
async function main() {
    try {
        statusEl.textContent = 'Accessing webcam…';
        await setupCamera();

        statusEl.textContent = 'Loading MobileNet model (this may take a moment)…';
        const model = await loadModel();

        statusEl.textContent = '✓ MobileNet loaded — classifying objects';
        statusEl.classList.add('ready');

        classifyLoop(model);
    } catch (error) {
        statusEl.textContent = `Error: ${error.message}`;
        console.error('Initialization failed:', error);
    }
}

main();
