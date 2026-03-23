/**
 * Assignment 2 — Label Overlay on Video Feed
 *
 * Draws the webcam feed onto a canvas and overlays MobileNet
 * classification labels with confidence bars directly on the video.
 * Top-3 predictions are shown as styled overlays on the canvas.
 */

// ─── Configuration ──────────────────────────────────────────
const OVERLAY_COLORS = ['#7c3aed', '#db2777', '#059669'];
const BAR_MAX_WIDTH = 200;
const BAR_HEIGHT = 10;

// ─── DOM Elements ───────────────────────────────────────────
const video       = document.getElementById('video');
const canvas      = document.getElementById('canvas');
const ctx         = canvas.getContext('2d');
const statusEl    = document.getElementById('status');
const fpsEl       = document.getElementById('fps-display');
const inferenceEl = document.getElementById('inference-display');
const topLabelEl  = document.getElementById('top-label');

// ─── FPS Tracking ───────────────────────────────────────────
let lastTime = performance.now();
let frameCount = 0;

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
        video.onloadedmetadata = () => {
            video.play();
            resolve(video);
        };
    });
}

/**
 * Load MobileNet v2.
 */
async function loadModel() {
    return await mobilenet.load({ version: 2, alpha: 1.0 });
}

/**
 * Draw a single prediction label with confidence bar on the canvas.
 *
 * @param {string} name       - Class name
 * @param {number} probability - Confidence 0–1
 * @param {number} index      - Prediction rank (0, 1, 2)
 * @param {string} color      - Bar and label color
 */
function drawOverlayLabel(name, probability, index, color) {
    const x = 16;
    const y = 30 + index * 55;
    const percent = (probability * 100).toFixed(1);
    const barWidth = probability * BAR_MAX_WIDTH;

    // Semi-transparent background pill
    const textWidth = ctx.measureText(`${name}  ${percent}%`).width;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    roundRect(ctx, x - 8, y - 20, Math.max(textWidth + 30, BAR_MAX_WIDTH + 20), 48, 8);
    ctx.fill();

    // Label text
    ctx.font = 'bold 16px DM Sans, system-ui';
    ctx.fillStyle = '#ffffff';
    ctx.fillText(name, x, y);

    // Percentage
    ctx.font = '500 14px DM Mono, monospace';
    ctx.fillStyle = color;
    ctx.fillText(`${percent}%`, x + ctx.measureText(name).width + 12, y);

    // Bar background
    ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
    roundRect(ctx, x, y + 8, BAR_MAX_WIDTH, BAR_HEIGHT, 4);
    ctx.fill();

    // Bar filled
    ctx.fillStyle = color;
    roundRect(ctx, x, y + 8, barWidth, BAR_HEIGHT, 4);
    ctx.fill();
}

/**
 * Utility: draw a rounded rectangle path.
 */
function roundRect(context, x, y, width, height, radius) {
    context.beginPath();
    context.moveTo(x + radius, y);
    context.lineTo(x + width - radius, y);
    context.quadraticCurveTo(x + width, y, x + width, y + radius);
    context.lineTo(x + width, y + height - radius);
    context.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    context.lineTo(x + radius, y + height);
    context.quadraticCurveTo(x, y + height, x, y + height - radius);
    context.lineTo(x, y + radius);
    context.quadraticCurveTo(x, y, x + radius, y);
    context.closePath();
}

/**
 * Update FPS counter.
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
 * Main loop: draw video to canvas, classify, overlay labels.
 *
 * @param {Object} model - Loaded MobileNet model
 */
async function classifyLoop(model) {
    // Draw current webcam frame onto the canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Classify
    const startTime = performance.now();
    const predictions = await model.classify(video, 3);
    const inferenceMs = performance.now() - startTime;

    inferenceEl.textContent = `Inference: ${inferenceMs.toFixed(0)} ms`;

    // Overlay top-3 labels on the canvas
    predictions.slice(0, 3).forEach((pred, i) => {
        const name = pred.className.split(',')[0].trim();
        drawOverlayLabel(name, pred.probability, i, OVERLAY_COLORS[i]);
    });

    // Update bottom info
    if (predictions.length > 0) {
        const topName = predictions[0].className.split(',')[0].trim();
        const topScore = (predictions[0].probability * 100).toFixed(1);
        topLabelEl.textContent = `Object: ${topName} (${topScore}%)`;
    }

    updateFPS();
    requestAnimationFrame(() => classifyLoop(model));
}

/**
 * Entry point.
 */
async function main() {
    try {
        statusEl.textContent = 'Accessing webcam…';
        await setupCamera();

        statusEl.textContent = 'Loading MobileNet model…';
        const model = await loadModel();

        statusEl.textContent = '✓ MobileNet loaded — labels overlaid on video';
        statusEl.classList.add('ready');

        classifyLoop(model);
    } catch (error) {
        statusEl.textContent = `Error: ${error.message}`;
        console.error('Initialization failed:', error);
    }
}

main();
