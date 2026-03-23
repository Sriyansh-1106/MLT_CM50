/**
 * Assignment 1 — Keypoint & Skeleton Detection
 *
 * Detects 17 body keypoints from laptop webcam using PoseNet.
 * Draws keypoints (blue) and skeleton connections (pink) on canvas.
 * Uses lower confidence threshold (0.3) and full accuracy model.
 */

// ─── Configuration ──────────────────────────────────────────
const MIN_CONFIDENCE = 0.3;
const KEYPOINT_COLOR = '#2563eb';
const KEYPOINT_RADIUS = 7;
const SKELETON_COLOR = '#db2777';
const SKELETON_WIDTH = 3;

// ─── DOM Elements ───────────────────────────────────────────
const video    = document.getElementById('video');
const canvas   = document.getElementById('canvas');
const ctx      = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const fpsEl    = document.getElementById('fps-display');
const kpEl     = document.getElementById('kp-count');

// ─── FPS Tracking ───────────────────────────────────────────
let lastTime = performance.now();
let frameCount = 0;

/**
 * Set up the laptop webcam stream.
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
 * Load PoseNet with full accuracy (multiplier 1.0).
 */
async function loadModel() {
    return await posenet.load({
        architecture: 'MobileNetV1',
        outputStride: 16,
        multiplier: 1.0,
        inputResolution: { width: 640, height: 480 }
    });
}

/**
 * Draw a single keypoint with label.
 */
function drawKeypoint(keypoint) {
    if (keypoint.score < MIN_CONFIDENCE) return;

    const { x, y } = keypoint.position;

    // Filled circle
    ctx.beginPath();
    ctx.arc(x, y, KEYPOINT_RADIUS, 0, 2 * Math.PI);
    ctx.fillStyle = KEYPOINT_COLOR;
    ctx.fill();

    // White border for visibility
    ctx.beginPath();
    ctx.arc(x, y, KEYPOINT_RADIUS, 0, 2 * Math.PI);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Keypoint name label
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 11px DM Sans, system-ui';
    ctx.fillText(keypoint.part, x + 10, y + 4);
}

/**
 * Draw skeleton connections between adjacent keypoints.
 */
function drawSkeleton(keypoints) {
    const adjacentPairs = posenet.getAdjacentKeyPoints(keypoints, MIN_CONFIDENCE);
    adjacentPairs.forEach(([from, to]) => {
        ctx.beginPath();
        ctx.moveTo(from.position.x, from.position.y);
        ctx.lineTo(to.position.x, to.position.y);
        ctx.strokeStyle = SKELETON_COLOR;
        ctx.lineWidth = SKELETON_WIDTH;
        ctx.stroke();
    });
}

/**
 * Update FPS display every 500ms.
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
 * Main detection loop.
 */
async function detectPose(net) {
    const pose = await net.estimateSinglePose(video, { flipHorizontal: false });

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Show detected count
    const detected = pose.keypoints.filter(kp => kp.score >= MIN_CONFIDENCE).length;
    kpEl.textContent = `Detected: ${detected}/17`;

    // Draw skeleton first, then keypoints on top
    drawSkeleton(pose.keypoints);
    pose.keypoints.forEach(drawKeypoint);

    updateFPS();
    requestAnimationFrame(() => detectPose(net));
}

/**
 * Entry point.
 */
async function main() {
    try {
        statusEl.textContent = 'Accessing webcam…';
        await setupCamera();

        statusEl.textContent = 'Loading PoseNet model (this may take a moment)…';
        const net = await loadModel();

        statusEl.textContent = '✓ Model loaded — detecting poses';
        statusEl.classList.add('ready');

        detectPose(net);
    } catch (error) {
        statusEl.textContent = `Error: ${error.message}`;
        console.error('Initialization failed:', error);
    }
}

main();
