/**
 * Assignment 1 — Keypoint & Skeleton Detection
 *
 * Detects 17 body keypoints from laptop webcam using PoseNet.
 * Draws keypoints (blue) and skeleton connections (pink) on canvas.
 * Uses lower confidence threshold (0.3) for better detection.
 * Uses higher-accuracy model settings for reliable results.
 */

// ─── Configuration ──────────────────────────────────────────
const MIN_CONFIDENCE = 0.3;          // Lowered for better detection
const KEYPOINT_COLOR = '#2563eb';    // Blue for keypoints
const KEYPOINT_RADIUS = 7;
const SKELETON_COLOR = '#db2777';    // Pink for skeleton lines
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
 * Set up the laptop webcam stream using getUserMedia.
 * Uses facingMode: 'user' to request front camera.
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
 * Load PoseNet model with higher accuracy settings.
 * Using multiplier 1.0 and outputStride 16 for best detection.
 */
async function loadModel() {
    const net = await posenet.load({
        architecture: 'MobileNetV1',
        outputStride: 16,
        multiplier: 1.0,           // Full accuracy (was 0.75)
        inputResolution: { width: 640, height: 480 }
    });
    return net;
}

/**
 * Draw a single keypoint as a filled circle with label.
 * Skips any keypoint below confidence threshold.
 *
 * @param {Object} keypoint - PoseNet keypoint with position {x, y} and score
 */
function drawKeypoint(keypoint) {
    if (keypoint.score < MIN_CONFIDENCE) return;

    const { x, y } = keypoint.position;

    // Filled inner circle
    ctx.beginPath();
    ctx.arc(x, y, KEYPOINT_RADIUS, 0, 2 * Math.PI);
    ctx.fillStyle = KEYPOINT_COLOR;
    ctx.fill();

    // White border for visibility on dark backgrounds
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
 * Only draws if both keypoints exceed confidence threshold.
 *
 * @param {Array} keypoints - Array of 17 PoseNet keypoints
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
 * Estimates pose from webcam and draws keypoints + skeleton.
 *
 * @param {Object} net - Loaded PoseNet model
 */
async function detectPose(net) {
    // Estimate a single pose from the current video frame
    const pose = await net.estimateSinglePose(video, { flipHorizontal: false });

    // Clear previous frame
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Count how many keypoints are detected above threshold
    const detected = pose.keypoints.filter(kp => kp.score >= MIN_CONFIDENCE).length;
    kpEl.textContent = `Detected: ${detected}/17`;

    // Draw skeleton first (so keypoints render on top)
    drawSkeleton(pose.keypoints);

    // Draw each keypoint with label
    pose.keypoints.forEach(drawKeypoint);

    // Update FPS
    updateFPS();

    // Continue loop
    requestAnimationFrame(() => detectPose(net));
}

/**
 * Application entry point.
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
