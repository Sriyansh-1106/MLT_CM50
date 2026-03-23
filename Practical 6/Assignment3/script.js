/**
 * Assignment 3 — Single-Pose vs Multi-Pose Detection Comparison
 *
 * Switch between single-pose and multi-pose modes via laptop webcam.
 * Displays detected persons count, inference time, and FPS.
 * Logs performance comparison to browser console.
 * Each person gets a unique color.
 */

// ─── Configuration ──────────────────────────────────────────
const MIN_CONFIDENCE = 0.3;
const MAX_POSES = 5;
const MIN_POSE_CONFIDENCE = 0.15;

const PERSON_COLORS = [
    { keypoint: '#2563eb', skeleton: '#1d4ed8' },
    { keypoint: '#db2777', skeleton: '#be185d' },
    { keypoint: '#059669', skeleton: '#047857' },
    { keypoint: '#d97706', skeleton: '#b45309' },
    { keypoint: '#7c3aed', skeleton: '#6d28d9' },
];

// ─── State ──────────────────────────────────────────────────
let detectionMode   = 'single';
let frameCount      = 0;
let lastFpsTime     = performance.now();
let singlePoseTimes = [];
let multiPoseTimes  = [];

// ─── DOM Elements ───────────────────────────────────────────
const video       = document.getElementById('video');
const canvas      = document.getElementById('canvas');
const ctx         = canvas.getContext('2d');
const statusEl    = document.getElementById('status');
const modeEl      = document.getElementById('mode-display');
const personEl    = document.getElementById('person-count');
const inferenceEl = document.getElementById('inference-time');
const fpsEl       = document.getElementById('fps-display');
const btnSingle   = document.getElementById('btn-single');
const btnMulti    = document.getElementById('btn-multi');

// ─── Mode Toggle ────────────────────────────────────────────
btnSingle.addEventListener('click', () => {
    detectionMode = 'single';
    modeEl.textContent = 'Single';
    btnSingle.classList.add('active');
    btnMulti.classList.remove('active');
    logModeSwitch('single');
});

btnMulti.addEventListener('click', () => {
    detectionMode = 'multi';
    modeEl.textContent = 'Multi';
    btnMulti.classList.add('active');
    btnSingle.classList.remove('active');
    logModeSwitch('multi');
});

/**
 * Log mode switch and performance comparison.
 */
function logModeSwitch(mode) {
    console.log(`%c═══ Switched to ${mode.toUpperCase()}-POSE mode ═══`, 'color: #2563eb; font-weight: bold');

    if (singlePoseTimes.length >= 5 && multiPoseTimes.length >= 5) {
        const avgSingle = (singlePoseTimes.reduce((a, b) => a + b, 0) / singlePoseTimes.length).toFixed(1);
        const avgMulti  = (multiPoseTimes.reduce((a, b) => a + b, 0) / multiPoseTimes.length).toFixed(1);
        console.log('%c── Performance Comparison ──', 'color: #059669; font-weight: bold');
        console.log(`  Single-pose avg: ${avgSingle} ms`);
        console.log(`  Multi-pose  avg: ${avgMulti} ms`);
        console.log(`  Multi-pose is ${(avgMulti - avgSingle).toFixed(1)} ms slower (${((avgMulti / avgSingle) * 100 - 100).toFixed(0)}%)`);
    }
}

/**
 * Set up laptop webcam.
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
 * Load PoseNet with full accuracy.
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
 * Draw one detected pose with unique colors.
 */
function drawPose(pose, colors, personIndex) {
    const keypoints = pose.keypoints;

    // Skeleton
    const pairs = posenet.getAdjacentKeyPoints(keypoints, MIN_CONFIDENCE);
    pairs.forEach(([from, to]) => {
        ctx.beginPath();
        ctx.moveTo(from.position.x, from.position.y);
        ctx.lineTo(to.position.x, to.position.y);
        ctx.strokeStyle = colors.skeleton;
        ctx.lineWidth = 3;
        ctx.stroke();
    });

    // Keypoints
    keypoints.forEach(kp => {
        if (kp.score < MIN_CONFIDENCE) return;
        ctx.beginPath();
        ctx.arc(kp.position.x, kp.position.y, 6, 0, 2 * Math.PI);
        ctx.fillStyle = colors.keypoint;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
    });

    // Person label in multi-pose mode
    if (detectionMode === 'multi') {
        const nose = keypoints.find(k => k.part === 'nose');
        if (nose && nose.score >= MIN_CONFIDENCE) {
            ctx.fillStyle = colors.keypoint;
            ctx.font = 'bold 16px DM Sans, system-ui';
            ctx.fillText(`Person ${personIndex + 1}`, nose.position.x - 35, nose.position.y - 20);
        }
    }
}

/**
 * Update FPS every 500ms.
 */
function updateFPS() {
    frameCount++;
    const now = performance.now();
    if (now - lastFpsTime >= 500) {
        const fps = Math.round((frameCount * 1000) / (now - lastFpsTime));
        fpsEl.textContent = fps;
        frameCount = 0;
        lastFpsTime = now;
    }
}

/**
 * Track and log inference time every 30 frames.
 */
function trackInferenceTime(ms) {
    const bucket = detectionMode === 'single' ? singlePoseTimes : multiPoseTimes;
    bucket.push(ms);
    if (bucket.length > 30) bucket.shift();

    if (bucket.length % 30 === 0) {
        const avg = (bucket.reduce((a, b) => a + b, 0) / bucket.length).toFixed(1);
        const min = Math.min(...bucket).toFixed(1);
        const max = Math.max(...bucket).toFixed(1);
        console.log(
            `%c[${detectionMode.toUpperCase()}] avg: ${avg}ms | min: ${min}ms | max: ${max}ms`,
            detectionMode === 'single' ? 'color: #2563eb' : 'color: #db2777'
        );
    }
}

/**
 * Main detection loop.
 */
async function detectPose(net) {
    const startTime = performance.now();
    let poses = [];

    if (detectionMode === 'single') {
        const pose = await net.estimateSinglePose(video, { flipHorizontal: false });
        if (pose.score > MIN_POSE_CONFIDENCE) poses = [pose];
    } else {
        poses = await net.estimateMultiplePoses(video, {
            flipHorizontal: false,
            maxDetections: MAX_POSES,
            scoreThreshold: MIN_CONFIDENCE,
            nmsRadius: 20
        });
        poses = poses.filter(p => p.score > MIN_POSE_CONFIDENCE);
    }

    const inferenceMs = performance.now() - startTime;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    poses.forEach((pose, i) => {
        drawPose(pose, PERSON_COLORS[i % PERSON_COLORS.length], i);
    });

    personEl.textContent    = poses.length;
    inferenceEl.textContent = `${inferenceMs.toFixed(0)} ms`;

    trackInferenceTime(inferenceMs);
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

        statusEl.textContent = 'Loading PoseNet model…';
        const net = await loadModel();

        statusEl.textContent = '✓ Ready — switch modes to compare';
        statusEl.classList.add('ready');

        console.log('%c══════════════════════════════════════', 'color: #2563eb');
        console.log('%c  PoseNet — Single vs Multi-Pose Log  ', 'color: #2563eb; font-weight: bold');
        console.log('%c══════════════════════════════════════', 'color: #2563eb');
        console.log('Performance metrics logged every 30 frames.\n');

        detectPose(net);
    } catch (error) {
        statusEl.textContent = `Error: ${error.message}`;
        console.error('Initialization failed:', error);
    }
}

main();
