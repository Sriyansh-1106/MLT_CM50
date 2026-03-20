/**
 * Assignment 2 — Squat Repetition Counter
 *
 * Uses PoseNet to track hip, knee, and ankle keypoints via laptop webcam.
 * Calculates the knee angle to determine squat position.
 * State machine: standing (>150°) → squatting (<110°) → standing = 1 rep.
 * Shows tracking status so user knows if legs are visible.
 */

// ─── Configuration ──────────────────────────────────────────
const MIN_CONFIDENCE = 0.25;      // Lowered for better detection

// Angle thresholds (slightly relaxed for easier counting)
const STANDING_ANGLE = 150;
const SQUATTING_ANGLE = 110;

// Drawing constants
const KEYPOINT_COLOR  = '#2563eb';
const SKELETON_COLOR  = '#db2777';
const ANGLE_ARC_COLOR = '#059669';

// ─── State ──────────────────────────────────────────────────
let repCount = 0;
let currentState = 'standing';

// ─── DOM Elements ───────────────────────────────────────────
const video      = document.getElementById('video');
const canvas     = document.getElementById('canvas');
const ctx        = canvas.getContext('2d');
const statusEl   = document.getElementById('status');
const repEl      = document.getElementById('rep-count');
const angleEl    = document.getElementById('knee-angle');
const stateEl    = document.getElementById('squat-state');
const trackingEl = document.getElementById('tracking-status');
const resetBtn   = document.getElementById('reset-btn');

// Reset button
resetBtn.addEventListener('click', () => {
    repCount = 0;
    currentState = 'standing';
    repEl.textContent = '0';
    stateEl.textContent = 'Standing';
});

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
 * Calculate angle at point B (knee) formed by points A (hip) and C (ankle).
 */
function calculateAngle(a, b, c) {
    const BA = { x: a.x - b.x, y: a.y - b.y };
    const BC = { x: c.x - b.x, y: c.y - b.y };

    const dot = BA.x * BC.x + BA.y * BC.y;
    const magBA = Math.sqrt(BA.x ** 2 + BA.y ** 2);
    const magBC = Math.sqrt(BC.x ** 2 + BC.y ** 2);

    const cosAngle = Math.max(-1, Math.min(1, dot / (magBA * magBC)));
    return (Math.acos(cosAngle) * 180) / Math.PI;
}

/**
 * Get keypoint by name, returns null if below confidence.
 */
function getKeypoint(keypoints, name) {
    const kp = keypoints.find(k => k.part === name);
    return (kp && kp.score >= MIN_CONFIDENCE) ? kp : null;
}

/**
 * Update squat state machine and count reps.
 */
function processSquat(angle) {
    if (angle < SQUATTING_ANGLE && currentState === 'standing') {
        currentState = 'squatting';
        stateEl.textContent = 'Squatting ↓';
    } else if (angle > STANDING_ANGLE && currentState === 'squatting') {
        currentState = 'standing';
        repCount++;
        repEl.textContent = repCount;
        stateEl.textContent = 'Standing ↑';
    }
}

/**
 * Draw skeleton and keypoints.
 */
function drawPose(keypoints) {
    const pairs = posenet.getAdjacentKeyPoints(keypoints, MIN_CONFIDENCE);
    pairs.forEach(([from, to]) => {
        ctx.beginPath();
        ctx.moveTo(from.position.x, from.position.y);
        ctx.lineTo(to.position.x, to.position.y);
        ctx.strokeStyle = SKELETON_COLOR;
        ctx.lineWidth = 3;
        ctx.stroke();
    });

    keypoints.forEach(kp => {
        if (kp.score < MIN_CONFIDENCE) return;
        ctx.beginPath();
        ctx.arc(kp.position.x, kp.position.y, 6, 0, 2 * Math.PI);
        ctx.fillStyle = KEYPOINT_COLOR;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
    });
}

/**
 * Draw angle arc and label at the knee.
 */
function drawAngleArc(hip, knee, ankle, angle) {
    ctx.beginPath();
    const startAngle = Math.atan2(hip.y - knee.y, hip.x - knee.x);
    const endAngle   = Math.atan2(ankle.y - knee.y, ankle.x - knee.x);
    ctx.arc(knee.x, knee.y, 30, startAngle, endAngle);
    ctx.strokeStyle = ANGLE_ARC_COLOR;
    ctx.lineWidth = 3;
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 16px DM Sans, system-ui';
    ctx.fillText(`${Math.round(angle)}°`, knee.x + 35, knee.y - 10);
}

/**
 * Main detection loop.
 */
async function detectPose(net) {
    const pose = await net.estimateSinglePose(video, { flipHorizontal: false });
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    drawPose(pose.keypoints);

    // Try left side first, then right side
    let hip   = getKeypoint(pose.keypoints, 'leftHip')   || getKeypoint(pose.keypoints, 'rightHip');
    let knee  = getKeypoint(pose.keypoints, 'leftKnee')  || getKeypoint(pose.keypoints, 'rightKnee');
    let ankle = getKeypoint(pose.keypoints, 'leftAnkle') || getKeypoint(pose.keypoints, 'rightAnkle');

    if (hip && knee && ankle) {
        // All three joints detected — tracking active
        trackingEl.textContent = '✓ Legs tracked';
        trackingEl.style.color = '#059669';

        const angle = calculateAngle(hip.position, knee.position, ankle.position);
        angleEl.textContent = `${Math.round(angle)}°`;

        drawAngleArc(hip.position, knee.position, ankle.position, angle);

        // Highlight tracked joints
        [hip, knee, ankle].forEach(kp => {
            ctx.beginPath();
            ctx.arc(kp.position.x, kp.position.y, 10, 0, 2 * Math.PI);
            ctx.strokeStyle = ANGLE_ARC_COLOR;
            ctx.lineWidth = 3;
            ctx.stroke();
        });

        processSquat(angle);
    } else {
        // Joints not visible — show what's missing
        const missing = [];
        if (!hip) missing.push('hip');
        if (!knee) missing.push('knee');
        if (!ankle) missing.push('ankle');
        trackingEl.textContent = `Missing: ${missing.join(', ')}`;
        trackingEl.style.color = '#d97706';
        angleEl.textContent = '--°';
    }

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

        statusEl.textContent = '✓ Ready — perform squats to count reps';
        statusEl.classList.add('ready');

        detectPose(net);
    } catch (error) {
        statusEl.textContent = `Error: ${error.message}`;
        console.error('Initialization failed:', error);
    }
}

main();
