/* ===================================================================
   SKYSHIELD C-UAS — app.js
   Radar PPI canvas, target approach animation, API integration,
   Web Audio alerts, telemetry display, micro-Doppler waveform
   =================================================================== */

(() => {
    'use strict';

    // ---------------------------------------------------------------
    // CONFIG
    // ---------------------------------------------------------------
    const API_BASE = '';  // same origin
    const SWEEP_PERIOD = 2400;       // ms for full 360° rotation (fast tactical sweep)
    const APPROACH_DURATION = 1100;  // ms for target to reach detection ring (fast 1.1s lock)
    const DETECTION_RING = 0.32;     // fraction of radius where target locks

    // ---------------------------------------------------------------
    // STATE
    // ---------------------------------------------------------------
    let mode = 'fusion';            // default to 'fusion' (matches Notebook 05)
    let audioCtx = null;
    let audioArmed = false;
    let sirenOsc = null;
    let sirenGain = null;
    let sirenInterval = null;
    let currentFile = null;
    let targetId = 0;
    let incidentLog = [];
    let isProcessing = false;
    let pendingPredictionPromise = null;
    let predictionResult = null;
    let targetLockedFired = false;

    // Radar canvas state
    let sweepAngle = 0;
    let radarAnimId = null;
    let blips = [];            // background noise blips
    let targetBlip = null;     // the active target moving in
    let lockedTarget = null;   // target that reached detection ring

    // Waveform state
    let waveformAnimId = null;
    let waveformClass = null;
    let waveformOffset = 0;

    // ---------------------------------------------------------------
    // DOM REFS
    // ---------------------------------------------------------------
    const $ = id => document.getElementById(id);
    const radarCanvas = $('radarCanvas');
    const ctx = radarCanvas.getContext('2d');
    const waveCanvas = $('waveformCanvas');
    const wctx = waveCanvas.getContext('2d');

    // ---------------------------------------------------------------
    // RADAR PPI CANVAS
    // ---------------------------------------------------------------
    function initRadar() {
        generateNoiseBlips();
        radarLoop();
    }

    function generateNoiseBlips() {
        blips = [];
        for (let i = 0; i < 40; i++) {
            blips.push({
                angle: Math.random() * Math.PI * 2,
                dist: 0.15 + Math.random() * 0.8,
                alpha: 0.05 + Math.random() * 0.15,
                size: 1 + Math.random() * 2
            });
        }
    }

    function radarLoop() {
        const now = performance.now();
        sweepAngle = ((now % SWEEP_PERIOD) / SWEEP_PERIOD) * Math.PI * 2 - Math.PI / 2;

        drawRadar(now);

        // Update azimuth readout
        const azDeg = ((sweepAngle + Math.PI / 2) / (Math.PI * 2) * 360 + 360) % 360;
        $('sweepAzimuth').textContent = `AZ: ${String(Math.floor(azDeg)).padStart(3, '0')}°`;

        radarAnimId = requestAnimationFrame(radarLoop);
    }

    function drawRadar(now) {
        const W = radarCanvas.width;
        const H = radarCanvas.height;
        const cx = W / 2;
        const cy = H / 2;
        const R = Math.min(cx, cy) - 10;

        // Fade previous frame (afterglow trail)
        ctx.fillStyle = 'rgba(3, 8, 14, 0.12)';
        ctx.fillRect(0, 0, W, H);

        // Range rings
        ctx.strokeStyle = 'rgba(0, 255, 136, 0.12)';
        ctx.lineWidth = 1;
        for (let i = 1; i <= 5; i++) {
            ctx.beginPath();
            ctx.arc(cx, cy, R * (i / 5), 0, Math.PI * 2);
            ctx.stroke();
        }

        // Cross hairs
        ctx.strokeStyle = 'rgba(0, 255, 136, 0.08)';
        ctx.beginPath();
        ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R);
        ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy);
        ctx.stroke();

        // Compass labels
        ctx.fillStyle = 'rgba(0, 255, 136, 0.3)';
        ctx.font = '11px "Orbitron", monospace';
        ctx.textAlign = 'center';
        ctx.fillText('N', cx, cy - R + 14);
        ctx.fillText('S', cx, cy + R - 6);
        ctx.fillText('E', cx + R - 10, cy + 4);
        ctx.fillText('W', cx - R + 10, cy + 4);

        // 30° tick marks
        for (let deg = 0; deg < 360; deg += 30) {
            const a = (deg * Math.PI / 180) - Math.PI / 2;
            const x1 = cx + Math.cos(a) * (R - 2);
            const y1 = cy + Math.sin(a) * (R - 2);
            const x2 = cx + Math.cos(a) * (R - 10);
            const y2 = cy + Math.sin(a) * (R - 10);
            ctx.strokeStyle = 'rgba(0, 255, 136, 0.2)';
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        }

        // Detection ring (inner)
        ctx.strokeStyle = 'rgba(0, 229, 255, 0.25)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 6]);
        ctx.beginPath();
        ctx.arc(cx, cy, R * DETECTION_RING, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

        // Sweep line with gradient
        const sweepLen = R;
        const sx = cx + Math.cos(sweepAngle) * sweepLen;
        const sy = cy + Math.sin(sweepAngle) * sweepLen;

        // Sweep gradient (pie-slice afterglow)
        const sweepGrad = ctx.createConicalGradient
            ? null  // not widely supported
            : null;

        // Draw sweep trail (manual arc)
        for (let i = 0; i < 40; i++) {
            const a = sweepAngle - (i / 40) * 0.6;
            const alpha = 0.25 * (1 - i / 40);
            ctx.strokeStyle = `rgba(0, 255, 136, ${alpha})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(cx + Math.cos(a) * R, cy + Math.sin(a) * R);
            ctx.stroke();
        }

        // Main sweep line
        ctx.strokeStyle = 'rgba(0, 255, 136, 0.8)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(sx, sy);
        ctx.stroke();

        // Center dot
        ctx.fillStyle = '#00ff88';
        ctx.beginPath();
        ctx.arc(cx, cy, 3, 0, Math.PI * 2);
        ctx.fill();

        // Noise blips
        for (const b of blips) {
            // Blips brighten near sweep line
            const angleDiff = Math.abs(normalizeAngle(b.angle - sweepAngle));
            const brightness = angleDiff < 0.3 ? b.alpha + 0.2 : b.alpha * 0.5;
            ctx.fillStyle = `rgba(0, 255, 136, ${brightness})`;
            ctx.beginPath();
            ctx.arc(
                cx + Math.cos(b.angle) * b.dist * R,
                cy + Math.sin(b.angle) * b.dist * R,
                b.size, 0, Math.PI * 2
            );
            ctx.fill();
        }

        // Active target blip (approaching)
        if (targetBlip) {
            const elapsed = now - targetBlip.startTime;
            const progress = Math.min(elapsed / APPROACH_DURATION, 1);
            const eased = easeOutCubic(progress);
            const currentDist = targetBlip.startDist * (1 - eased) + DETECTION_RING * eased;

            targetBlip.currentDist = currentDist;

            // Trail echoes
            for (let t = 0; t < 5; t++) {
                const trailDist = currentDist + t * 0.04;
                if (trailDist > 1) continue;
                const ta = 0.5 - t * 0.1;
                ctx.fillStyle = `rgba(255, 100, 50, ${Math.max(ta, 0)})`;
                ctx.beginPath();
                ctx.arc(
                    cx + Math.cos(targetBlip.angle) * trailDist * R,
                    cy + Math.sin(targetBlip.angle) * trailDist * R,
                    4 - t * 0.5, 0, Math.PI * 2
                );
                ctx.fill();
            }

            // Main blip
            const bx = cx + Math.cos(targetBlip.angle) * currentDist * R;
            const by = cy + Math.sin(targetBlip.angle) * currentDist * R;

            // Glow
            ctx.fillStyle = 'rgba(255, 80, 40, 0.3)';
            ctx.beginPath();
            ctx.arc(bx, by, 12, 0, Math.PI * 2);
            ctx.fill();

            // Core
            ctx.fillStyle = '#ff5533';
            ctx.beginPath();
            ctx.arc(bx, by, 5, 0, Math.PI * 2);
            ctx.fill();

            // Range readout near blip
            const rangekm = (currentDist * 25).toFixed(1);
            ctx.fillStyle = '#ff8844';
            ctx.font = '10px "JetBrains Mono", monospace';
            ctx.textAlign = 'left';
            ctx.fillText(`${rangekm} km`, bx + 10, by - 6);

            // Update radar mode display
            $('radarModeDisplay').textContent = 'TRACKING';
            $('trackCountDisplay').textContent = '1';

            // Check if reached detection ring
            if (progress >= 1 && !targetBlip.locked) {
                targetBlip.locked = true;
                onTargetLocked();
            }
        }

        // Locked target (pulsing at detection ring)
        if (lockedTarget) {
            const lx = cx + Math.cos(lockedTarget.angle) * DETECTION_RING * R;
            const ly = cy + Math.sin(lockedTarget.angle) * DETECTION_RING * R;

            const pulseScale = 1 + 0.3 * Math.sin(now / 200);

            // Pulsing ring
            ctx.strokeStyle = `rgba(0, 229, 255, ${0.4 + 0.3 * Math.sin(now / 150)})`;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(lx, ly, 14 * pulseScale, 0, Math.PI * 2);
            ctx.stroke();

            // Core
            ctx.fillStyle = '#00e5ff';
            ctx.beginPath();
            ctx.arc(lx, ly, 5, 0, Math.PI * 2);
            ctx.fill();

            // TARGET LOCKED text
            ctx.fillStyle = '#00e5ff';
            ctx.font = 'bold 11px "Orbitron", monospace';
            ctx.textAlign = 'center';
            ctx.fillText('TARGET LOCKED', lx, ly - 20);
        }

        // Range labels on rings
        ctx.fillStyle = 'rgba(0, 255, 136, 0.2)';
        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.textAlign = 'left';
        for (let i = 1; i <= 5; i++) {
            ctx.fillText(`${i * 5}km`, cx + R * (i / 5) - 20, cy - 4);
        }
    }

    function normalizeAngle(a) {
        while (a > Math.PI) a -= Math.PI * 2;
        while (a < -Math.PI) a += Math.PI * 2;
        return a;
    }

    function easeOutCubic(t) {
        return 1 - Math.pow(1 - t, 3);
    }

    // ---------------------------------------------------------------
    // TARGET APPROACH ANIMATION
    // ---------------------------------------------------------------
    function startTargetApproach() {
        // Spawn blip at random edge
        const angle = Math.random() * Math.PI * 2;
        targetBlip = {
            angle: angle,
            startDist: 0.92,
            currentDist: 0.92,
            startTime: performance.now(),
            locked: false
        };
        lockedTarget = null;

        // Approach ping sounds
        if (audioArmed) playApproachPings();

        // Update UI
        $('radarModeDisplay').textContent = 'TRACKING';
        $('systemStatusText').textContent = 'TARGET INBOUND';
        $('systemBadge').querySelector('.status-dot').className = 'status-dot amber';
    }

    function onTargetLocked() {
        // Freeze at detection ring
        lockedTarget = {
            angle: targetBlip ? targetBlip.angle : 0
        };
        targetBlip = null;
        targetLockedFired = true;

        // Play lock sound
        if (audioArmed) playLockSound();

        // Update UI
        $('radarModeDisplay').textContent = 'LOCKED';
        $('previewStatus').textContent = 'TARGET LOCKED';

        // If prediction already arrived during the 1.1s approach, display instantly!
        if (predictionResult) {
            displayResults(predictionResult);
        } else {
            $('analyzingState').classList.remove('hidden');
        }
    }

    // ---------------------------------------------------------------
    // FILE UPLOAD & IMAGE HANDLING
    // ---------------------------------------------------------------
    function setupUpload() {
        const dropZone = $('dropZone');
        const fileInput = $('fileInput');

        dropZone.addEventListener('click', () => fileInput.click());

        dropZone.addEventListener('dragover', e => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', e => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            } else {
                // Dragged from another browser tab or Google Images
                const html = e.dataTransfer.getData('text/html');
                const uri = e.dataTransfer.getData('text/uri-list');
                const plain = e.dataTransfer.getData('text/plain');

                let extractedUrl = null;
                if (html) {
                    const match = html.match(/src=["'](.*?)["']/i);
                    if (match) extractedUrl = match[1];
                }
                if (!extractedUrl && uri) extractedUrl = uri;
                if (!extractedUrl && plain && (plain.match(/^https?:\/\//i) || plain.startsWith('data:image/'))) {
                    extractedUrl = plain;
                }

                if (extractedUrl) {
                    handleUrl(extractedUrl);
                } else {
                    showToast('Drop an image file or drag directly from web search');
                }
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                handleFile(fileInput.files[0]);
            }
        });

        $('urlFetchBtn').addEventListener('click', () => {
            const url = $('urlInput').value.trim();
            if (url) handleUrl(url);
        });

        $('urlInput').addEventListener('keydown', e => {
            if (e.key === 'Enter') {
                const url = $('urlInput').value.trim();
                if (url) handleUrl(url);
            }
        });
    }

    function handleFile(file) {
        if (isProcessing) return;
        if (!file.type.startsWith('image/')) {
            showToast('Invalid file — must be an image (JPEG/PNG)');
            return;
        }
        currentFile = file;
        isProcessing = true;

        // Show preview
        const reader = new FileReader();
        reader.onload = e => {
            $('previewImg').src = e.target.result;
            $('previewStrip').classList.remove('hidden');
            $('previewStatus').textContent = 'TARGET ACQUIRED';
        };
        reader.readAsDataURL(file);

        // Reset previous results
        resetResults();

        // Start radar approach animation and parallel prediction
        startTargetApproach();
        startParallelPrediction();
    }

    function handleUrl(url) {
        if (isProcessing) return;
        currentFile = null;
        isProcessing = true;

        $('previewImg').src = url;
        $('previewStrip').classList.remove('hidden');
        $('previewStatus').textContent = 'TARGET ACQUIRED';

        resetResults();
        startTargetApproach();
        startParallelPrediction();
    }

    function resetResults() {
        $('verdictCard').classList.add('hidden');
        $('analyzingState').classList.add('hidden');
        $('gradcamSection').classList.add('hidden');
        $('gradcamImages').classList.add('hidden');
        $('alertOverlay').classList.add('hidden');
        $('headerBar').classList.remove('drone-alert');
        $('threatTag').textContent = 'PROCESSING';
        $('threatTag').style.color = '';

        // Stop siren
        stopSiren();

        // Reset bars
        ['cnn', 'radar', 'fusion'].forEach(prefix => {
            ['Bird', 'Drone', 'Other'].forEach(cls => {
                const barId = `${prefix}${cls}Bar`;
                const el = $(barId);
                if (el) el.style.width = '0%';
            });
        });

        lockedTarget = null;
        pendingPredictionPromise = null;
        predictionResult = null;
        targetLockedFired = false;
    }

    // ---------------------------------------------------------------
    // FAST CLIENT-SIDE IMAGE COMPRESSION (For near-instant cloud upload)
    // Downscales to max 384x384 (CNN requires 224x224), reducing 4MB to ~25KB
    // ---------------------------------------------------------------
    async function compressImageForUpload(source) {
        return new Promise((resolve) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => {
                const maxDim = 384;
                let w = img.width;
                let h = img.height;
                if (w > maxDim || h > maxDim) {
                    if (w > h) {
                        h = Math.round((h * maxDim) / w);
                        w = maxDim;
                    } else {
                        w = Math.round((w * maxDim) / h);
                        h = maxDim;
                    }
                }
                const canvas = document.createElement('canvas');
                canvas.width = w;
                canvas.height = h;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, w, h);
                canvas.toBlob((blob) => {
                    resolve(blob || source);
                }, 'image/jpeg', 0.85);
            };
            img.onerror = () => resolve(source);

            if (typeof source === 'string') {
                img.src = source;
            } else if (source instanceof Blob || source instanceof File) {
                img.src = URL.createObjectURL(source);
            } else {
                resolve(source);
            }
        });
    }

    // Helper to convert base64 data URLs to Blob without canvas re-compression
    function dataURLtoBlob(dataurl) {
        try {
            const arr = dataurl.split(',');
            const mime = arr[0].match(/:(.*?);/)[1];
            const bstr = atob(arr[1]);
            let n = bstr.length;
            const u8arr = new Uint8Array(n);
            while (n--) {
                u8arr[n] = bstr.charCodeAt(n);
            }
            return new Blob([u8arr], {type: mime});
        } catch (e) {
            return dataurl;
        }
    }

    // ---------------------------------------------------------------
    // PARALLEL API CALL (Executes during radar sweep for 0ms lag)
    // ---------------------------------------------------------------
    async function startParallelPrediction() {
        try {
            const formData = new FormData();

            if (currentFile) {
                // Upload original file directly to preserve 100% pixel integrity (no canvas compression)
                formData.append('file', currentFile, currentFile.name || 'target.jpg');
            } else {
                const url = $('previewImg').src || $('urlInput').value.trim();
                if (url.startsWith('data:image/')) {
                    formData.append('file', dataURLtoBlob(url), 'target.jpg');
                } else {
                    formData.append('image_url', url);
                }
            }

            let endpoint;
            if (mode === 'fusion') {
                const scenario = document.querySelector('input[name="radarScenario"]:checked')?.value || 'auto';
                formData.append('radar_scenario', scenario);
                endpoint = `${API_BASE}/predict/fused`;
            } else {
                endpoint = `${API_BASE}/predict/image`;
            }

            const response = await fetch(endpoint, { method: 'POST', body: formData });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Prediction failed');
            }

            const data = await response.json();
            predictionResult = data;

            // If target has already locked, display immediately!
            if (targetLockedFired) {
                displayResults(data);
            }
        } catch (error) {
            console.error('Prediction error:', error);
            showToast(`Error: ${error.message}`);
            $('analyzingState').classList.add('hidden');
            $('previewStatus').textContent = 'ERROR';
            isProcessing = false;
        }
    }

    // ---------------------------------------------------------------
    // DISPLAY RESULTS
    // ---------------------------------------------------------------
    function displayResults(data) {
        $('analyzingState').classList.add('hidden');
        $('verdictCard').classList.remove('hidden');

        const isFusion = mode === 'fusion' && data.fused_probabilities;

        // Determine final verdict info
        const status = data.status;
        const alertLevel = data.alert_level;
        const predictedClass = data.predicted_class;

        // Vision (CNN) probabilities
        const vProbs = isFusion
            ? data.vision_result.probabilities
            : data.probabilities;

        // Radar probabilities
        const rProbs = isFusion
            ? data.radar_result.probabilities
            : null;

        // Fused probabilities
        const fProbs = isFusion
            ? data.fused_probabilities
            : data.probabilities;  // In vision-only mode, show CNN probs as "final"

        // Animate CNN bars
        animateBars('cnn', vProbs);

        // Animate Radar bars (if fusion)
        if (rProbs) {
            animateBars('radar', rProbs);
            $('radarCard').style.opacity = '1';
            const radarPred = data.radar_result.prediction;
            $('radarVerdict').textContent = `VERDICT: ${radarPred.toUpperCase()}`;
            $('radarVerdict').style.color = getClassColor(radarPred);
        } else {
            // Show simulated radar that agrees with vision
            const simRadar = simulateRadarProbs(vProbs);
            animateBars('radar', simRadar);
            $('radarCard').style.opacity = '1';
            const rPred = getMaxClass(simRadar);
            $('radarVerdict').textContent = `VERDICT: ${rPred.toUpperCase()}`;
            $('radarVerdict').style.color = getClassColor(rPred);
        }

        // Animate Fusion bars
        animateBars('fusion', fProbs);

        // CNN verdict
        const cnnPred = isFusion ? (data.vision_result?.dominant_class || data.vision_result?.prediction || predictedClass) : predictedClass;
        $('cnnVerdict').textContent = `VERDICT: ${cnnPred.toUpperCase()}`;
        $('cnnVerdict').style.color = getClassColor(cnnPred);

        // Fusion verdict
        $('fusionVerdict').textContent = (status === 'UNIDENTIFIED' ? '★ UNIDENTIFIED (LOW CONFIDENCE) ★' : `★ FINAL: ${predictedClass.toUpperCase()} ★`);
        $('fusionVerdict').style.color = getClassColor(predictedClass);

        // Main verdict banner
        const banner = $('verdictBanner');
        banner.className = 'verdict-banner';
        if (status === 'DRONE')        banner.classList.add('drone');
        else if (status === 'BIRD')    banner.classList.add('bird');
        else if (status === 'OTHER')   banner.classList.add('other');
        else                           banner.classList.add('amber');

        $('verdictClass').textContent = (status === 'UNIDENTIFIED' ? 'UNIDENTIFIED (OUT OF DOMAIN)' : predictedClass.toUpperCase());
        $('verdictBadge').textContent = alertLevel;

        // Threat tag
        $('threatTag').textContent = `STATUS: ${status}`;
        $('threatTag').style.color = getClassColor(predictedClass);

        // Inference stats
        const latency = data.total_latency_ms || data.latency_ms || 0;
        const confidence = isFusion
            ? (data.fused_confidence || 0)
            : (data.confidence || 0);
        $('infLatency').textContent = `${latency.toFixed(1)} ms`;
        $('infConf').textContent = `${(confidence * 100).toFixed(1)}%`;
        $('infOverride').textContent = data.drone_override_active ? '🚨 ACTIVE' : 'INACTIVE';
        $('infOverride').style.color = data.drone_override_active ? '#ff3344' : '';
        $('infTemp').textContent = (data.vision_result?.temperature || data.temperature || 0.64).toFixed(2);

        // Telemetry
        if (data.telemetry) {
            displayTelemetry(data.telemetry, predictedClass);
        } else {
            // Generate client-side telemetry for vision-only mode
            displayTelemetry(generateClientTelemetry(predictedClass), predictedClass);
        }

        // Start waveform animation
        startWaveform(predictedClass);

        // Grad-CAM
        const gcB64 = isFusion
            ? data.vision_result?.gradcam_base64
            : data.gradcam_base64;
        if (gcB64) {
            $('gradcamSection').classList.remove('hidden');
            $('gcTarget').src = $('previewImg').src;
            $('gcHeatmap').src = `data:image/png;base64,${gcB64}`;
        }

        // Audio alerts
        if (audioArmed) {
            if (status === 'DRONE') {
                startSiren();
                $('alertOverlay').classList.remove('hidden');
                $('headerBar').classList.add('drone-alert');
                $('ackAlarmBtn').classList.remove('hidden');
            } else if (status === 'BIRD') {
                playSafeSound();
            } else if (status === 'OTHER') {
                playInfoSound();
            } else {
                playAmberSound();
            }
        }

        // DRONE red flash
        if (status === 'DRONE') {
            $('alertOverlay').classList.remove('hidden');
            $('headerBar').classList.add('drone-alert');
            $('ackAlarmBtn').classList.remove('hidden');
            $('systemStatusText').textContent = '🚨 DRONE DETECTED';
            $('systemBadge').style.borderColor = '#ff3344';
            $('systemBadge').style.color = '#ff3344';
            $('systemBadge').querySelector('.status-dot').className = 'status-dot red';
        } else if (status === 'BIRD') {
            $('systemStatusText').textContent = 'BIRD — NO THREAT';
            $('systemBadge').style.borderColor = '';
            $('systemBadge').style.color = '';
            $('systemBadge').querySelector('.status-dot').className = 'status-dot green';
        } else {
            $('systemStatusText').textContent = `${status} DETECTED`;
            $('systemBadge').querySelector('.status-dot').className = 'status-dot amber';
        }

        // Add to incident log
        targetId++;
        addLogEntry({
            id: targetId,
            mode: mode.toUpperCase(),
            cnn: cnnPred,
            radar: rProbs ? (isFusion ? data.radar_result.prediction : '—') : 'SIM',
            fused: predictedClass,
            confidence: confidence,
            alert: alertLevel,
            latency: latency
        });

        // Preview status
        $('previewStatus').textContent = `${status} — ${(confidence * 100).toFixed(1)}%`;

        // Done processing
        isProcessing = false;

        // Update track count
        $('trackCountDisplay').textContent = '1';
    }

    function simulateRadarProbs(visionProbs) {
        // For vision-only mode: generate radar-like probs that roughly agree with vision
        // but with some realistic noise to show they're independent sensors
        const result = {};
        const dominant = getMaxClass(visionProbs);
        const noise = () => 0.02 + Math.random() * 0.06;

        for (const cls of ['Bird', 'Drone', 'Other']) {
            if (cls === dominant) {
                result[cls] = 0.80 + Math.random() * 0.15;
            } else {
                result[cls] = noise();
            }
        }
        // Normalize
        const sum = Object.values(result).reduce((a, b) => a + b, 0);
        for (const cls in result) result[cls] /= sum;
        return result;
    }

    function getMaxClass(probs) {
        let max = 0, maxCls = 'Bird';
        for (const cls in probs) {
            if (probs[cls] > max) { max = probs[cls]; maxCls = cls; }
        }
        return maxCls;
    }

    function animateBars(prefix, probs) {
        if (!probs) return;
        setTimeout(() => {
            for (const cls of ['Bird', 'Drone', 'Other']) {
                const pct = (probs[cls] || 0) * 100;
                const barEl = $(`${prefix}${cls}Bar`);
                const pctEl = $(`${prefix}${cls}`);
                if (barEl) barEl.style.width = `${pct}%`;
                if (pctEl) pctEl.textContent = `${pct.toFixed(1)}%`;
            }
        }, 100);
    }

    function getClassColor(cls) {
        const c = (cls || '').toLowerCase();
        if (c.includes('drone'))  return '#ff3344';
        if (c.includes('bird'))   return '#00ff88';
        if (c.includes('other'))  return '#2979ff';
        return '#ffaa00';
    }

    // ---------------------------------------------------------------
    // TELEMETRY DISPLAY
    // ---------------------------------------------------------------
    function displayTelemetry(telem, className) {
        const animate = (elId, value, suffix) => {
            const el = $(elId);
            if (!el) return;
            const target = parseFloat(value) || 0;
            let current = 0;
            const steps = 30;
            const inc = target / steps;
            let step = 0;
            const timer = setInterval(() => {
                step++;
                current += inc;
                if (step >= steps) {
                    current = target;
                    clearInterval(timer);
                }
                if (target >= 100) {
                    el.textContent = `${Math.round(current)} ${suffix}`;
                } else if (target >= 1) {
                    el.textContent = `${current.toFixed(1)} ${suffix}`;
                } else {
                    el.textContent = `${current.toFixed(3)} ${suffix}`;
                }
            }, 25);
        };

        animate('telemSpeed', telem.speed_ms, 'm/s');
        animate('telemVelocity', telem.velocity_kmh, 'km/h');
        animate('telemDoppler', telem.doppler_hz, 'Hz');
        animate('telemRCS', telem.rcs_m2, 'm²');
        animate('telemAlt', telem.altitude_m, 'm');
        animate('telemRange', telem.range_km, 'km');
        animate('telemHeading', telem.heading_deg, '°');
        animate('telemRotor', telem.rotor_wing_hz, 'Hz');
    }

    function generateClientTelemetry(className) {
        const ranges = {
            Bird:  { speed_ms: [5,25], velocity_kmh: [18,90], doppler_hz: [80,450], rcs_m2: [0.001,0.01], altitude_m: [10,500], range_km: [0.5,5], heading_deg: [0,360], rotor_wing_hz: [2,15] },
            Drone: { speed_ms: [8,45], velocity_kmh: [29,162], doppler_hz: [150,900], rcs_m2: [0.01,0.5], altitude_m: [5,400], range_km: [0.3,8], heading_deg: [0,360], rotor_wing_hz: [20,200] },
            Other: { speed_ms: [60,250], velocity_kmh: [216,900], doppler_hz: [1200,5000], rcs_m2: [1,100], altitude_m: [500,12000], range_km: [2,25], heading_deg: [0,360], rotor_wing_hz: [0,0] }
        };
        const r = ranges[className] || ranges.Other;
        const result = {};
        for (const key in r) {
            const [lo, hi] = r[key];
            result[key] = lo + Math.random() * (hi - lo);
        }
        return result;
    }

    // ---------------------------------------------------------------
    // MICRO-DOPPLER WAVEFORM
    // ---------------------------------------------------------------
    function startWaveform(className) {
        waveformClass = className;
        waveformOffset = 0;
        if (waveformAnimId) cancelAnimationFrame(waveformAnimId);
        drawWaveformLoop();
    }

    function drawWaveformLoop() {
        const W = waveCanvas.width;
        const H = waveCanvas.height;
        wctx.clearRect(0, 0, W, H);

        // Background grid
        wctx.strokeStyle = 'rgba(0, 255, 136, 0.06)';
        wctx.lineWidth = 0.5;
        for (let y = 0; y < H; y += 20) {
            wctx.beginPath(); wctx.moveTo(0, y); wctx.lineTo(W, y); wctx.stroke();
        }
        for (let x = 0; x < W; x += 20) {
            wctx.beginPath(); wctx.moveTo(x, 0); wctx.lineTo(x, H); wctx.stroke();
        }

        // Draw waveform based on class
        wctx.strokeStyle = getClassColor(waveformClass);
        wctx.lineWidth = 1.5;
        wctx.shadowColor = getClassColor(waveformClass);
        wctx.shadowBlur = 6;
        wctx.beginPath();

        const cls = (waveformClass || '').toLowerCase();
        for (let x = 0; x < W; x++) {
            const t = (x + waveformOffset) * 0.02;
            let y;

            if (cls === 'bird') {
                // Low-frequency sinusoidal (wing flaps 2-15Hz)
                y = H/2 + Math.sin(t * 0.8) * 25
                    + Math.sin(t * 2.3) * 10
                    + Math.sin(t * 0.3) * 8;
            } else if (cls === 'drone') {
                // High-frequency sawtooth/complex (multi-rotor 20-200Hz)
                y = H/2 + Math.sin(t * 4) * 18
                    + Math.sin(t * 8.5) * 12
                    + Math.sin(t * 12) * 6
                    + (Math.random() - 0.5) * 4;
            } else {
                // Aircraft: smooth, high-amplitude
                y = H/2 + Math.sin(t * 0.4) * 28
                    + Math.sin(t * 1.2) * 8;
            }

            y = Math.max(4, Math.min(H - 4, y));

            if (x === 0) wctx.moveTo(x, y);
            else wctx.lineTo(x, y);
        }
        wctx.stroke();
        wctx.shadowBlur = 0;

        waveformOffset += 2;
        waveformAnimId = requestAnimationFrame(drawWaveformLoop);
    }

    // ---------------------------------------------------------------
    // WEB AUDIO — SOUNDS
    // ---------------------------------------------------------------
    function initAudio() {
        if (audioCtx) return;
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }

    function playTone(freq, duration, type = 'sine', volume = 0.15) {
        if (!audioCtx) return;
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = type;
        osc.frequency.value = freq;
        gain.gain.value = volume;
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + duration);
    }

    function playApproachPings() {
        let count = 0;
        const interval = setInterval(() => {
            if (count >= 6 || targetBlip?.locked) {
                clearInterval(interval);
                return;
            }
            const freq = 400 + count * 80;
            playTone(freq, 0.08, 'sine', 0.1);
            count++;
        }, 500);
    }

    function playLockSound() {
        playTone(1000, 0.08, 'square', 0.12);
        setTimeout(() => playTone(1200, 0.08, 'square', 0.12), 120);
    }

    function startSiren() {
        if (!audioCtx) return;
        stopSiren();

        sirenOsc = audioCtx.createOscillator();
        sirenGain = audioCtx.createGain();
        sirenOsc.type = 'sawtooth';
        sirenGain.gain.value = 0.12;
        sirenOsc.connect(sirenGain);
        sirenGain.connect(audioCtx.destination);
        sirenOsc.start();

        // Sweep 600-1200 Hz
        let rising = true;
        sirenInterval = setInterval(() => {
            if (!sirenOsc) return;
            const now = audioCtx.currentTime;
            if (rising) {
                sirenOsc.frequency.linearRampToValueAtTime(1200, now + 0.3);
            } else {
                sirenOsc.frequency.linearRampToValueAtTime(600, now + 0.3);
            }
            rising = !rising;
        }, 300);
    }

    function stopSiren() {
        if (sirenOsc) {
            try { sirenOsc.stop(); } catch (e) {}
            sirenOsc = null;
        }
        if (sirenInterval) {
            clearInterval(sirenInterval);
            sirenInterval = null;
        }
    }

    function playSafeSound() {
        playTone(523, 0.25, 'sine', 0.1);
    }

    function playInfoSound() {
        playTone(800, 0.15, 'sine', 0.08);
    }

    function playAmberSound() {
        playTone(600, 0.1, 'square', 0.08);
        setTimeout(() => playTone(600, 0.1, 'square', 0.08), 150);
    }

    // ---------------------------------------------------------------
    // INCIDENT LOG
    // ---------------------------------------------------------------
    function addLogEntry(entry) {
        incidentLog.push(entry);

        const tbody = $('logBody');
        // Remove empty message if present
        if (tbody.querySelector('.empty-log')) {
            tbody.innerHTML = '';
        }

        const tr = document.createElement('tr');
        const now = new Date();
        const ts = now.toLocaleTimeString('en-GB', { hour12: false }) + '.' + String(now.getMilliseconds()).padStart(3, '0');

        const badgeClass = entry.alert === 'CRITICAL' ? 'badge-critical' :
                           entry.alert === 'INFO' ? 'badge-info' :
                           entry.alert === 'WARNING' ? 'badge-warning' : 'badge-amber';

        tr.innerHTML = `
            <td>${ts}</td>
            <td>TGT-${String(entry.id).padStart(3, '0')}</td>
            <td>${entry.mode}</td>
            <td style="color:${getClassColor(entry.cnn)}">${entry.cnn}</td>
            <td style="color:${getClassColor(entry.radar)}">${entry.radar}</td>
            <td style="color:${getClassColor(entry.fused)}; font-weight:700">${entry.fused}</td>
            <td>${(entry.confidence * 100).toFixed(1)}%</td>
            <td><span class="badge ${badgeClass}">${entry.alert}</span></td>
            <td>${entry.latency.toFixed(1)} ms</td>
        `;

        tbody.insertBefore(tr, tbody.firstChild);
    }

    function exportCsv() {
        if (incidentLog.length === 0) return;
        const headers = ['Timestamp', 'TargetID', 'Mode', 'CNN', 'Radar', 'Fused', 'Confidence', 'Alert', 'Latency_ms'];
        const rows = incidentLog.map((e, i) => [
            new Date().toISOString(),
            `TGT-${String(e.id).padStart(3, '0')}`,
            e.mode, e.cnn, e.radar, e.fused,
            (e.confidence * 100).toFixed(1),
            e.alert,
            e.latency.toFixed(1)
        ]);
        const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `skyshield_incident_log_${Date.now()}.csv`;
        a.click();
        URL.revokeObjectURL(url);
    }

    // ---------------------------------------------------------------
    // TOAST NOTIFICATIONS
    // ---------------------------------------------------------------
    function showToast(msg) {
        const toast = document.createElement('div');
        toast.style.cssText = `
            position: fixed; top: 20px; right: 20px; z-index: 1000;
            background: rgba(255,51,68,0.9); color: #fff;
            padding: 12px 20px; border-radius: 6px;
            font-family: var(--font-mono); font-size: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            animation: fadeIn 0.3s ease;
        `;
        toast.textContent = msg;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 4000);
    }

    // ---------------------------------------------------------------
    // MODE SWITCHING
    // ---------------------------------------------------------------
    function setupModes() {
        $('btnModeVision').addEventListener('click', () => setMode('vision'));
        $('btnModeFusion').addEventListener('click', () => setMode('fusion'));
    }

    function setMode(m) {
        mode = m;
        document.querySelectorAll('.btn-mode').forEach(b => b.classList.remove('active'));
        if (m === 'vision') {
            $('btnModeVision').classList.add('active');
            $('fusionOptions').classList.add('hidden');
        } else {
            $('btnModeFusion').classList.add('active');
            $('fusionOptions').classList.remove('hidden');
        }
    }

    // ---------------------------------------------------------------
    // EVENT LISTENERS
    // ---------------------------------------------------------------
    function setupEvents() {
        // ARM AUDIO
        $('armAudioBtn').addEventListener('click', () => {
            initAudio();
            audioArmed = true;
            $('armAudioBtn').classList.add('hidden');
            $('muteBtn').classList.remove('hidden');
            playTone(880, 0.15, 'sine', 0.08);
        });

        // MUTE
        $('muteBtn').addEventListener('click', () => {
            audioArmed = false;
            stopSiren();
            $('muteBtn').classList.add('hidden');
            $('armAudioBtn').classList.remove('hidden');
        });

        // SILENCE ALARM
        $('ackAlarmBtn').addEventListener('click', () => {
            stopSiren();
            $('alertOverlay').classList.add('hidden');
            $('headerBar').classList.remove('drone-alert');
            $('ackAlarmBtn').classList.add('hidden');
            $('systemStatusText').textContent = 'PERIMETER ARMED';
            $('systemBadge').style.borderColor = '';
            $('systemBadge').style.color = '';
            $('systemBadge').querySelector('.status-dot').className = 'status-dot green';
        });

        // EXPORT CSV
        $('exportCsvBtn').addEventListener('click', exportCsv);

        // CLEAR LOG
        $('clearLogBtn').addEventListener('click', () => {
            incidentLog = [];
            $('logBody').innerHTML = '<tr><td colspan="9" class="empty-log">No incidents recorded. System armed and scanning.</td></tr>';
        });

        // GRAD-CAM toggle
        $('toggleCamBtn').addEventListener('click', () => {
            $('gradcamImages').classList.toggle('hidden');
            $('toggleCamBtn').textContent = $('gradcamImages').classList.contains('hidden') ? 'SHOW HEATMAP' : 'HIDE HEATMAP';
        });
    }

    // ---------------------------------------------------------------
    // INIT
    // ---------------------------------------------------------------
    function init() {
        setupUpload();
        setupModes();
        setupEvents();
        initRadar();
        setMode('fusion');

        // Resize waveform canvas
        const wBox = waveCanvas.parentElement;
        waveCanvas.width = wBox.clientWidth - 20;

        console.log('%c[SKYSHIELD] C-UAS Defense System v2.0 initialized', 'color: #00ff88; font-weight: bold;');
    }

    // Wait for DOM
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
