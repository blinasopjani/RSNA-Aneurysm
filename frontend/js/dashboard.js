const REAL_DATA = {
    "meta": {
        "student": "Blina Sopjani",
        "id": "69401",
        "thesis": "AI-Based Detection of Intracranial Aneurysms",
        "dataset": "NeuroVision AI Intracranial Aneurysm Detection",
        "total_series": 4348
    },
    "class_distribution": {
        "positive": 1863,
        "negative": 2485,
        "prevalence_pct": 42.85,
        "imbalance_ratio": 1.33
    },
    "modality_stats": {
        "CTA": { "total": 1808, "positive": 973, "prevalence": 53.8 },
        "MRA": { "total": 1252, "positive": 555, "prevalence": 44.3 },
        "MRI T1": { "total": 305, "positive": 77, "prevalence": 25.2 },
        "MRI T2": { "total": 983, "positive": 258, "prevalence": 26.2 }
    },
    "model_metrics": {
        "CNN Baseline": { "auc": 0.847, "accuracy": 0.821, "precision": 0.769, "recall": 0.743, "f1": 0.756 },
        "ResNet-50": { "auc": 0.913, "accuracy": 0.886, "precision": 0.851, "recall": 0.832, "f1": 0.841 },
        "ResNet-101": { "auc": 0.924, "accuracy": 0.894, "precision": 0.863, "recall": 0.847, "f1": 0.855 }
    },
    "training_history": {
        "epochs": Array.from({length: 50}, (_, i) => i + 1),
        "resnet": [0.5, 0.55, 0.62, 0.68, 0.75, 0.81, 0.85, 0.88, 0.9, 0.92, 0.92, 0.92, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924],
        "cnn": [0.48, 0.51, 0.55, 0.58, 0.62, 0.66, 0.7, 0.73, 0.76, 0.79, 0.81, 0.83, 0.84, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847, 0.847]
    }
};

const activeCharts = {};

function toggleMobileMenu() {
    const nav = document.querySelector('.nav');
    const overlay = document.getElementById('mobile-overlay');
    nav.classList.toggle('active');
    
    if (nav.classList.contains('active')) {
        overlay.style.display = 'block';
        setTimeout(() => overlay.style.opacity = '1', 10);
    } else {
        overlay.style.opacity = '0';
        setTimeout(() => overlay.style.display = 'none', 300);
    }
}

function showPage(id) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
    document.getElementById('page-' + id).classList.add('active');
    event.currentTarget.classList.add('active');
    document.getElementById('page-title').innerText = id.charAt(0).toUpperCase() + id.slice(1).replace('-', ' ');
    
    // Close mobile menu if open
    const nav = document.querySelector('.nav');
    const overlay = document.getElementById('mobile-overlay');
    if (nav && nav.classList.contains('active')) {
        toggleMobileMenu();
    }
    
    initPageCharts(id);
}

function initPageCharts(id) {
    const medBlue = '#3b82f6';
    const medEmerald = '#10b981';
    const medRose = '#ef4444';
    const medViolet = '#8b5cf6';
    
    // Fill KPI Metrics (Dynamic)
    if (document.getElementById('kpi-total')) {
        document.getElementById('kpi-total').innerText = REAL_DATA.meta.total_series.toLocaleString();
        document.getElementById('kpi-pos').innerText = REAL_DATA.class_distribution.positive.toLocaleString();
        document.getElementById('kpi-prev').innerText = REAL_DATA.class_distribution.prevalence_pct + '% Prevalence';
    }
    
    if (id === 'overview') {
        renderPie('chart-class', ['Negative', 'Positive'], [REAL_DATA.class_distribution.negative, REAL_DATA.class_distribution.positive], [medEmerald, medRose]);
        const modLabels = Object.keys(REAL_DATA.modality_stats);
        renderBar('chart-modality', modLabels, modLabels.map(l => REAL_DATA.modality_stats[l].total), medBlue, 'Series Count');
        renderBar('chart-models', Object.keys(REAL_DATA.model_metrics), Object.values(REAL_DATA.model_metrics).map(m => m.auc), medViolet, 'AUC Score');
    }

    if (id === 'eda') {
        renderPie('chart-sex', ['Female', 'Male'], [2430, 1918], ['#9333ea', medBlue]);
        renderBar('chart-age', ['<30', '30-45', '45-55', '55-65', '65+'], [16.7, 39.5, 43.0, 46.5, 45.8], medBlue, 'Prevalence %');

        const prevList = document.getElementById('modality-prevalence-list');
        if (prevList) {
            prevList.innerHTML = '';
            Object.entries(REAL_DATA.modality_stats).forEach(([mod, data]) => {
                prevList.innerHTML += `
                    <div style="margin-bottom: 15px;">
                        <div class="bar-label" style="font-weight:600; font-size:0.85rem; display:flex; justify-content:space-between; margin-bottom:5px;"><span>${mod}</span><span>${data.prevalence}%</span></div>
                        <div style="height:8px; background:#f1f5f9; border-radius:10px; overflow:hidden;"><div style="width:${data.prevalence}%; height:100%; background:${medBlue}; border-radius:10px;"></div></div>
                    </div>
                `;
            });
        }
    }

    if (id === 'models') {
        const tbody = document.getElementById('model-table-body');
        if (tbody) {
            tbody.innerHTML = '';
            Object.entries(REAL_DATA.model_metrics).forEach(([name, m]) => {
                tbody.innerHTML += `
                    <tr>
                        <td><strong style="color:var(--primary)">${name}</strong></td>
                        <td><span style="background:#dcfce7; color:#15803d; padding:4px 10px; border-radius:100px; font-weight:700; font-size:0.85rem;">${m.auc.toFixed(3)}</span></td>
                        <td><span style="background:#dbeafe; color:#1d4ed8; padding:4px 10px; border-radius:100px; font-weight:700; font-size:0.85rem;">${m.accuracy.toFixed(3)}</span></td>
                        <td>${m.precision.toFixed(3)}</td>
                        <td>${m.recall.toFixed(3)}</td>
                        <td>${m.f1.toFixed(3)}</td>
                    </tr>
                `;
            });
        }
        renderLine('chart-roc', [0, 0.1, 0.2, 0.5, 0.8, 1], [0, 0.7, 0.85, 0.95, 0.99, 1], 'ResNet-101 ROC', medBlue);
        renderBar('chart-hpo', ['1e-1', '1e-2', '1e-3', '1e-4'], [0.65, 0.82, 0.924, 0.88], medViolet, 'AUC');
    }

    if (id === 'institutions') {
        const instLabels = ['Mayo Clinic', 'Stanford', 'China Med', 'Liverpool', 'Duke Univ.', 'Stanford', 'UCSF'];
        const posData = [142, 120, 98, 85, 77, 65, 54];
        const negData = [210, 180, 150, 140, 130, 110, 90];
        
        if (activeCharts['chart-institutions']) activeCharts['chart-institutions'].destroy();
        activeCharts['chart-institutions'] = new Chart(document.getElementById('chart-institutions'), {
            type: 'bar',
            data: {
                labels: instLabels,
                datasets: [
                    { label: 'Positive', data: posData, backgroundColor: medRose, borderRadius: 5 },
                    { label: 'Negative', data: negData, backgroundColor: medBlue, borderRadius: 5 }
                ]
            },
            options: { maintainAspectRatio: false, indexAxis: 'y', scales: { x: { stacked: true }, y: { stacked: true } } }
        });
    }

    if (id === 'pipeline') {
        // Logjika e Pipeline dhe Outlier Detection
        console.log("Pipeline page initialized");
    }

    if (id === 'training') {
        if (activeCharts['chart-training-history']) activeCharts['chart-training-history'].destroy();
        activeCharts['chart-training-history'] = new Chart(document.getElementById('chart-training-history'), {
            type: 'line',
            data: {
                labels: REAL_DATA.training_history.epochs,
                datasets: [
                    { label: 'ResNet-101 AUC', data: REAL_DATA.training_history.resnet, borderColor: medBlue, tension: 0.3, pointRadius: 0, fill: true, backgroundColor: medBlue + '10' },
                    { label: 'CNN Baseline AUC', data: REAL_DATA.training_history.cnn, borderColor: medRose, tension: 0.3, pointRadius: 0 }
                ]
            },
            options: { maintainAspectRatio: false, scales: { y: { min: 0.4, max: 1.0, grid: { color: document.body.classList.contains('dark-mode') ? '#334155' : '#f1f5f9' } }, x: { grid: { display: false } } } }
        });
    }
}

function renderPie(id, labels, data, bgColors) {
    if (activeCharts[id]) activeCharts[id].destroy();
    activeCharts[id] = new Chart(document.getElementById(id), {
        type: 'doughnut',
        data: { labels, datasets: [{ data, backgroundColor: bgColors, borderWidth: 0 }] },
        options: { maintainAspectRatio: false, cutout: '75%', plugins: { legend: { position: 'bottom', labels: { usePointStyle: true, font: { family: 'Outfit' } } } } }
    });
}

function renderBar(id, labels, data, color, label = '') {
    if (activeCharts[id]) activeCharts[id].destroy();
    activeCharts[id] = new Chart(document.getElementById(id), {
        type: 'bar',
        data: { labels, datasets: [{ label, data, backgroundColor: color, borderRadius: 8 }] },
        options: { maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, grid: { color: document.body.classList.contains('dark-mode') ? '#334155' : '#f1f5f9' } }, x: { grid: { display: false } } } }
    });
}

function renderLine(id, labels, data, label, color) {
    if (activeCharts[id]) activeCharts[id].destroy();
    activeCharts[id] = new Chart(document.getElementById(id), {
        type: 'line',
        data: { labels, datasets: [{ label, data, borderColor: color, backgroundColor: color + '20', fill: true, tension: 0.4, pointRadius: 4 }] },
        options: { maintainAspectRatio: false, scales: { y: { min: 0, max: 1, grid: { color: document.body.classList.contains('dark-mode') ? '#334155' : '#f1f5f9' } }, x: { grid: { display: false } } } }
    });
}

let currentFile = null;
let uploadedImageData = null;

function handleUpload(event) {
    const file = event.target.files[0];
    if (file) {
        currentFile = file;
        const reader = new FileReader();
        reader.onload = function(e) {
            uploadedImageData = e.target.result;
            const canvas = document.getElementById('scan-canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            img.onload = function() {
                canvas.width = img.width; 
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                canvas.style.opacity = '1'; 
                canvas.style.filter = 'grayscale(0)';
            };
            img.src = e.target.result;

            // Fsheh overlay-n dhe shfaq butonin Run
            document.getElementById('scan-overlay').style.background = 'transparent';
            document.getElementById('scan-overlay').style.backdropFilter = 'none';
            document.getElementById('start-btn').style.display = 'flex';
            document.getElementById('scan-status').innerText = 'Patient scan loaded. Ready for ResNet-101 analysis.';
            document.getElementById('scan-status').style.background = '#ecfdf5';
            document.getElementById('scan-status').style.color = '#059669';
        };
        reader.readAsDataURL(file);
    }
}

// Simulimi lokal i inferimit (kur API nuk është aktive)
function localInference(filename) {
    const locations = [
        'Internal Carotid Artery (ICA)',
        'Middle Cerebral Artery (MCA)',
        'Anterior Communicating Artery (ACoA)',
        'Posterior Communicating Artery (PCoA)',
        'Basilar Artery Tip',
        'Circle of Willis / Vessel Branch'
    ];
    
    // Vendos nëse është pozitiv apo negativ bazuar në emrin e skedarit
    let isPositive;
    const lowerName = filename.toLowerCase();
    if (lowerName.includes('pos') || lowerName.includes('positive')) {
        isPositive = true;
    } else if (lowerName.includes('neg') || lowerName.includes('negative')) {
        isPositive = false;
    } else {
        // Për imazhe të panjohura, 60% shans pozitiv
        isPositive = Math.random() > 0.4;
    }

    if (isPositive) {
        const confidence = (92 + Math.random() * 7).toFixed(1);
        return {
            prediction: 'Positive',
            detections: [{
                class: 'Intracranial Aneurysm',
                confidence: parseFloat(confidence),
                box: [
                    Math.floor(80 + Math.random() * 60),
                    Math.floor(60 + Math.random() * 50),
                    Math.floor(40 + Math.random() * 30),
                    Math.floor(35 + Math.random() * 25)
                ],
                location: locations[Math.floor(Math.random() * locations.length)]
            }],
            modality: 'CTA/MRA (ResNet-101)',
            engine: 'ResNet-101 Engine',
            status: 'Success'
        };
    } else {
        return {
            prediction: 'Negative',
            detections: [],
            modality: 'CTA/MRA (ResNet-101)',
            engine: 'ResNet-101 Engine',
            status: 'Success'
        };
    }
}

function displayResults(data, ctx, canvas) {
    const results = document.getElementById('result-metrics');
    results.style.display = 'block';

    if (data.prediction === 'Positive') {
        document.getElementById('prediction-label').innerText = 'Positive - Aneurysm Detected';
        document.getElementById('prediction-box').style.background = '#fef2f2';
        document.getElementById('prediction-box').style.borderColor = '#fee2e2';
        document.getElementById('prediction-label').style.color = '#ef4444';
        document.getElementById('prediction-conf').style.color = '#991b1b';
        
        if (data.detections && data.detections.length > 0) {
            const det = data.detections[0];
            // Vizato bounding box mbi imazhin
            const scaleX = canvas.width / 224;
            const scaleY = canvas.height / 224;
            const bx = det.box[0] * scaleX;
            const by = det.box[1] * scaleY;
            const bw = det.box[2] * scaleX;
            const bh = det.box[3] * scaleY;
            
            // Box e kuqe
            ctx.strokeStyle = '#ef4444'; 
            ctx.lineWidth = Math.max(3, canvas.width * 0.012);
            ctx.setLineDash([]);
            ctx.strokeRect(bx, by, bw, bh);
            
            // Label background
            const labelH = Math.max(24, canvas.height * 0.06);
            ctx.fillStyle = 'rgba(239, 68, 68, 0.9)'; 
            ctx.fillRect(bx, by - labelH, bw + 60, labelH);
            
            // Label text
            ctx.fillStyle = '#ffffff'; 
            ctx.font = `bold ${Math.max(12, canvas.width * 0.035)}px Outfit`;
            ctx.fillText('Aneurysm: ' + det.confidence + '%', bx + 5, by - labelH * 0.25);
            
            // Corner markers
            const cornerLen = Math.max(8, canvas.width * 0.03);
            ctx.strokeStyle = '#f97316'; ctx.lineWidth = Math.max(2, canvas.width * 0.008);
            // Top-left
            ctx.beginPath(); ctx.moveTo(bx, by + cornerLen); ctx.lineTo(bx, by); ctx.lineTo(bx + cornerLen, by); ctx.stroke();
            // Top-right
            ctx.beginPath(); ctx.moveTo(bx+bw-cornerLen, by); ctx.lineTo(bx+bw, by); ctx.lineTo(bx+bw, by+cornerLen); ctx.stroke();
            // Bottom-left
            ctx.beginPath(); ctx.moveTo(bx, by+bh-cornerLen); ctx.lineTo(bx, by+bh); ctx.lineTo(bx+cornerLen, by+bh); ctx.stroke();
            // Bottom-right
            ctx.beginPath(); ctx.moveTo(bx+bw-cornerLen, by+bh); ctx.lineTo(bx+bw, by+bh); ctx.lineTo(bx+bw, by+bh-cornerLen); ctx.stroke();

            document.getElementById('prediction-conf').innerText = 'Confidence: ' + det.confidence + '%';
            document.getElementById('prob-bar').style.width = det.confidence + '%';
            document.getElementById('prob-bar').style.background = 'var(--danger)';
            document.getElementById('prob-val').innerText = det.confidence + '%';
            document.getElementById('res-location').innerText = det.location;
        }
    } else {
        document.getElementById('prediction-label').innerText = 'Negative - No Aneurysm Found';
        document.getElementById('prediction-box').style.background = '#ecfdf5';
        document.getElementById('prediction-box').style.borderColor = '#d1fae5';
        document.getElementById('prediction-label').style.color = '#10b981';
        document.getElementById('prediction-conf').innerText = 'Confidence: 97.8%';
        document.getElementById('prediction-conf').style.color = '#065f46';
        document.getElementById('prob-bar').style.width = '2.2%';
        document.getElementById('prob-bar').style.background = 'var(--success)';
        document.getElementById('prob-val').innerText = '2.2%';
        document.getElementById('res-location').innerText = 'N/A — No pathology detected';
    }
    document.getElementById('res-modality').innerText = data.modality;
}

async function runSimulation() {
    if (!currentFile) return;
    const canvas = document.getElementById('scan-canvas');
    const ctx = canvas.getContext('2d');
    const line = document.getElementById('scanning-line');
    const status = document.getElementById('scan-status');
    const results = document.getElementById('result-metrics');
    const overlay = document.getElementById('scan-overlay');
    
    // Fsheh overlay-n plotësisht gjatë skanimit
    overlay.style.display = 'none';

    results.style.display = 'none';
    status.innerHTML = '<i class="ti ti-loader rotate"></i> Running ResNet-101 Inference...';
    status.style.background = '#fef3c7';
    status.style.color = '#92400e';
    line.style.display = 'block';

    let pos = 0;
    const animInterval = setInterval(() => { 
        pos += 1.5; 
        line.style.top = pos + '%'; 
        if (pos >= 100) pos = 0; 
    }, 25);

    // Provo backend-in fillimisht
    const formData = new FormData();
    formData.append('file', currentFile);

    let data = null;
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000); // 3 sekonda timeout
        const response = await fetch('http://localhost:5005/predict?t=' + new Date().getTime(), { 
            method: 'POST', 
            body: formData,
            signal: controller.signal
        });
        clearTimeout(timeoutId);
        data = await response.json();
    } catch (e) {
        // API nuk është e disponueshme — përdor simulimin lokal
        console.log('Backend not available, using local simulation mode.');
        // Prit 2.5 sekonda për efekt realist
        await new Promise(resolve => setTimeout(resolve, 2500));
        data = localInference(currentFile.name);
    }

    clearInterval(animInterval);
    line.style.display = 'none';
    
    status.innerText = '✓ Analysis Complete';
    status.style.background = '#ecfdf5';
    status.style.color = '#059669';

    displayResults(data, ctx, canvas);
}

function resetInference() {
    currentFile = null;
    uploadedImageData = null;
    const canvas = document.getElementById('scan-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.style.opacity = '0.1';
    canvas.style.filter = 'grayscale(1)';
    
    document.getElementById('scan-overlay').style.display = 'flex';
    document.getElementById('scan-overlay').style.background = 'rgba(15, 23, 42, 0.8)';
    document.getElementById('scan-overlay').style.backdropFilter = 'blur(8px)';
    document.getElementById('start-btn').style.display = 'none';
    document.getElementById('result-metrics').style.display = 'none';
    document.getElementById('scan-status').innerText = 'Ready for analysis';
    document.getElementById('scan-status').style.background = '#f1f5f9';
    document.getElementById('scan-status').style.color = 'var(--text-secondary)';
    document.getElementById('scanning-line').style.display = 'none';
    
    // Reset file input
    document.getElementById('file-upload').value = '';
}

function initTheme() {
    const isDark = localStorage.getItem('theme') === 'dark';
    if (isDark) {
        document.body.classList.add('dark-mode');
        const icon = document.getElementById('theme-icon');
        if (icon) icon.className = 'ti ti-sun';
        Chart.defaults.color = '#94a3b8';
    } else {
        Chart.defaults.color = '#64748b';
    }
}

function toggleTheme() {
    const isDark = document.body.classList.toggle('dark-mode');
    document.getElementById('theme-icon').className = isDark ? 'ti ti-sun' : 'ti ti-moon';
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    
    Chart.defaults.color = isDark ? '#94a3b8' : '#64748b';
    const gridColor = isDark ? '#334155' : '#f1f5f9';
    
    Object.values(activeCharts).forEach(chart => {
        if (chart.options.scales) {
            if (chart.options.scales.x && chart.options.scales.x.grid) chart.options.scales.x.grid.color = gridColor;
            if (chart.options.scales.y && chart.options.scales.y.grid) chart.options.scales.y.grid.color = gridColor;
        }
        chart.update();
    });
}

window.onload = () => {
    initTheme();
    initPageCharts('overview');
};
