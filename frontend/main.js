const fileInput = document.getElementById('fileInput');
const previewImage = document.getElementById('previewImage');
const heatmapImage = document.getElementById('heatmapImage');
const loading = document.getElementById('loading');
const uploadStatusText = document.getElementById('uploadStatusText');

// UI Elements
const statusBanner = document.getElementById('statusBanner');
const diagnosisLabel = document.getElementById('diagnosisLabel');
const confidenceBadge = document.getElementById('confidenceBadge');
const confidenceBar = document.getElementById('confidenceBar');
const insightRegion = document.getElementById('insightRegion');
const insightDesc = document.getElementById('insightDesc');
const insightPart = document.getElementById('insightPart');
const downloadPdfBtn = document.getElementById('downloadPdfBtn');
const reportL1 = document.getElementById('reportL1');
const reportL2 = document.getElementById('reportL2');

// Trigger file select via the wrapped input inside gradient container
fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
        handleFile(fileInput.files[0]);
    }
});

function handleFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
    };
    reader.readAsDataURL(file);

    analyzeImage(file);
}

async function analyzeImage(file) {
    loading.style.display = 'block';
    uploadStatusText.style.display = 'none';
    downloadPdfBtn.style.display = 'none';

    // Clear prev heatmap
    heatmapImage.src = "https://via.placeholder.com/400x400/141b22/8b949e?text=Analyzing...";
    confidenceBar.style.width = '0%';
    confidenceBadge.innerText = '0.0%';
    statusBanner.className = 'alert-box neutral';
    diagnosisLabel.innerText = 'Analysis in Progress...';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("Server error");
        const data = await response.json();
        
        setTimeout(() => { displayResults(data); }, 800);

    } catch (error) {
        console.error(error);
        alert("Error analyzing image: " + error.message);
        loading.style.display = 'none';
        uploadStatusText.style.display = 'block';
    }
}

function displayResults(data) {
    loading.style.display = 'none';
    uploadStatusText.style.display = 'block';
    uploadStatusText.innerText = "Scan Analysis Complete";

    // Counter Animation
    animateValue(confidenceBadge, 0, parseFloat(data.confidence), 1500, "%");
    setTimeout(() => { confidenceBar.style.width = `${data.confidence}%`; }, 100); 

    // Banner Logic
    statusBanner.className = 'alert-box';
    if (data.is_tumor) {
        statusBanner.classList.add('warning');
        statusBanner.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> <span id="diagnosisLabel">Anomaly Detected: ${data.label}</span>`;
        
        if (data.tumor_info) {
            insightRegion.innerText = data.tumor_info.region;
            insightPart.innerText = data.tumor_info.part;
            insightDesc.innerText = data.tumor_info.desc;
        }
        reportL1.innerText = `Identification of pattern consistent with ${data.label} (Confidence: ${data.confidence}%).`;
        reportL2.innerText = `Signal density concentrated around ${data.tumor_info ? data.tumor_info.part : 'brain tissue'}.`;
    } else {
        statusBanner.classList.add('success');
        statusBanner.innerHTML = `<i class="fa-solid fa-circle-check"></i> <span id="diagnosisLabel">Scan Appears Normal</span>`;
        insightRegion.innerText = "N/A";
        insightPart.innerText = "N/A";
        insightDesc.innerText = "The neural network identified healthy brain tissue volume and structure.";
        reportL1.innerText = "No localized masses or neoplastic anomalies verified.";
        reportL2.innerText = "Symmetric ventricular distribution maintained.";
    }

    // Heatmap Update
    if (data.heatmap_base64) {
        heatmapImage.src = `data:image/jpeg;base64,${data.heatmap_base64}`;
    } else {
        heatmapImage.src = previewImage.src; 
    }
    
    // PDF Download
    if (data.report_base64) {
        downloadPdfBtn.style.display = 'block';
        downloadPdfBtn.onclick = () => {
            const link = document.createElement('a');
            link.href = `data:application/pdf;base64,${data.report_base64}`;
            link.download = `NeuroScan_Report_${data.label ? data.label : 'Normal'}.pdf`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        };
    }
}

// Hardware-accelerated number counter
function animateValue(obj, start, end, duration, suffix) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const easeOut = 1 - Math.pow(1 - progress, 4);
        const current = (start + (end - start) * easeOut).toFixed(1);
        obj.innerHTML = current + suffix;
        if (progress < 1) window.requestAnimationFrame(step);
        else obj.innerHTML = end.toFixed(1) + suffix;
    };
    window.requestAnimationFrame(step);
}
