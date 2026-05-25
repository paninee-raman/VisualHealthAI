// UI Element Handles
const modalOverlay = document.getElementById('modalOverlay');
const openModalBtn = document.getElementById('openModalBtn');
const closeModalBtn = document.getElementById('closeModalBtn');
const cancelBtn = document.getElementById('cancelBtn');
const doneBtn = document.getElementById('doneBtn');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const dropzoneEmpty = document.getElementById('dropzoneEmpty');
const dropzonePreview = document.getElementById('dropzonePreview');
const imagePreview = document.getElementById('imagePreview');
const fileNameTxt = document.getElementById('fileNameTxt');
const runAnalysisBtn = document.getElementById('runAnalysisBtn');
const btnText = document.getElementById('btnText');
const uploadScreen = document.getElementById('uploadScreen');
const resultScreen = document.getElementById('resultScreen');

// Output Targets
const resultCondition = document.getElementById('resultCondition');
const resultConfidence = document.getElementById('resultConfidence');
const resultId = document.getElementById('resultId');

let selectedFile = null;

// Modal Open/Close Controls
openModalBtn.addEventListener('click', () => {
    modalOverlay.classList.remove('hidden');
    setTimeout(() => modalOverlay.classList.remove('opacity-0'), 10);
    resetModal();
});

function closeModal() {
    modalOverlay.classList.add('opacity-0');
    setTimeout(() => modalOverlay.classList.add('hidden'), 200);
}

closeModalBtn.addEventListener('click', closeModal);
cancelBtn.addEventListener('click', closeModal);
doneBtn.addEventListener('click', closeModal);

// File Upload Triggers
dropzone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', function() {
    if(this.files && this.files[0]) handleFile(this.files[0]);
});

function handleFile(file) {
    selectedFile = file;
    fileNameTxt.innerText = `${file.name}`;
    const reader = new FileReader();
    reader.onload = function(e) {
        imagePreview.src = e.target.result;
        dropzoneEmpty.classList.add('hidden');
        dropzonePreview.classList.remove('hidden');
        
        // Light up the Run Analysis button (turns green)
        runAnalysisBtn.disabled = false;
        runAnalysisBtn.className = "flex-1 py-3 px-4 rounded-xl text-sm font-semibold text-black bg-emerald-400 hover:bg-emerald-300 cursor-pointer shadow-lg shadow-emerald-400/10 transition-all flex items-center justify-center gap-2";
    }
    reader.readAsDataURL(file);
}

// Simulated Analysis Flow (Bypasses Python backend)
runAnalysisBtn.addEventListener('click', async () => {
    if(!selectedFile || runAnalysisBtn.disabled) return;
    runAnalysisBtn.disabled = true;
    runAnalysisBtn.className =
    "flex-1 py-3 px-4 rounded-xl text-sm font-medium text-gray-400 bg-[#111222] border border-gray-800 cursor-not-allowed transition-all flex items-center justify-center gap-2";
    btnText.innerHTML =
    `<i class="fa-solid fa-spinner animate-spin text-xs"></i> Analysing...`;
    const formData = new FormData();
    formData.append('image', selectedFile);
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if(data.status === 'success') {
            resultCondition.innerText = data.condition;
            resultConfidence.innerText = `${data.confidence}%`;
            resultId.innerText = data.scan_id;
            uploadScreen.classList.add('hidden');
            resultScreen.classList.remove('hidden');
        } else {
            alert(data.error || "Analysis failed.");
        }
    } catch(error) {
        console.error(error);
        alert("Backend connection failed.");
    } finally {
        runAnalysisBtn.disabled = false;
    }
});

function resetModal() {
    uploadScreen.classList.remove('hidden');
    resultScreen.classList.add('hidden');
    fileInput.value = '';
    selectedFile = null;
    dropzoneEmpty.classList.remove('hidden');
    dropzonePreview.classList.add('hidden');
    imagePreview.src = '';
    runAnalysisBtn.disabled = true;
    runAnalysisBtn.className = "flex-1 py-3 px-4 rounded-xl text-sm font-medium text-gray-500 bg-gray-900 border border-gray-800/50 cursor-not-allowed transition-all flex items-center justify-center gap-2";
    btnText.innerHTML = "Run Analysis";
}