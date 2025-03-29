// Global variables
let model;
let labels = [];

// Load model and labels when page loads
window.addEventListener('DOMContentLoaded', async () => {
    console.log("DOM Content Loaded - Starting initialization");
    
    try {
        // Load metadata first to get labels
        console.log("Fetching metadata.json");
        const metadataResponse = await fetch('/model/metadata.json');
        console.log("Metadata response:", metadataResponse);
        
        if (!metadataResponse.ok) {
            throw new Error(`Failed to load metadata: ${metadataResponse.status} ${metadataResponse.statusText}`);
        }
        
        const metadata = await metadataResponse.json();
        console.log("Metadata loaded:", metadata);
        
        if (metadata.labels && metadata.labels.length > 0) {
            labels = metadata.labels;
            console.log('Labels loaded from metadata:', labels);
        } else {
            console.error('No labels found in metadata');
        }
        
        // Load TensorFlow.js model
        console.log("Loading TensorFlow.js model");
        model = await tf.loadLayersModel('/model/model.json');
        console.log('Model loaded successfully:', model);
    } catch (error) {
        console.error('Error during initialization:', error);
        alert('Error loading model: ' + error.message);
    }

    // Set up custom event listeners for button clicks
    setupEventListeners();
});

function setupEventListeners() {
    console.log("Setting up event listeners");
    
    // Handle submit photo button click
    document.addEventListener('submitPhoto', async function() {
        console.log("Submit photo event received");
        
        try {
            showLoading();
            const prediction = await predict(canvasElement);
            if (prediction) {
                // Convert canvas to blob and submit with prediction
                canvasElement.toBlob(function(blob) {
                    submitImageWithPrediction(blob, prediction.className, prediction.confidence);
                }, 'image/jpeg', 0.95);
            } else {
                hideLoading();
                alert('Could not analyze the image. Please try again.');
            }
        } catch (error) {
            hideLoading();
            console.error('Error analyzing image:', error);
            alert('Error analyzing image: ' + error.message);
        }
    });
    
    // Handle analyze image button click
    document.addEventListener('analyzeImage', async function(e) {
        console.log("Analyze image event received", e.detail);
        
        try {
            const imgElement = document.getElementById('image-preview');
            const prediction = await predict(imgElement);
            
            if (prediction) {
                submitImageWithPrediction(e.detail.fileInput.files[0], prediction.className, prediction.confidence, e.detail.query);
            } else {
                hideLoading();
                alert('Could not analyze the image. Please try again.');
            }
        } catch (error) {
            hideLoading();
            console.error('Error analyzing image:', error);
            alert('Error analyzing image: ' + error.message);
        }
    });
    
    // Handle ask question button click
    document.addEventListener('askQuestion', function(e) {
        console.log("Ask question event received", e.detail);
        submitTextQuery(e.detail.textQuery);
    });
    
    // File upload preview
    const fileUpload = document.getElementById('file-upload');
    if (fileUpload) {
        fileUpload.addEventListener('change', function(e) {
            console.log("File selected");
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const imagePreview = document.getElementById('image-preview');
                    imagePreview.src = e.target.result;
                    document.getElementById('preview-container').style.display = 'block';
                    console.log("Image preview displayed");
                }
                reader.readAsDataURL(file);
            }
        });
    } else {
        console.error("File upload element not found");
    }
    
    // Camera controls
    setupCameraControls();
}

function setupCameraControls() {
    console.log("Setting up camera controls");
    
    // Initialize camera when camera tab is opened
    document.getElementById('camera-tab').addEventListener('click', initCamera);

    // Stop camera when switching to other tabs
    document.getElementById('upload-tab').addEventListener('click', stopCamera);
    document.getElementById('text-tab').addEventListener('click', stopCamera);
    
    // Set up canvas and camera buttons
    const captureButton = document.getElementById('capture-btn');
    const retakeButton = document.getElementById('retake-btn');
    
    if (captureButton) {
        captureButton.addEventListener('click', function() {
            console.log("Capture button clicked");
            canvasElement.width = videoElement.videoWidth;
            canvasElement.height = videoElement.videoHeight;
            context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
            
            videoContainer.style.display = 'none';
            capturedPhoto.style.display = 'block';
            submitPhotoButton.style.display = 'block';
        });
    }
    
    if (retakeButton) {
        retakeButton.addEventListener('click', function() {
            console.log("Retake button clicked");
            capturedPhoto.style.display = 'none';
            videoContainer.style.display = 'block';
            submitPhotoButton.style.display = 'none';
        });
    }
    
    // Initialize camera on page load if camera tab is active
    if (document.getElementById('camera-tab').classList.contains('active')) {
        initCamera();
    }
}

// Camera variables
const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('canvas');
const submitPhotoButton = document.getElementById('submit-photo');
const videoContainer = document.getElementById('video-container');
const capturedPhoto = document.getElementById('captured-photo');
const locationInput = document.getElementById('location-input');
const cropTypeInput = document.getElementById('crop-type-input');

let stream;
const context = canvasElement ? canvasElement.getContext('2d') : null;

function initCamera() {
    console.log("Initializing camera");
    if (!videoElement) {
        console.error("Video element not found");
        return;
    }
    
    navigator.mediaDevices.getUserMedia({ 
        video: { 
            facingMode: 'environment',
            width: { ideal: 1280 },
            height: { ideal: 720 } 
        } 
    })
    .then(function(mediaStream) {
        stream = mediaStream;
        videoElement.srcObject = mediaStream;
        console.log("Camera initialized successfully");
    })
    .catch(function(error) {
        console.error('Could not access camera:', error);
        alert('Could not access camera. Please check permissions or try uploading an image instead.');
    });
}

function stopCamera() {
    console.log("Stopping camera");
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
}

// Process image through TensorFlow.js model
async function preprocessImage(imgElement) {
    console.log("Preprocessing image");
    return tf.tidy(() => {
        // Convert the image to a tensor
        const imgTensor = tf.browser.fromPixels(imgElement);
        console.log("Image converted to tensor", imgTensor.shape);
        
        // Resize the image to 224x224 (standard input size for many models)
        const resized = tf.image.resizeBilinear(imgTensor, [224, 224]);
        console.log("Image resized");
        
        // Normalize values to [-1, 1]
        const normalized = resized.toFloat().div(tf.scalar(127.5)).sub(tf.scalar(1));
        console.log("Image normalized");
        
        // Add batch dimension [1, 224, 224, 3]
        const batched = normalized.expandDims(0);
        console.log("Added batch dimension", batched.shape);
        
        return batched;
    });
}

// Run prediction
async function predict(imgElement) {
    console.log("Starting prediction on image");
    
    if (!model) {
        console.error('Model not loaded yet');
        alert('The disease detection model is not loaded yet. Please refresh the page and try again.');
        return null;
    }
    
    if (!imgElement) {
        console.error('Image element is null or undefined');
        alert('No image found to analyze.');
        return null;
    }
    
    try {
        // Preprocess the image
        console.log("Preprocessing image for prediction");
        const tensor = await preprocessImage(imgElement);
        
        // Run the prediction
        console.log("Running model prediction");
        const predictions = await model.predict(tensor);
        console.log("Raw prediction result:", predictions);
        
        // Get results
        const data = await predictions.data();
        console.log("Prediction data:", data);
        const predictionArray = Array.from(data);
        
        // Get highest confidence prediction
        const maxIndex = predictionArray.indexOf(Math.max(...predictionArray));
        const confidence = predictionArray[maxIndex];
        console.log("Highest confidence class:", maxIndex, "with confidence:", confidence);
        
        // Clean up tensors
        tensor.dispose();
        predictions.dispose();
        
        // Get the class name from labels
        let className = "Unknown";
        if (maxIndex < labels.length) {
            className = labels[maxIndex];
        }
        
        console.log("Prediction complete:", className, confidence);
        return {
            className: className,
            confidence: confidence
        };
    } catch (error) {
        console.error('Error during prediction:', error);
        throw new Error('Error analyzing image: ' + error.message);
    }
}

function submitImageWithPrediction(imageFile, className, confidence, query = '') {
    console.log("Submitting image with prediction:", className, confidence);
    
    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('predicted_class', className);
    formData.append('confidence', confidence);
    
    // Add location and crop type
    const location = locationInput ? locationInput.value.trim() : '';
    const cropType = cropTypeInput ? cropTypeInput.value.trim() : '';
    
    if (location) {
        formData.append('location', location);
    }
    if (cropType) {
        formData.append('crop_type', cropType);
    }
    
    if (query) {
        formData.append('query', query);
    }
    
    // Log form data contents
    for (let pair of formData.entries()) {
        console.log(pair[0] + ': ' + pair[1]);
    }
    
    console.log("Sending POST request to /detect");
    fetch('/detect', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        console.log("Response received:", response.status, response.statusText);
        if (!response.ok) {
            throw new Error(`Server error: ${response.status} ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        console.log("Response data:", data);
        hideLoading();
        displayResults(data);
    })
    .catch(error => {
        console.error('Error during fetch:', error);
        hideLoading();
        alert('Error processing image: ' + error.message);
    });
}

function submitTextQuery(query) {
    console.log("Submitting text query:", query);
    
    const formData = new FormData();
    formData.append('query', query);
    
    // Add location and crop type
    const location = locationInput ? locationInput.value.trim() : '';
    const cropType = cropTypeInput ? cropTypeInput.value.trim() : '';
    
    if (location) {
        formData.append('location', location);
    }
    if (cropType) {
        formData.append('crop_type', cropType);
    }
    
    console.log("Sending POST request to /detect for text query");
    fetch('/detect', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        console.log("Response received:", response.status, response.statusText);
        if (!response.ok) {
            throw new Error(`Server error: ${response.status} ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        console.log("Response data:", data);
        hideLoading();
        displayResults(data);
    })
    .catch(error => {
        console.error('Error during fetch:', error);
        hideLoading();
        alert('Error processing your question: ' + error.message);
    });
}

function showLoading() {
    console.log("Showing loading spinner");
    const loadingSpinner = document.querySelector('.loading-spinner');
    const resultCard = document.querySelector('.result-card');
    
    if (loadingSpinner) {
        loadingSpinner.style.display = 'block';
    }
    if (resultCard) {
        resultCard.style.display = 'none';
    }
}

function hideLoading() {
    console.log("Hiding loading spinner");
    const loadingSpinner = document.querySelector('.loading-spinner');
    const resultCard = document.querySelector('.result-card');
    
    if (loadingSpinner) {
        loadingSpinner.style.display = 'none';
    }
    if (resultCard) {
        resultCard.style.display = 'block';
    }
}

function displayResults(data) {
    console.log("Displaying results:", data);
    
    if (!data) {
        console.error("No data to display");
        alert("Error: No response data received");
        return;
    }
    
    if (data.error) {
        console.error("Error in response:", data.error);
        alert("Error: " + data.error);
        return;
    }
    
    const resultTitle = document.querySelector('.result-title');
    if (resultTitle) {
        resultTitle.innerHTML = data.type === 'image_analysis' 
            ? '<i class="fas fa-leaf me-2"></i> Disease Detection Results' 
            : '<i class="fas fa-question-circle me-2"></i> Query Results';
    }
    
    const diseaseResultSection = document.getElementById('disease-result');
    
    if (data.type === 'image_analysis' && diseaseResultSection) {
        diseaseResultSection.style.display = 'block';
        
        const diseaseNameTag = document.getElementById('disease-name-tag');
        if (diseaseNameTag) {
            diseaseNameTag.textContent = data.disease;
        }
        
        const confidencePercentage = (data.confidence * 100).toFixed(1);
        const confidenceBar = document.getElementById('confidence-bar');
        
        if (confidenceBar) {
            confidenceBar.style.width = confidencePercentage + '%';
            confidenceBar.textContent = confidencePercentage + '%';
            
            // Adjust color based on confidence
            if (confidencePercentage > 80) {
                confidenceBar.style.backgroundColor = '#198754'; // Green
            } else if (confidencePercentage > 60) {
                confidenceBar.style.backgroundColor = '#ffc107'; // Yellow
            } else {
                confidenceBar.style.backgroundColor = '#dc3545'; // Red
            }
        }
    } else if (diseaseResultSection) {
        diseaseResultSection.style.display = 'none';
    }
    
    // Format and display LLM response (process markdown-like formatting)
    const responseContent = document.getElementById('response-content');
    if (responseContent && data.llm_response) {
        let formattedResponse = data.llm_response
            .replace(/\n\n/g, '</p><p>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/^# (.*?)$/gm, '<h1>$1</h1>')
            .replace(/^## (.*?)$/gm, '<h2>$1</h2>')
            .replace(/^### (.*?)$/gm, '<h3>$1</h3>');
            
        responseContent.innerHTML = '<p>' + formattedResponse + '</p>';
    } else if (responseContent) {
        responseContent.innerHTML = '<p>No analysis available.</p>';
    }
    
    // Scroll to results
    const resultCard = document.querySelector('.result-card');
    if (resultCard) {
        resultCard.scrollIntoView({ behavior: 'smooth' });
    }
    
    console.log("Results displayed successfully");
}

// Add a simple test function for debugging
window.testConnection = function() {
    fetch('/detect', {
        method: 'POST',
        body: new FormData()
    })
    .then(response => response.json())
    .then(data => console.log("Test connection response:", data))
    .catch(error => console.error("Test connection error:", error));
};

console.log("main.js fully loaded");