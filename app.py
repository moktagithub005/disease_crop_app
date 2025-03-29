from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import requests
from werkzeug.utils import secure_filename
import numpy as np
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()  # Load variables from .env file

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the labels from metadata.json
try:
    with open(os.path.join('model', 'metadata.json'), 'r') as f:
        metadata = json.load(f)
        if 'labels' in metadata:
            class_names = metadata['labels']
            print(f"Loaded {len(class_names)} labels from metadata.json")
        else:
            class_names = []
            print("No labels found in metadata.json")
except FileNotFoundError:
    print("Warning: metadata.json file not found. Using empty labels list.")
    class_names = []

# Configure OpenAI API
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY environment variable not set or empty")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_openai_response(disease_name, confidence_score, user_query=None):
    print(f"Getting OpenAI response for: {disease_name} with confidence {confidence_score}")
    location = user_query.get('location', 'the area') if isinstance(user_query, dict) else 'the area'
    crop_type = user_query.get('crop_type', 'this crop') if isinstance(user_query, dict) else 'this crop'
    
    # Create prompt based on available information
    if disease_name:
        prompt = f"""
Plant disease detected: {disease_name} with {confidence_score:.2f} confidence.

1. Provide a detailed description of this disease and its symptoms
2. Explain the causes and conditions that favor this disease
3. Suggest organic and chemical treatment options for farmers
4. Recommend preventive measures farmers can take
5. Suggest when farmers should consult with agricultural experts in {location}
6. Provide any additional information specific to {crop_type} that would be helpful

Keep your answer concise, practical and farmer-friendly.
"""
    else:
        text_query = user_query if isinstance(user_query, str) else "crop diseases"
        prompt = f"User asked about plant diseases: {text_query}\n\nProvide helpful information for farmers about this query, including treatment options and when to consult experts."
    
    print(f"OpenAI Prompt: {prompt[:100]}...")
    
    # OpenAI API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    payload = {
        "model": "gpt-4o",  # Using GPT-4o for better analysis
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 800,
        "temperature": 0.7
    }
    
    try:
        print("Sending request to OpenAI API")
        response = requests.post(OPENAI_API_URL, headers=headers, data=json.dumps(payload))
        print(f"OpenAI API response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"OpenAI API error: {response.text}")
            return f"Error from OpenAI API: {response.status_code} - Please check your API key and try again."
            
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Exception during OpenAI API call: {str(e)}")
        traceback.print_exc()
        return f"Error getting analysis: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    print("\n===== Request received at /detect =====")
    print(f"Form data: {request.form}")
    print(f"Files: {request.files.keys()}")
    
    try:
        # Get location and crop type if provided
        location = request.form.get('location', '')
        crop_type = request.form.get('crop_type', '')
        user_query = {'location': location, 'crop_type': crop_type}
        
        print(f"Location: {location}, Crop Type: {crop_type}")
        
        # Process text-only queries
        if 'file' not in request.files:
            print("Processing text-only query")
            text_query = request.form.get('query', '')
            
            if not text_query and not (location or crop_type):
                print("Error: No file or text query provided")
                return jsonify({"error": "No file or text query provided"}), 400
            
            # Text-only query to OpenAI
            print(f"Text query: {text_query}")
            if text_query:
                llm_response = get_openai_response(None, None, text_query)
            else:
                llm_response = get_openai_response(None, None, user_query)
                
            response_data = {
                "type": "text_only",
                "query": text_query or f"Information about crops in {location}" if location else "General crop information",
                "llm_response": llm_response
            }
            
            print(f"Sending text query response: {response_data['type']}")
            return jsonify(response_data)
        
        # Process image files
        print("Processing image file")
        file = request.files['file']
        
        if file.filename == '':
            print("Error: No selected file")
            return jsonify({"error": "No selected file"}), 400
        
        if file and allowed_file(file.filename):
            print(f"Saving file: {file.filename}")
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print(f"File saved to: {file_path}")
            
            # For image analysis with prediction from frontend
            disease_name = request.form.get('predicted_class', '')
            confidence_str = request.form.get('confidence', '0')
            
            print(f"Prediction from client - Class: {disease_name}, Confidence: {confidence_str}")
            
            try:
                confidence_score = float(confidence_str)
            except ValueError:
                print(f"Error converting confidence to float: {confidence_str}")
                confidence_score = 0
            
            if disease_name and confidence_score > 0:
                print("Getting OpenAI response for disease prediction")
                # Get OpenAI response based on the model's prediction
                llm_response = get_openai_response(disease_name, confidence_score, user_query)
                
                response_data = {
                    "type": "image_analysis",
                    "disease": disease_name,
                    "confidence": confidence_score,
                    "crop_type": crop_type,
                    "location": location,
                    "llm_response": llm_response,
                    "file_url": f"/uploads/{filename}"
                }
                
                print(f"Sending image analysis response for disease: {disease_name}")
                return jsonify(response_data)
            else:
                print("Error: No valid prediction from client")
                return jsonify({
                    "error": "Could not get prediction from client. Please try again."
                }), 400
        
        print("Error: File type not allowed")
        return jsonify({"error": "File type not allowed"}), 400
    
    except Exception as e:
        print(f"Exception in /detect route: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Serve model files for TensorFlow.js
@app.route('/model/<path:path>')
def serve_model(path):
    print(f"Serving model file: {path}")
    return send_from_directory('model', path)

# Add a test endpoint for debugging
@app.route('/test')
def test():
    return jsonify({
        "status": "success",
        "message": "Test endpoint is working",
        "api_key_set": bool(OPENAI_API_KEY),
        "labels_count": len(class_names)
    })

if __name__ == '__main__':
    app.run(debug=True)