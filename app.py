import streamlit as st
import os
import json
import requests
import numpy as np
from PIL import Image, ImageOps
import tempfile

# Set page config
st.set_page_config(
    page_title="Crop Disease Detection Assistant",
    page_icon="🌿",
    layout="wide"
)

# Load the labels from metadata.json
try:
    with open(os.path.join('model', 'metadata.json'), 'r') as f:
        metadata = json.load(f)
        if 'labels' in metadata:
            class_names = metadata['labels']
            st.session_state['labels'] = class_names
            print(f"Loaded {len(class_names)} labels from metadata.json")
        else:
            class_names = []
            print("No labels found in metadata.json")
except FileNotFoundError:
    print("Warning: metadata.json file not found. Using empty labels list.")
    class_names = []

# Get OpenAI API key from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

def get_openai_response(disease_name, confidence_score, location="the area", crop_type="this crop"):
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
        query = st.session_state.get('query', "crop diseases")
        prompt = f"User asked about plant diseases: {query}\n\nProvide helpful information for farmers about this query, including treatment options and when to consult experts."
    
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
        response = requests.post(OPENAI_API_URL, headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            return f"Error from OpenAI API: {response.status_code} - Please check your API key and try again."
            
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error getting analysis: {str(e)}"

# Streamlit UI
st.title("🌿 Crop Disease Detection Assistant")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📷 Camera", "📤 Upload", "❓ Ask Question"])

# Sidebar for location and crop type
with st.sidebar:
    st.header("Location & Crop Information")
    location = st.text_input("Your Location", placeholder="e.g., Punjab, Maharashtra, etc.")
    crop_type = st.text_input("Crop Type", placeholder="e.g., Tomato, Apple, Wheat, etc.")

# Ask Question Tab
with tab3:
    st.header("Ask about crop diseases")
    query = st.text_area("Your question", placeholder="E.g., What are the symptoms of tomato early blight?")
    
    if st.button("Ask Question", key="ask_btn"):
        if query:
            st.session_state['query'] = query
            with st.spinner("Getting information..."):
                response = get_openai_response(None, None, location, crop_type)
                st.session_state['response'] = response
                st.session_state['show_results'] = True
                st.session_state['is_image'] = False
        else:
            st.warning("Please enter a question")

# Upload Tab
with tab2:
    st.header("Upload crop image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    optional_query = st.text_input("Optional question", placeholder="E.g., How do I treat this disease?")
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Note: In a real implementation, you would process the image with TensorFlow.js
        # For this example, we'll simulate a prediction
        if st.button("Analyze Image"):
            # Simulate prediction (in a real app, this would come from your model)
            import random
            random_index = random.randint(0, len(class_names)-1) if class_names else 0
            disease_name = class_names[random_index] if class_names else "Sample Disease"
            confidence = random.uniform(0.75, 0.98)
            
            with st.spinner("Analyzing image..."):
                response = get_openai_response(disease_name, confidence, location, crop_type)
                
                st.session_state['disease'] = disease_name
                st.session_state['confidence'] = confidence
                st.session_state['response'] = response
                st.session_state['show_results'] = True
                st.session_state['is_image'] = True

# Camera Tab (Note: This is just a placeholder as Streamlit doesn't support direct camera access in the same way)
with tab1:
    st.header("Take a photo")
    st.info("Camera functionality is limited in Streamlit Cloud. Please use the Upload tab to analyze an image.")
    
    # In a full implementation, you might use a third-party service or JavaScript component
    # to capture camera images, but that's beyond the scope of this example

# Display results if available
if st.session_state.get('show_results', False):
    st.header("Analysis Results")
    
    if st.session_state.get('is_image', False):
        # Image analysis results
        st.subheader(f"Detected Disease: {st.session_state['disease']}")
        st.progress(st.session_state['confidence'])
        st.write(f"Confidence: {st.session_state['confidence']*100:.1f}%")
    
    st.subheader("Expert Analysis:")
    st.markdown(st.session_state['response'])
    
    st.success("Always consult with agricultural experts for professional advice.")