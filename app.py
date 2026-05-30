import os
import numpy as np
import secrets
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageFilter

# Suppress TensorFlow startup logs for a cleaner terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Prevents cross-origin blocking if testing paths vary

# 1. Load your H5 deep learning model on startup
MODEL_PATH = 'skin_cancer_model.h5'

if os.path.exists(MODEL_PATH):
    print(f"--> Loading Machine Learning Engine [{MODEL_PATH}]...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("--> Model successfully loaded into memory.")
else:
    print(f"⚠️ CRITICAL WARNING: '{MODEL_PATH}' not found!")
    print("Please run the 'generate_mock_model.py' script first to create it.")
    model = None

@app.route('/')
def home():
    # Serves your local index.html interface from the /templates directory
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if model is None:
        return jsonify({'error': 'Machine learning engine is offline. H5 file missing.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image block detected in request context.'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty image payload sequence.'}), 400

    try:
        # 2. Image Preprocessing
        # Open the image file directly from the stream into Pillow
        img = Image.open(file.stream).convert('RGB')
        # Deblur / sharpen image
        img = img.filter(
            ImageFilter.UnsharpMask(
                radius=1.5,
                percent=120,
                threshold=3
                )
            )
        # Resize image to match your model's input dimension shapes
        img = img.resize((224, 224))
        
        # Convert image bytes to a normalized NumPy matrix float array [0, 1]
        img_array = np.array(img) / 255.0
        
        # Expand dimensions to fit batch processing format: (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # 3. Compile Model Inference
        predictions = model.predict(img_array)
        
        # Assuming a binary classification model out-parameter (e.g., Sigmoid output between 0 and 1)
        # 0 = Benign, 1 = Malignant/Cancerous
        raw_score = float(predictions[0][0])
        
        THRESHOLD = 0.7
        if raw_score >= THRESHOLD:
            condition = "Benign"
            confidence = round(raw_score * 100)
        else:
            condition = "Cancerous"
            confidence = round((1 - raw_score) * 100)
        # 4. Return Data Payload matching your interface fields
        mock_id = f"F8260B6{secrets.token_hex(3).upper()}"
        
        return jsonify({
            "status": "success",
            "scan_id": mock_id,
            "condition": condition,
            "confidence": confidence
        }), 200

    except Exception as e:
        print(f"Error handling image inference computation: {str(e)}")
        return jsonify({'error': f'Inference engine pipeline error: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )