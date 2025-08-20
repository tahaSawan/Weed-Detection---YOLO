"""
Weed Detection Web Application
A simple Flask web app for uploading images and detecting weeds using YOLO.
"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import base64
from datetime import datetime
import json

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Initialize YOLO model
model = None

def load_model():
    """Load the trained YOLO model."""
    global model
    model_path = '../models/weed_detector/weights/best.pt'
    
    if os.path.exists(model_path):
        model = YOLO(model_path)
        print(f"✅ Model loaded from: {model_path}")
        return True
    else:
        print(f"❌ Model not found: {model_path}")
        print("Please train a model first using the training script.")
        return False

def allowed_file(filename):
    """Check if uploaded file is allowed."""
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def detect_weeds_in_image(image_path, confidence=0.5):
    """Detect weeds in an image using YOLO."""
    if model is None:
        return None, "Model not loaded"
    
    try:
        # Run inference
        results = model(image_path, conf=confidence)
        result = results[0]
        
        # Extract detections
        detections = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                detection = {
                    'bbox': box.tolist(),
                    'confidence': float(conf),
                    'class_id': int(class_id),
                    'class_name': 'weed'
                }
                detections.append(detection)
        
        # Save result image
        result_path = os.path.join(app.config['RESULTS_FOLDER'], 
                                 f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        result.save(result_path)
        
        return detections, result_path
        
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and weed detection."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get confidence threshold
        confidence = float(request.form.get('confidence', 0.5))
        
        # Detect weeds
        detections, result_path = detect_weeds_in_image(filepath, confidence)
        
        if detections is None:
            return jsonify({'error': result_path}), 500
        
        # Prepare response
        response = {
            'success': True,
            'detections': detections,
            'weed_count': len(detections),
            'result_image': result_path,
            'original_image': filepath
        }
        
        return jsonify(response)
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/result/<filename>')
def get_result_image(filename):
    """Serve result images."""
    return send_file(os.path.join(app.config['RESULTS_FOLDER'], filename))

@app.route('/upload/<filename>')
def get_upload_image(filename):
    """Serve uploaded images."""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/api/detect', methods=['POST'])
def api_detect():
    """API endpoint for weed detection."""
    try:
        data = request.get_json()
        image_data = data.get('image')  # Base64 encoded image
        confidence = float(data.get('confidence', 0.5))
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        
        # Save temporary image
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                               f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
        
        # Detect weeds
        detections, result_path = detect_weeds_in_image(temp_path, confidence)
        
        if detections is None:
            return jsonify({'error': result_path}), 500
        
        # Read result image as base64
        with open(result_path, 'rb') as f:
            result_image_data = base64.b64encode(f.read()).decode('utf-8')
        
        response = {
            'success': True,
            'detections': detections,
            'weed_count': len(detections),
            'result_image': f"data:image/jpeg;base64,{result_image_data}"
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    """Check if model is loaded."""
    return jsonify({
        'model_loaded': model is not None,
        'model_path': '../models/weed_detector/weights/best.pt' if model else None
    })

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        print("🚀 Starting web application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model. Please train a model first.")
