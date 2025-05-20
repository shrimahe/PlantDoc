import json
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import os
import uuid
from predict import predict_image

app = Flask(__name__)

# Load remedies
with open('static/remedies.json') as f:
    remedies = json.load(f)

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'})
    if file:
        filename = secure_filename(file.filename)
        # Create a unique filename to avoid conflicts
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join('static/uploads', unique_filename)
        file.save(filepath)
        
        prediction = predict_image(filepath)
        
        if prediction['status'] == 'success':
            class_name = prediction['class_name']
            remedy = remedies.get(class_name, {"remedy": "No remedy found", "links": [], "contact_support": ""})
        else:
            remedy = {"remedy": "Prediction failed", "links": [], "contact_support": ""}
        
        # Keep the file for display and pass its path to the template
        image_path = f"uploads/{unique_filename}"
        
        # Render the results page with all needed data
        return render_template('results.html', 
                              prediction=prediction, 
                              remedy=remedy,
                              image_path=image_path)
    return None


@app.route('/cleanup/<filename>')
def cleanup(filename):
    try:
        os.remove(os.path.join('static/uploads', filename))
    except:
        pass
    return redirect(url_for('home'))

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)