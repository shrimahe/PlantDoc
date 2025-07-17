import json
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import os
import uuid
from predict import predict_image

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Required for flash messages

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load remedies
with open('static/remedies.json') as f:
    remedies = json.load(f)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
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
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join('static/uploads', unique_filename)
        file.save(filepath)

        prediction = predict_image(filepath)

        if prediction['status'] == 'success':
            class_name = prediction['class_name']
            remedy = remedies.get(class_name, {"remedy": "No remedy found", "links": [], "contact_support": ""})
        else:
            remedy = {"remedy": "Prediction failed", "links": [], "contact_support": ""}

        image_path = f"uploads/{unique_filename}"

        return render_template('results.html',
                               prediction=prediction,
                               remedy=remedy,
                               image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
