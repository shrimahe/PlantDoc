from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from predict import predict_image

app = Flask(__name__)

def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224))  # Match model's expected sizing
    img_array = np.array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)


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

    # Save temporarily
    filename = secure_filename(file.filename)
    filepath = os.path.join('static/uploads', filename)
    file.save(filepath)

    prediction = predict_image(filepath)

    # Cleanup
    os.remove(filepath)

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)