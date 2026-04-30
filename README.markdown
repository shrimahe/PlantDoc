# PlantDoc: AI-Powered Plant Disease and Deficiency Identifier

PlantDoc is a web-based application designed to identify plant diseases and deficiencies from leaf images using a Convolutional Neural Network (CNN). The system processes uploaded images, predicts potential diseases, and suggests remedies, making it a valuable tool for farmers, gardeners, and agricultural professionals. Built with Python, TensorFlow-Keras, and Flask, it provides a user-friendly interface and high-accuracy predictions.

## Features
- **Image-Based Disease Detection**: Upload leaf images to detect plant diseases with a trained CNN model.
- **Remedy Suggestions**: Provides curated remedies for identified diseases, including organic and conventional approaches.
- **Responsive Web Interface**: Built with HTML5, CSS3, Bootstrap, and Flask for cross-device compatibility.
- **High Accuracy**: Achieves 98% weighted average accuracy across 25,664 test images.
- **Scalable Backend**: Modular Flask architecture supports future enhancements like cloud hosting or database integration

## Project Structure
```
PlantDoc/
├── static/
│   ├── uploads/              # Stores uploaded images
│   ├── class_labels.json     # Class label mappings
│   ├── remedies.json         # Remedy data for diseases
├── templates/
│   ├── index.html            # Home page
│   ├── results.html          # Prediction results page
├── predict.py                # Image prediction logic
├── PlantDocNet_train.py      # Model training script
├── app.py                    # Flask web application
├── PlantDocNet_v1.h5         # Trained CNN model
└── README.md                 
```

## Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- Flask
- NumPy
- PIL (Pillow)

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Shrimahe-16/PlantDoc.git
   cd PlantDoc
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Pre-trained Model**:
   - Place `PlantDocNet_v1.h5` in the project root.
   - Ensure `class_labels.json` and `remedies.json` are in the `static/` directory.

4. **Run the Application**:
   ```bash
   python app.py
   ```
   - The app runs in debug mode by default. Access it at `http://127.0.0.1:5000`.

## Usage
1. **Access the Web App**:
   - Open `http://127.0.0.1:5000` in a browser.
   - Navigate to the home page and upload a plant leaf image (JPG, PNG, or JPEG).

2. **View Results**:
   - The app processes the image, displays the predicted disease, confidence score, and top-3 predictions.
   - Suggested remedies are shown based on the identified disease.

## Technical Details

### Model Architecture
The CNN model is implemented using TensorFlow-Keras with the following layers:
- **Convolutional Layers**: 
  - 32 filters (3x3, ReLU) → MaxPooling (2x2)
  - 64 filters (3x3, ReLU) → MaxPooling (2x2)
  - 128 filters (3x3, ReLU) → MaxPooling (2x2)
  - 256 filters (3x3, ReLU) → MaxPooling (2x2)
- **Flatten Layer**: Converts feature maps to a 1D vector.
- **Dense Layers**: 
  - 512 neurons (ReLU) with 50% Dropout to prevent overfitting.
  - Output layer with softmax activation for multi-class classification.
- **Optimizer**: Adam (learning rate=0.001).
- **Loss Function**: Categorical Crossentropy.
- **Metrics**: Accuracy.

### Training
- **Dataset**: 131,457 leaf images (80% training, 20% validation) from the PlantVillage dataset.
- **Data Augmentation**: Rotation, zoom, shear, flip, and shift to enhance model robustness.
- **Training Parameters**:
  - Batch Size: 32
  - Epochs: 30
  - Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Performance**: 98% validation accuracy, 0.98 weighted F1-score across 31 classes.

### Prediction Pipeline
- **Image Preprocessing**: Resized to 224x224, normalized to [0,1].
- **Prediction**: Uses `predict_image()` in `predict.py` to return top-3 disease predictions with confidence scores.
- **Error Handling**: Returns user-friendly error messages for invalid inputs or processing failures.

### Web Application
- **Frontend**: HTML5, CSS3, Bootstrap for responsive design.
- **Backend**: Flask handles image uploads, model inference, and dynamic result rendering.
- **File Management**: Uploaded images are stored temporarily in `static/uploads` and deleted after processing.

## Model Performance
- **Accuracy**: 98% on 25,664 validation images.
- **F1-Score**: 0.98 weighted average, with 16/31 classes achieving perfect F1-scores.
- **Confusion Matrix**: Strong diagonal concentration, minimal misclassifications (e.g., 46 instances of chili_murdaComplex misclassified as chili_healthy).

## References
- Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using deep learning for image-based plant disease detection*. Frontiers in Plant Science, 7, 1419.
- Ferentinos, K. P. (2018). *Deep learning models for plant disease detection and diagnosis*. Computers and Electronics in Agriculture, 145, 311-318.
- TensorFlow Documentation: https://www.tensorflow.org
- Keras ImageDataGenerator API: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
- Adam Optimizer: Kingma, D. P., & Ba, J. (2015). *A Method for Stochastic Optimization*.
