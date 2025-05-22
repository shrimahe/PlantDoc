import numpy as np
import json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# ✅ Load model and class labels ONCE at the top
model_path = "PlantDoc_ver_2_model_final.h5"
class_labels_path = "static/class_labels.json"
img_size = (224, 224)

# Load model
model = load_model(model_path)

# Load class labels
with open(class_labels_path) as cl:
    inv_class_indices = {v: k for k, v in json.load(cl).items()}


def predict_image(img_path, top_k=3):
    """
    Predicts plant disease from an image using the trained model
    """
    try:
        # Process image
        img_array = img_to_array(load_img(img_path, target_size=img_size)) / 255.0

        # Make prediction
        predictions = model.predict(np.expand_dims(img_array, 0))[0]

        # Get top k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]

        # Format results
        top_predictions = {inv_class_indices[i]: float(predictions[i]) * 100 for i in top_indices}

        return {
            "status": "success",
            "class_name": inv_class_indices[top_indices[0]],
            "confidence": float(predictions[top_indices[0]]) * 100,
            "top_predictions": top_predictions
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
