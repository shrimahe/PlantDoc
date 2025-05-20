import numpy as np
import json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

def predict_single_image(image_path, model_path, class_labels_path, img_size=(224, 224), top_k=3):
    model = load_model(model_path)
    
    with open(class_labels_path) as cl:
        inv_class_indices = {v: k for k, v in json.load(cl).items()}

    # Process image
    img_array = img_to_array(load_img(image_path, target_size=img_size)) / 255.0
    
    # prediction
    predictions = model.predict(np.expand_dims(img_array, 0))[0]
    
    # indices of top k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    # corresponding class names and confidences
    top_predictions = {inv_class_indices[i]: float(predictions[i]) * 100 for i in top_indices}
    
    return {
        "class_name": inv_class_indices[top_indices[0]],
        "confidence": float(predictions[top_indices[0]]) * 100,
        "top_predictions": top_predictions
    }

if __name__ == "__main__":
    image_path = "S:\\4th_sem\\PlantDoc\\Dataset\\Test\\TomatoYellowCurlVirus1.JPG"  #image_path
    model_path = "S:\\4th_sem\\PlantDoc\\PlantDoc_ver_2_model.h5"                    #model_path
    class_labels_path = "S:\\4th_sem\\PlantDoc\\class_labels.json"                  #class_labels_path

    result = predict_single_image(image_path, model_path, class_labels_path)
    print(f"Predicted class: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.2f}%")

    print("\nTop predictions:")
    for class_name, confidence in result['top_predictions'].items():
        print(f"{class_name}: {confidence:.2f}%")