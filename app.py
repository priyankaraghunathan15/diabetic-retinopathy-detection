import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
from PIL import Image
import keras
import gradio as gr

CLASS_LABELS = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

model = keras.models.load_model('models/diabetic_retinopathy_model.keras')

def predict(image):
    image = image.resize((224, 224)).convert('RGB')
    img_array = np.array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    probs = model.predict(img_array, verbose=0)[0]
    return {CLASS_LABELS[i]: float(probs[i]) for i in range(len(CLASS_LABELS))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Retinal Image"),
    outputs=gr.Label(num_top_classes=5, label="Severity Classification"),
    title="👁️ Diabetic Retinopathy Detection",
    description="Upload a retinal fundus image to classify diabetic retinopathy severity. Built on EfficientNetB3 trained on the APTOS 2019 dataset.",
    examples=None,
    flagging_mode="never"
)

demo.launch()
