import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Page config
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="centered"
)

# Title
st.title("👁️ Diabetic Retinopathy Detection")
st.write("Upload a retinal image to detect diabetic retinopathy severity")

# Class labels
CLASS_LABELS = {
    0: "No DR",
    1: "Mild DR", 
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

@st.cache_resource
def download_and_load_model():
    """Download weights and load model"""
    try:
        # Download weights if not exists
        if not os.path.exists('dr_model.weights.h5'):
            file_id = "13rrhte8UAxSlOyEj8ae74n0LrrzeYaUJ"
            url = f"https://drive.google.com/uc?id={file_id}"
            st.info("Downloading model weights... This may take a moment.")
            gdown.download(url, 'dr_model.weights.h5', quiet=False)
        
        # Create model with exact architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.applications.EfficientNetB3(
                include_top=False,
                weights=None,
                input_shape=(224, 224, 3)
            ),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        # Load weights
        model.load_weights('dr_model.weights.h5')
        
        return model
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize to 224x224
    image = image.resize((224, 224))
    
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array and normalize
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image(model, image):
    """Make prediction on preprocessed image"""
    try:
        prediction = model.predict(image, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        return predicted_class, confidence, prediction[0]
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

# Load model
model = download_and_load_model()

if model is not None:
    st.success("✅ Model loaded successfully!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a retinal image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a retinal fundus image"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("Prediction Results")
            
            # Preprocess and predict
            with st.spinner("Analyzing image..."):
                processed_image = preprocess_image(image)
                predicted_class, confidence, probabilities = predict_image(model, processed_image)
            
            if predicted_class is not None:
                # Display prediction
                st.markdown(f"**Prediction:** {CLASS_LABELS[predicted_class]}")
                st.markdown(f"**Confidence:** {confidence:.2%}")
                
                # Progress bar for confidence
                st.progress(confidence)
                
                # Show all probabilities
                st.subheader("Class Probabilities")
                for i, prob in enumerate(probabilities):
                    st.write(f"{CLASS_LABELS[i]}: {prob:.3f}")
                    st.progress(prob)
        
        # Additional info
        st.info("""
        **Model Information:**
        - Architecture: EfficientNetB3 + Dense layers
        - Input size: 224×224×3
        - Classes: 5 (No DR, Mild, Moderate, Severe, Proliferative)
        - Parameters: 10.98M trainable
        """)
    
    else:
        st.info("👆 Please upload a retinal image to get started")
        
        # Show example or instructions
        st.markdown("""
        ### Instructions:
        1. Upload a retinal fundus image (PNG, JPG, or JPEG)
        2. The model will analyze the image
        3. Results will show the predicted diabetic retinopathy severity
        
        ### Classes:
        - **No DR**: No diabetic retinopathy
        - **Mild DR**: Mild non-proliferative diabetic retinopathy
        - **Moderate DR**: Moderate non-proliferative diabetic retinopathy  
        - **Severe DR**: Severe non-proliferative diabetic retinopathy
        - **Proliferative DR**: Proliferative diabetic retinopathy
        """)
        
else:
    st.error("❌ Failed to load model. Please refresh the page to try again.")
