import streamlit as st
import tensorflow as tf
import os
import sys
import numpy as np

st.set_page_config(page_title="Model Loading Test", layout="wide")

st.title("🔍 Diabetic Retinopathy Model Loading Test")

# System info
st.header("System Information")
col1, col2 = st.columns(2)

with col1:
    st.write(f"**TensorFlow Version:** {tf.__version__}")
    st.write(f"**Python Version:** {sys.version}")

with col2:
    st.write(f"**Current Directory:** {os.getcwd()}")
    st.write(f"**Available Files:** {os.listdir('.')}")

# Check for model files
st.header("Model File Detection")
keras_file = "diabetic_retinopathy_model.keras"
h5_file = "diabetic_retinopathy_model.h5"

keras_exists = os.path.exists(keras_file)
h5_exists = os.path.exists(h5_file)

col1, col2 = st.columns(2)

with col1:
    if keras_exists:
        st.success(f"✅ {keras_file} found")
        file_size = os.path.getsize(keras_file) / (1024*1024)  # MB
        st.write(f"Size: {file_size:.1f} MB")
    else:
        st.error(f"❌ {keras_file} not found")

with col2:
    if h5_exists:
        st.success(f"✅ {h5_file} found")
        file_size = os.path.getsize(h5_file) / (1024*1024)  # MB
        st.write(f"Size: {file_size:.1f} MB")
    else:
        st.error(f"❌ {h5_file} not found")

# Model loading tests
st.header("Model Loading Tests")

if not keras_exists and not h5_exists:
    st.error("No model files found! Please upload your model file to the repository.")
    st.stop()

# Choose which model to test
model_to_test = None
if keras_exists and h5_exists:
    model_to_test = st.selectbox("Select model to test:", [keras_file, h5_file])
elif keras_exists:
    model_to_test = keras_file
    st.info(f"Testing {keras_file}")
else:
    model_to_test = h5_file
    st.info(f"Testing {h5_file}")

if st.button("Run Model Loading Tests"):
    
    # Test 1: Direct loading
    st.subheader("Test 1: Direct Loading")
    try:
        with st.spinner("Loading model directly..."):
            model = tf.keras.models.load_model(model_to_test)
        
        st.success("✅ Direct loading successful!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Input Shape:** {model.input_shape}")
            st.write(f"**Output Shape:** {model.output_shape}")
        with col2:
            st.write(f"**Number of Layers:** {len(model.layers)}")
            st.write(f"**Parameters:** {model.count_params():,}")
        
        # Test prediction
        st.subheader("Test Prediction")
        try:
            test_input = np.random.random((1, 224, 224, 3))
            prediction = model.predict(test_input, verbose=0)
            
            st.success("✅ Test prediction successful!")
            st.write(f"**Prediction Shape:** {prediction.shape}")
            
            # Show prediction values
            class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
            pred_values = prediction[0]
            
            for i, (name, value) in enumerate(zip(class_names, pred_values)):
                st.write(f"**{name}:** {value:.4f}")
            
            st.success("🎉 Model is working perfectly!")
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
        
        del model  # Free memory
        
    except Exception as e:
        st.error(f"❌ Direct loading failed: {str(e)}")
        
        # Test 2: Loading without compilation
        st.subheader("Test 2: Loading Without Compilation")
        try:
            with st.spinner("Loading without compilation..."):
                model = tf.keras.models.load_model(model_to_test, compile=False)
            
            st.success("✅ Loading without compilation successful!")
            
            try:
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                st.success("✅ Recompilation successful!")
                
                # Test prediction
                test_input = np.random.random((1, 224, 224, 3))
                prediction = model.predict(test_input, verbose=0)
                st.success("✅ Test prediction successful!")
                
                st.success("🎉 Model works with recompilation!")
                
            except Exception as e2:
                st.error(f"❌ Recompilation failed: {str(e2)}")
            
            del model
            
        except Exception as e:
            st.error(f"❌ Loading without compilation failed: {str(e)}")
            
            # Show detailed error info
            st.subheader("Error Details")
            st.code(str(e))
            
            # Recommendations
            st.subheader("Recommendations")
            st.write("Try these solutions:")
            st.write("1. Update TensorFlow version in requirements.txt")
            st.write("2. Re-save model with compatible TensorFlow version")
            st.write("3. Convert model to SavedModel format")

# Additional diagnostics
st.header("Additional Diagnostics")

if st.button("Show TensorFlow Configuration"):
    st.subheader("TensorFlow Build Info")
    st.code(tf.config.list_physical_devices())
    
    st.subheader("Available Operations")
    # Check if specific operations are available
    ops_to_check = ['Conv2D', 'Dense', 'GlobalAveragePooling2D', 'Dropout']
    for op in ops_to_check:
        try:
            tf.raw_ops.__dict__[op]
            st.write(f"✅ {op} available")
        except:
            st.write(f"❌ {op} not available")

# Show model architecture if possible
if st.button("Inspect Model Architecture (if loadable)"):
    try:
        model = tf.keras.models.load_model(model_to_test, compile=False)
        
        st.subheader("Model Summary")
        # Capture model summary
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            model.summary()
        model_summary = f.getvalue()
        
        st.code(model_summary)
        
        st.subheader("Layer Details")
        for i, layer in enumerate(model.layers):
            st.write(f"**Layer {i}:** {layer.__class__.__name__}")
            st.write(f"  - Config: {layer.get_config()}")
            
    except Exception as e:
        st.error(f"Could not inspect model: {str(e)}")
