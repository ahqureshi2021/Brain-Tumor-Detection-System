import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# --- Configuration ---
MODEL_PATH = 'brain_tumor_detector_finetuned.h5'
IMG_SIZE = (224, 224)
CLASSES = ["No Tumor", "Tumor"] # Must match the classes used during training

# --- Functions ---

@st.cache_resource
def load_model(path):
    """Loads the fine-tuned Keras model."""
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """Resizes and preprocesses the image for the model."""
    # Convert to RGB (in case the uploaded image is RGBA or grayscale)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize to the required input size
    image = image.resize(IMG_SIZE)

    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    
    # Add batch dimension (1, 224, 224, 3)
    return np.expand_dims(image_array, axis=0)

# --- Streamlit Application ---

st.set_page_config(
    page_title="🧠 Brain Tumor Detector",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🧠 Brain Tumor Detection with Fine-Tuned VGG-16")
st.markdown("Upload an MRI image to get a prediction from the trained model.")

# 1. Load the Model
model = load_model(MODEL_PATH)

if model is None:
    st.warning(f"Please run `train_model.py` first to create and save the model file: `{MODEL_PATH}`")
else:
    # 2. File Uploader
    uploaded_file = st.file_uploader(
        "Choose an MRI image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")

        # 3. Prediction Button
        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                # Preprocess
                processed_image = preprocess_image(image)

                # Predict
                predictions = model.predict(processed_image)
                
                # Get the result
                predicted_class_index = np.argmax(predictions, axis=1)[0]
                confidence = predictions[0][predicted_class_index]
                predicted_label = CLASSES[predicted_class_index]

                # 4. Display Results
                st.subheader("Prediction Result")
                
                if predicted_label == "Tumor":
                    st.error(f"**Result:** {predicted_label}")
                    st.markdown(f"**Confidence:** {confidence:.2f} (High probability of a tumor)")
                else:
                    st.success(f"**Result:** {predicted_label}")
                    st.markdown(f"**Confidence:** {confidence:.2f} (High probability of no tumor)")
                    
                st.bar_chart({
                    'No Tumor': predictions[0][0],
                    'Tumor': predictions[0][1]
                })

                st.info("Disclaimer: This is a machine learning model prediction and should not be used as a substitute for professional medical advice.")
                