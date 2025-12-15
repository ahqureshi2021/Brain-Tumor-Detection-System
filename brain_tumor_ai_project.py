import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import os
import io

# --- Configuration & Hyperparameters (from CELL 1) ---
MODEL_PATH = 'brain_tumor_detector_finetuned.h5'
IMG_SIZE = (224, 224)
INIT_LR = 1e-4
FINETUNE_LR = 1e-5
EPOCHS_STAGE_1 = 5 # Reduced for quicker demo training
EPOCHS_STAGE_2 = 5 # Reduced for quicker demo training
BS = 32

# --- Placeholder Data & Classes (MUST BE REPLACED FOR REAL USE) ---
# NOTE: In a real application, you would load these from files (e.g., NumPy or image datasets).
# For this merged demo, we use small random placeholder data.
TRAIN_SAMPLES = 100
TEST_SAMPLES = 20
CLASSES = ["No Tumor", "Tumor"]
trainX = np.random.rand(TRAIN_SAMPLES, 224, 224, 3).astype('float32') 
trainY = tf.keras.utils.to_categorical(np.random.randint(0, 2, size=(TRAIN_SAMPLES,)), num_classes=len(CLASSES))
testX = np.random.rand(TEST_SAMPLES, 224, 224, 3).astype('float32') 
testY = tf.keras.utils.to_categorical(np.random.randint(0, 2, size=(TEST_SAMPLES,)), num_classes=len(CLASSES))


# --- Helper Functions (from original script) ---

def build_head_model(baseModel):
    """Builds the new classification head for the VGG-16 base."""
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(len(CLASSES), activation="softmax")(headModel) #
    return Model(inputs=baseModel.input, outputs=headModel) 

def combine_history(H1, H2):
    """Combines metrics from two Keras History objects."""
    combined_history = {}
    for key in H1.history.keys():
        combined_history[key] = H1.history[key] + H2.history[key] 
    return combined_history 

def preprocess_image(image):
    """Resizes and preprocesses the image for the model."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# --- Training Function (Combined Cell 1, 2, 3, 4) ---

def train_and_save_model():
    st.info("Starting VGG-16 Fine-Tuning...")
    
    # Data Augmentation Setup [cite: 1, 2]
    trainAug = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest" [cite: 2]
    )
    valAug = ImageDataGenerator() # No augmentation for validation [cite: 2]

    # --- Stage 1: Feature Extraction (Training Head) [cite: 2] ---
    with st.spinner("Stage 1/2: Training Classification Head..."):
        # Load VGG-16 Base Model [cite: 2]
        baseModel = VGG16(
            weights="imagenet",
            include_top=False, 
            input_tensor=tf.keras.layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        )
        baseModel.trainable = False # Freeze base layers [cite: 2]
        
        # Build and Compile Model
        model = build_head_model(baseModel) [cite: 3]
        opt = Adam(learning_rate=INIT_LR)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"]) [cite: 2]

        # Train Stage 1 [cite: 2]
        H = model.fit(
            trainAug.flow(trainX, trainY, batch_size=BS),
            steps_per_epoch=len(trainX) // BS,
            validation_data=valAug.flow(testX, testY, batch_size=BS),
            validation_steps=len(testX) // BS,
            epochs=EPOCHS_STAGE_1
        )
        st.success(f"Stage 1 Complete. Accuracy: {H.history['val_accuracy'][-1]:.4f}")

    # --- Stage 2: Fine-Tuning Top Layers [cite: 4] ---
    with st.spinner("Stage 2/2: Fine-Tuning Top VGG Blocks..."):
        baseModel.trainable = True # Unfreeze the entire VGG-16 base model [cite: 4]

        # Freeze Block 1, 2, and 3 [cite: 4]
        for layer in baseModel.layers:
            if layer.name.startswith("block1") or layer.name.startswith("block2") or layer.name.startswith("block3"):
                layer.trainable = False
            else:
                layer.trainable = True

        # Re-compile with lower LR [cite: 5]
        opt = Adam(learning_rate=FINETUNE_LR) [cite: 5]
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"]) [cite: 5]

        # Train Stage 2 [cite: 5]
        H_finetune = model.fit(
            trainAug.flow(trainX, trainY, batch_size=BS),
            steps_per_epoch=len(trainX) // BS,
            validation_data=valAug.flow(testX, testY, batch_size=BS),
            validation_steps=len(testX) // BS,
            epochs=EPOCHS_STAGE_1 + EPOCHS_STAGE_2,
            initial_epoch=H.epoch[-1] + 1
        )
        st.success(f"Stage 2 Complete. Final Validation Accuracy: {H_finetune.history['val_accuracy'][-1]:.4f}")

    # --- Save Model [cite: 6] ---
    model.save(MODEL_PATH) 
    st.balloons()
    st.success(f"Training Complete! Model saved as '{MODEL_PATH}'")


# --- Model Loading and Caching ---
@st.cache_resource
def load_cached_model(path):
    """Loads the fine-tuned Keras model, caching it to avoid re-loading on refresh."""
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        return None

# --- Streamlit Application ---

st.set_page_config(
    page_title="🧠 Streamlit Fine-Tuning Demo",
    layout="wide"
)

st.title("🧠 VGG-16 Brain Tumor Detector")
st.markdown("This single-file app runs the two-stage fine-tuning process on synthetic data and then uses the trained model for image prediction.")
st.caption(f"**Note**: Training uses small placeholder data and the model will be trained and saved as `{MODEL_PATH}` on first run.")

# --- Training Section ---
st.header("1. Model Training")
if os.path.exists(MODEL_PATH):
    st.success(f"Model `{MODEL_PATH}` found. Ready for prediction.")
else:
    st.warning("Model not found. Click the button below to train and save the model (This may take a few minutes).")
    
if st.button("Start 2-Stage Model Training"):
    train_and_save_model()


# --- Prediction Section ---
st.header("2. Image Prediction")
model = load_cached_model(MODEL_PATH)

if model is None:
    st.error("Model is not loaded. Please train the model in step 1.")
else:
    uploaded_file = st.file_uploader(
        "Upload an MRI image for prediction:",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

        with col2:
            st.subheader("Analysis")
            if st.button("Predict"):
                with st.spinner("Analyzing image..."):
                    # Preprocess and Predict
                    processed_image = preprocess_image(image)
                    predictions = model.predict(processed_image)
                    
                    # Get the result
                    predicted_class_index = np.argmax(predictions, axis=1)[0]
                    confidence = predictions[0][predicted_class_index]
                    predicted_label = CLASSES[predicted_class_index]

                    # Display Results
                    st.markdown("### Prediction:")
                    if predicted_label == "Tumor":
                        st.error(f"**Result:** {predicted_label}")
                        st.markdown(f"**Confidence:** {confidence:.4f}")
                    else:
                        st.success(f"**Result:** {predicted_label}")
                        st.markdown(f"**Confidence:** {confidence:.4f}")
                        
                    st.markdown("### Class Probabilities:")
                    st.bar_chart({
                        'No Tumor': predictions[0][0],
                        'Tumor': predictions[0][1]
                    })
