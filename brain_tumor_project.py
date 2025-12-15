import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# --- Configuration ---
MODEL_PATH = 'brain_tumor_detector_finetuned.h5'
IMG_SIZE = (224, 224)
CLASSES = ["No Tumor", "Tumor"] # Must match the classes used during training

# --- Model Training/Fine-Tuning Functions (From train_model.py) ---

def build_head_model(baseModel):
    """Builds the new classification head for the VGG-16 base."""
    headModel = baseModel.output
    # Apply Average Pooling after the VGG-16 feature extraction
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    # Final dense layer with softmax for classification
    headModel = Dense(len(CLASSES), activation="softmax")(headModel)
    return Model(inputs=baseModel.input, outputs=headModel) 

def combine_history(H1, H2):
    """Combines metrics from two Keras History objects."""
    combined_history = {}
    for key in H1.history.keys():
        combined_history[key] = H1.history[key] + H2.history[key] 
    return combined_history 

def run_training_and_finetuning():
    """
    Main function to run the two-stage transfer learning process.
    Note: Dummy data is used here. Replace with real data loading.
    """
    print("\n--- Preparing Data and Configuration for Training ---")

    # --- Hyperparameters ---
    INIT_LR = 1e-4  
    FINETUNE_LR = 1e-5 
    EPOCHS_STAGE_1 = 10
    EPOCHS_STAGE_2 = 10
    BS = 32

    # --- Dummy Data Simulation (REPLACE THIS WITH YOUR REAL DATA LOADING) ---
    # Example: Loading real data from disk using ImageDataGenerator.flow_from_directory() 
    # is the standard way to replace these dummy numpy arrays.
    trainX = np.random.rand(100, 224, 224, 3) 
    # Use one-hot encoded labels for binary_crossentropy/softmax output
    trainY = tf.keras.utils.to_categorical(np.random.randint(0, 2, size=(100,)), num_classes=len(CLASSES)) 
    testX = np.random.rand(20, 224, 224, 3) 
    testY = tf.keras.utils.to_categorical(np.random.randint(0, 2, size=(20,)), num_classes=len(CLASSES))
    # --------------------------------------------------------------------


    # 1. Initialize Data Augmentation for Training
    trainAug = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest" 
    )

    # 2. No Augmentation for Validation/Testing (only normalization)
    valAug = ImageDataGenerator() 

    # ## CELL 2: Model Definition and Feature Extraction (Stage 1)
    print("\n--- Starting Stage 1: Feature Extraction (Training Head) ---")

    # 1. Load the VGG-16 base model
    baseModel = VGG16(
        weights="imagenet",
        include_top=False, 
        input_tensor=tf.keras.layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    )

    # 2. Freeze the base model layers
    baseModel.trainable = False 

    # 3. Build and Combine the new head and base models
    model = build_head_model(baseModel) 

    # 5. Compile the model for Stage 1
    opt = Adam(learning_rate=INIT_LR) 
    # Use 'categorical_crossentropy' if labels are one-hot encoded, or 
    # 'binary_crossentropy' as done in the original file for the 2-class problem.
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"]) 

    print(f"Trainable layers (Head only): {len(model.trainable_variables)}") 

    # 6. Train the classification head
    H = model.fit(
        trainAug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=valAug.flow(testX, testY, batch_size=BS),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS_STAGE_1
    )
    print("Stage 1 complete.") 


    # ## CELL 3: Fine-Tuning the Base Layers (Stage 2)
    print("\n--- Starting Stage 2: Fine-Tuning (Unfreezing Top VGG-16 Layers) ---")

    # 1. Unfreeze the entire VGG-16 base model
    baseModel.trainable = True 

    # 2. Freeze the first two blocks and block 3 layers
    for layer in baseModel.layers:
        # Freeze 'block1', 'block2', and 'block3' layers
        if layer.name.startswith("block1") or layer.name.startswith("block2") or layer.name.startswith("block3"): 
            layer.trainable = False 
        else:
            layer.trainable = True 

    # 3. Re-compile the model with an even lower learning rate
    opt = Adam(learning_rate=FINETUNE_LR) 
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"]) 

    print(f"Trainable layers (Head + Top VGG Blocks): {len(model.trainable_variables)}") 

    # 4. Continue training for a few more epochs
    H_finetune = model.fit(
        trainAug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=valAug.flow(testX, testY, batch_size=BS),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS_STAGE_1 + EPOCHS_STAGE_2,
        initial_epoch=H.epoch[-1] + 1
    )
    print("Stage 2 complete. Model is fully fine-tuned.") 


    # ## CELL 4: Plotting and Reporting Final Metrics
    combined_H = combine_history(H, H_finetune) 

    # 1. Extract Metrics
    final_train_loss = combined_H["loss"][-1]
    final_train_acc = combined_H["accuracy"][-1]
    final_val_loss = combined_H["val_loss"][-1]
    final_val_acc = combined_H["val_accuracy"][-1]

    print("\n--- FINAL MODEL METRICS ---")
    print(f"Training Loss: {final_train_loss:.4f}")
    print(f"Training Accuracy: {final_train_acc:.4f}")
    print(f"Validation Loss: {final_val_loss:.4f}")
    print(f"Validation Accuracy: {final_val_acc:.4f}")

    # Optional: Save the final fine-tuned model
    model.save(MODEL_PATH)
    print(f"\nFinal fine-tuned model saved as '{MODEL_PATH}'")


# --- Streamlit Application Functions (From app.py) ---

@st.cache_resource
def load_model(path):
    """Loads the fine-tuned Keras model."""
    try:
        # Load the model using the recommended Keras loading function
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        # Check for model file existence
        if "No such file or directory" in str(e):
            st.error(f"Error: Model file '{path}' not found. Please run the training script first.")
        else:
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

# --- Main Execution Block ---

# Check if the file is being run directly from the command line for training
# This is a good pattern for combining the app and the training script
if __name__ == "__main__":
    
    # A simple command-line argument check to decide whether to run training or the Streamlit app
    # To run training, execute: python vscode_and_streamlit_solution.py train
    # To run the Streamlit app, execute: streamlit run vscode_and_streamlit_solution.py
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        print("Running model training and saving...")
        run_training_and_finetuning()
        sys.exit(0) # Exit after training is done
    
    # --- Streamlit Application Code ---
    st.set_page_config(
        page_title="🧠 Brain Tumor Detector",
        layout="centered",
        initial_sidebar_state="auto"
    )

    st.title("🧠 Brain Tumor Detection with Fine-Tuned VGG-16")
    st.markdown("Upload an MRI image to get a prediction from the trained model.")
    st.markdown("""
        **How to use:**
        1. **Ensure the model is trained:** The script must be run once with the `train` argument 
           (e.g., `python vscode_and_streamlit_solution.py train`) to generate the 
           `brain_tumor_detector_finetuned.h5` file.
        2. **Run the Streamlit app:** Use `streamlit run vscode_and_streamlit_solution.py`.
        3. Upload an MRI image below.
    """)

    # 1. Load the Model
    model = load_model(MODEL_PATH)

    if model is None:
        st.warning(f"The model file `{MODEL_PATH}` could not be loaded. Please run the training command first.")
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
                    # np.argmax for the index of the highest probability
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
                        
                    # Create data for the bar chart
                    chart_data = {
                        CLASSES[0]: predictions[0][0],
                        CLASSES[1]: predictions[0][1]
                    }
                    st.bar_chart(chart_data)

                    st.info("Disclaimer: This is a machine learning model prediction and should not be used as a substitute for professional medical advice.")
