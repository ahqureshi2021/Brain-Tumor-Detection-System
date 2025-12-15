import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


trainX = np.random.rand(100, 224, 224, 3) # Replace with your training images
trainY = np.random.randint(0, 2, size=(100, 2)) # Replace with your training labels (one-hot encoded)
testX = np.random.rand(20, 224, 224, 3) # Replace with your test images
testY = np.random.randint(0, 2, size=(20, 2)) # Replace with your test labels (one-hot encoded)
CLASSES = ["No Tumor", "Tumor"] # Replace with your actual class names
# --------------------------------------------------------------------


# ## CELL 1: Imports and Data Augmentation Setup
# --- Hyperparameters ---
IMG_SIZE = (224, 224)
INIT_LR = 1e-4  
FINETUNE_LR = 1e-5 
EPOCHS_STAGE_1 = 10
EPOCHS_STAGE_2 = 10
BS = 32

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


def build_head_model(baseModel):
    """Builds the new classification head for the VGG-16 base."""
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(len(CLASSES), activation="softmax")(headModel)
    return Model(inputs=baseModel.input, outputs=headModel) 

def combine_history(H1, H2):
    """Combines metrics from two Keras History objects."""
    combined_history = {}
    for key in H1.history.keys():
        combined_history[key] = H1.history[key] + H2.history[key] 
    return combined_history 

# --- Main Training Function ---
if __name__ == "__main__":
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
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"]) 

    print(f"Total layers in model: {len(model.layers)}") 
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
        if layer.name.startswith("block1") or layer.name.startswith("block2") or layer.name.startswith("block3"): # [cite: 4]
            layer.trainable = False 
        else:
            layer.trainable = True 

    # 3. Re-compile the model with an even lower learning rate
    opt = Adam(learning_rate=FINETUNE_LR) 
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"]) 

    print(f"Total layers in model: {len(model.layers)}") 
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
    model_save_path = 'brain_tumor_detector_finetuned.h5'
    model.save(model_save_path)
    print(f"\nFinal fine-tuned model saved as '{model_save_path}'")
