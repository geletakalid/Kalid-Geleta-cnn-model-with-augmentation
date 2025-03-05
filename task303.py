import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import random

# Define dataset path
dataset_dir = "C:/Users/kalid/Desktop/pythonProject34/dataset/train"

# Image size and batch size
img_size = (224, 224)
batch_size = 32

# Data generator WITHOUT augmentation (for original images & validation)
original_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

# Data generator WITH augmentation (for training only)
augmented_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Train generator (original images only)
original_train_generator = original_datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Train generator (with augmentation)
augmented_train_generator = augmented_datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Validation generator (NO augmentation)
val_generator = original_datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# ------ IMAGE AUGMENTATION PREVIEW --------
x_batch, y_batch = next(original_train_generator)

random_indices = random.sample(range(len(x_batch)), 5)
selected_images = np.array([x_batch[i] for i in random_indices])

augmented_images = np.array([augmented_datagen.random_transform(img) for img in selected_images])

selected_images_uint8 = (selected_images * 255).astype(np.uint8)
augmented_images_uint8 = (augmented_images * 255).astype(np.uint8)

def plotter(selected_images_uint8, augmented_images_uint8):
    plt.figure(figsize=(10, 5))

    for i in range(5):
        # Original image
        plt.subplot(2, 5, i + 1)
        plt.imshow(selected_images_uint8[i])
        plt.title("Original")
        plt.axis("off")

        # Augmented image
        plt.subplot(2, 5, i + 6)
        plt.imshow(augmented_images_uint8[i])
        plt.title("Augmented")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Display augmented images
plotter(selected_images_uint8, augmented_images_uint8)

# ------ TRAIN RESNET50 MODEL --------

# Load the ResNet50 model (without top layers)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
output_layer = Dense(original_train_generator.num_classes, activation='softmax')(x)

# Create final model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the augmented dataset
history = model.fit(
    augmented_train_generator,
    validation_data=val_generator,
    epochs=10,  # Adjust based on dataset size
    steps_per_epoch=len(augmented_train_generator),
    validation_steps=len(val_generator),
    verbose=1
)

# Save the trained model
model.save("resnet50_brain_cancer_model.h5")

# ------ PLOT TRAINING RESULTS --------

plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()

plt.show()

