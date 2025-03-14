import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, MaxPool2D, GlobalAveragePooling2D ,Dense, Input,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
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
    horizontal_flip=True,
validation_split=0.2
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



# ------ PLOT TRAINING RESULTS --------

plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history_fine.history['accuracy'], label='Fine-Tune Train Accuracy')
plt.plot(history_fine.history['val_accuracy'], label='Fine-Tune Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training & Fine-Tuning Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history_fine.history['loss'], label='Fine-Tune Train Loss')
plt.plot(history_fine.history['val_loss'], label='Fine-Tune Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Fine-Tuning Loss')
plt.legend()

plt.show()


# Load the trained model
model = tf.keras.models.load_model("unagumented_model.h5")

# Define test dataset path
test_dir = "C:/Users/kalid/Desktop/pythonProject34/dataset/test"
img_size = (224, 224)  # Image input size

# Get class labels from training generator
class_labels = list(model.class_names)  # If using an ImageDataGenerator, otherwise manually set class labels

# Randomly pick 5 images from the test directory
random_images = []
random_filenames = []

for class_name in os.listdir(test_dir):
    class_path = os.path.join(test_dir, class_name)
    if os.path.isdir(class_path):
        image_files = os.listdir(class_path)
        if len(image_files) > 0:
            selected_image = random.choice(image_files)
            random_images.append(os.path.join(class_path, selected_image))
            random_filenames.append(selected_image)

# Predict and display results
plt.figure(figsize=(10, 5))

for i, img_path in enumerate(random_images):
    # Load and preprocess the image
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims for batch

    # Predict
    predictions = model.predict(img_array)
    predicted_label = class_labels[np.argmax(predictions)]  # Get highest probability class

    # Display the image
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.title(predicted_label)
    plt.axis("off")

plt.tight_layout()
plt.show()
