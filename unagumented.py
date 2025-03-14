import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, MaxPool2D, GlobalAveragePooling2D ,Dense, Input,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import random

# Define dataset path
dataset_dir = "C:/Users/kalid/Desktop/pythonProject34/dataset/train"

# Image size and batch size
img_size = (224, 224)
batch_size = 16

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

