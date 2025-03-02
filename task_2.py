import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
train_dir = r"C:\Users\kalid\Desktop\pythonProject34\PetImages_Split\train"
validation_dir = r"C:\Users\kalid\Desktop\pythonProject34\PetImages_Split\validation"

# Code for Task 3: Visualize Augmented Images
def plot_images(images, titles):
    plt.figure(figsize=(10, 5))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis("off")
    plt.show()

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize pixel values (0-1 range)
    rotation_range=40,        # Random rotation
    width_shift_range=0.2,    # Random horizontal shift
    height_shift_range=0.2,   # Random vertical shift
    shear_range=0.2,          # Shear transformation
    zoom_range=0.2,           # Random zoom
    horizontal_flip=True,     # Flip images horizontally
    fill_mode='nearest'       # Fill pixels when shifting
)

# Validation data should NOT be augmented (only rescaled)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Create training data generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Resize images to 150x150 pixels
    batch_size=32,           # Load 32 images at a time
    class_mode='binary'      # Binary classification (Dog vs. Cat)
)

# Create validation data generator
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Print class indices to verify label mapping
print("Class Indices:", train_generator.class_indices)

# Get a batch of augmented images
augmented_images = [train_generator[0][0][i] for i in range(4)]
plot_images(augmented_images, ['Augmented Image'] * 4)