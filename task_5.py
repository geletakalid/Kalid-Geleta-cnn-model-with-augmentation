import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define the image directory (use any image from your dataset)
image_dir = r"C:\Users\kalid\Desktop\pythonProject34\PetImages\Cat"  # Change to 'Dog' if needed

# Load a single image from the directory
image_path = os.path.join(image_dir, os.listdir(image_dir)[0])  # Pick the first image
img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))  # Resize image
img_array = tf.keras.preprocessing.image.img_to_array(img)  # Convert to array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Different Augmentation Techniques
augmentations = {
    "Original": ImageDataGenerator(rescale=1./255),  # No augmentation, just rescaling
    "Rotation": ImageDataGenerator(rotation_range=40),
    "Zoom": ImageDataGenerator(zoom_range=0.5),
    "Horizontal Flip": ImageDataGenerator(horizontal_flip=True),
    "Shear": ImageDataGenerator(shear_range=0.2),
    "Width Shift": ImageDataGenerator(width_shift_range=0.2),
    "Height Shift": ImageDataGenerator(height_shift_range=0.2)
}

# Plot the results
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()

for ax, (key, datagen) in zip(axes, augmentations.items()):
    augmented_img = next(datagen.flow(img_array, batch_size=1))  # Apply transformation
    ax.imshow(augmented_img[0] / 255.0)  # Normalize to [0,1] for display
    ax.set_title(key)
    ax.axis("off")

plt.tight_layout()
plt.show()