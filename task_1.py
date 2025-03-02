import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Define paths
train_dir = 'path/to/train/dataset'
validation_dir = 'path/to/validation/dataset'


# Display a few images
def plot_images(images, titles):
    plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')
    plt.show()


# Load a few images
sample_images = [os.path.join(train_dir, 'cat.1.jpg'), os.path.join(train_dir, 'dog.1.jpg')]
images = [plt.imread(img) for img in sample_images]
plot_images(images, ['Cat', 'Dog'])
