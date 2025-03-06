import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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

# Display augmented images
#plotter(selected_images_uint8, augmented_images_uint8)










# ------ TRAIN RESNET50 MODEL --------

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model to train only new classification layers
base_model.trainable = False

# Add new classification head
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(original_train_generator.num_classes, activation='softmax')(x)

# Build model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

# Train only the new classification head first
history = model.fit(
    original_train_generator,
    validation_data=val_generator,
    epochs=10,
    steps_per_epoch=len(original_train_generator),
    validation_steps=len(val_generator),
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

# Unfreeze some layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:140]:  # Freeze first 140 layers
    layer.trainable = False

# Recompile with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model
history_fine = model.fit(
    original_train_generator,
    validation_data=val_generator,
    epochs=10,
    steps_per_epoch=len(original_train_generator),
    validation_steps=len(val_generator),
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

# Save final model
model.save("unaugmented.h5")

# ------ PLOT TRAINING RESULTS --------

plt.figure(figsize=(12, 5))

# Accuracy plot
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