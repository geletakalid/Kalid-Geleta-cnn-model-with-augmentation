










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
