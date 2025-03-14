# Flower Classification using Deep Learning

## Project Overview
This project implements a **deep learning model** to classify flower images into **five different categories**. The model is trained on a dataset containing flower images and utilizes a **convolutional neural network (CNN)** for classification.

## Dataset
The dataset consists of:
- **Unaugmented Data**: Raw images without any modifications.
- **Augmented Data**: Processed images with transformations such as **rotation, flipping, and scaling** to enhance model generalization.

## Prerequisites
Ensure you have the following dependencies installed before running the project:

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python
```

## Model Architecture
The classification model is built using a **CNN (Convolutional Neural Network)** with the following layers:
- **Convolutional layers** with ReLU activation
- **Max pooling layers** for dimensionality reduction
- **Fully connected dense layers**
- **Softmax activation** for multi-class classification

## Training Process
1. Load the dataset (**augmented and unaugmented images**).
2. Preprocess images (**resize, normalize, augment**).
3. Split data into **training and testing sets**.
4. Train the **CNN model**.
5. Evaluate performance using **accuracy and loss metrics**.

## Code Explanation
### `train.py` (Training the Model)
#### Step 1: Import Libraries
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
```
**Explanation:**
- TensorFlow and Keras are used for deep learning.
- `Sequential` is used to create a linear stack of layers.
- `Conv2D` and `MaxPooling2D` are used for feature extraction.
- `Flatten` and `Dense` layers create the final classification structure.
- `ImageDataGenerator` helps in data augmentation.

#### Step 2: Load and Preprocess the Dataset
```python
data_dir = 'dataset/'
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training')
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation')
```
**Explanation:**
- Images are rescaled to values between 0 and 1.
- Data is split into training (80%) and validation (20%).
- `flow_from_directory()` loads images from subdirectories.

#### Step 3: Define the CNN Model
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])
```
**Explanation:**
- Three convolutional layers extract features from images.
- `MaxPooling2D` reduces dimensionality and prevents overfitting.
- `Flatten` converts the matrix into a one-dimensional vector.
- Fully connected `Dense` layers classify the images.
- `Dropout` prevents overfitting by randomly deactivating neurons.
- `Softmax` outputs probabilities for five flower categories.

#### Step 4: Compile and Train the Model
```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=validation_generator, epochs=10)
model.save('flower_classification_model.h5')
```
**Explanation:**
- The model is compiled using the Adam optimizer and categorical cross-entropy loss.
- The model is trained for 10 epochs.
- The trained model is saved for future use.

### `classify.py` (Classifying Images)
#### Step 1: Import Libraries
```python
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
```
**Explanation:**
- TensorFlow is used to handle deep learning tasks.
- OpenCV (`cv2`) is used for image processing.
- The trained model is loaded with `load_model`.

#### Step 2: Load the Trained Model
```python
model = load_model('flower_classification_model.h5')
```
**Explanation:**
- The saved model is loaded to classify new images.

#### Step 3: Load and Preprocess the Image
```python
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (150, 150))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image
```
**Explanation:**
- The image is read using OpenCV.
- The image is resized to match the model input size (150x150 pixels).
- The pixel values are normalized.
- `expand_dims` is used to add an extra dimension for batch processing.

#### Step 4: Make a Prediction
```python
image_path = 'test_flower.jpg'
processed_image = preprocess_image(image_path)
predictions = model.predict(processed_image)
class_index = np.argmax(predictions)
class_labels = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']
predicted_class = class_labels[class_index]
print(f'Predicted Flower: {predicted_class}')
```
**Explanation:**
- The model predicts the class probabilities.
- `argmax` finds the class with the highest probability.
- The corresponding flower name is retrieved and displayed.

## Usage
Run the following command to **start training**:

```bash
python train.py
```

To **classify a new image**, use:
```bash
python classify.py --image path_to_image.jpg
```

## Evaluation Metrics
- **Accuracy**
- **Precision, Recall, and F1-score**
- **Confusion Matrix**

## Results
The trained model achieves **high accuracy** in classifying flower images into **five categories**. Results can be visualized using:
- **Confusion Matrix**
- **Accuracy/Loss plots**

## Future Enhancements
- Implement **transfer learning** using pre-trained models like **VGG16, ResNet**
- Optimize **hyperparameters** for better performance
- Deploy as a **web application** using Flask or FastAPI

## Author
**Geleta Kalid**

## License
This project is **open-source** and available under the **MIT License**.

