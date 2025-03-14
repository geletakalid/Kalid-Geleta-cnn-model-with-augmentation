# 🌸 Flower Classification using Deep Learning

## 📌 Project Overview
This project implements a **deep learning model** to classify flower images into **five different categories**. The model is trained on a dataset containing flower images and utilizes a **convolutional neural network (CNN)** for classification.

## 📂 Dataset
The dataset consists of:
- **Unaugmented Data**: Raw images without any modifications.
- **Augmented Data**: Processed images with transformations such as **rotation, flipping, and scaling** to enhance model generalization.

## ⚙️ Prerequisites
Ensure you have the following dependencies installed before running the project:

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python
```

## 🏗️ Model Architecture
The classification model is built using a **CNN (Convolutional Neural Network)** with the following layers:
✅ **Convolutional layers** with ReLU activation
✅ **Max pooling layers** for dimensionality reduction
✅ **Fully connected dense layers**
✅ **Softmax activation** for multi-class classification

## 🎯 Training Process
1️⃣ Load the dataset (**augmented and unaugmented images**).
2️⃣ Preprocess images (**resize, normalize, augment**).
3️⃣ Split data into **training and testing sets**.
4️⃣ Train the **CNN model**.
5️⃣ Evaluate performance using **accuracy and loss metrics**.

## 📜 Code Explanation
### 🔹 `train.py`
This script is responsible for training the deep learning model.
- Loads the dataset.
- Applies image preprocessing (resizing, normalization, and augmentation).
- Defines the CNN architecture.
- Trains the model using training data.
- Evaluates performance using validation data.
- Saves the trained model for later use.

### 🔹 `classify.py`
This script is used to classify new flower images.
- Loads the trained model.
- Accepts an image file as input.
- Preprocesses the image to match the model input format.
- Predicts the flower category.
- Displays the classification result.

## 🚀 Usage
Run the following command to **start training**:

```bash
python train.py
```

To **classify a new image**, use:
```bash
python classify.py --image path_to_image.jpg
```

## 📊 Evaluation Metrics
- ✅ **Accuracy**
- ✅ **Precision, Recall, and F1-score**
- ✅ **Confusion Matrix**

## 📈 Results
The trained model achieves **high accuracy** in classifying flower images into **five categories**. Results can be visualized using:
- 📊 **Confusion Matrix**
- 📉 **Accuracy/Loss plots**

## 🔮 Future Enhancements
🔹 Implement **transfer learning** using pre-trained models like **VGG16, ResNet**
🔹 Optimize **hyperparameters** for better performance
🔹 Deploy as a **web application** using Flask or FastAPI

## 👨‍💻 Author
📌 **Geleta Kalid**

## 📜 License
This project is **open-source** and available under the **MIT License**.

