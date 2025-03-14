Flower Classification using Deep Learning

Project Overview

This project implements a deep learning model to classify flower images into five different categories. The model is trained on a dataset containing flower images and uses a convolutional neural network (CNN) for image classification.

Dataset

The dataset consists of:

Unaugmented Data: Raw images without any modifications.

Augmented Data: Processed images with transformations such as rotation, flipping, and scaling to enhance model generalization.

Prerequisites

To run this project, ensure you have the following dependencies installed:

pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python

Model Architecture

The classification model is built using a CNN with the following layers:

Convolutional layers with ReLU activation

Max pooling layers

Fully connected dense layers

Softmax activation for classification

Training

Load the dataset (augmented and unaugmented images).

Preprocess images (resize, normalize, augment).

Split data into training and testing sets.

Train the CNN model.

Evaluate performance using accuracy and loss metrics.

Usage

Run the following command to start training:

python train.py

To classify a new image:

python classify.py --image path_to_image.jpg

Evaluation Metrics

Accuracy

Precision, Recall, and F1-score

Confusion Matrix

Results

The trained model achieves high accuracy in classifying flower images into five categories. Results can be visualized using a confusion matrix and accuracy/loss plots.

Future Enhancements

Implementing transfer learning using pre-trained models like VGG16, ResNet.

Optimizing hyperparameters for better performance.

Deploying as a web application using Flask or FastAPI.

Authors

Geleta Kalid

License

This project is open-source and available under the MIT License.

