# Task_2_deeplearning
CIFAR-10 Image Classification (CODTECH Internship Task 2)
Project Overview

This project is part of the CODTECH Internship Task 2.
We implemented a Deep Learning model to classify images from the CIFAR-10 dataset using a Convolutional Neural Network (CNN) in TensorFlow/Keras.

The goal was to build a functional model and produce visualizations of the results, including accuracy, loss curves, and sample predictions.
CIFAR-10 Image Classification (CODTECH Internship Task 2)
Dataset

CIFAR-10: 60,000 32x32 color images in 10 classes:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Training set: 50,000 images

Test set: 10,000 images

Loaded directly from TensorFlow/Keras datasets.

Model Architecture

Input: 32x32 RGB images

Layers:

Conv2D → 32 filters, 3x3 kernel, ReLU activation

MaxPooling2D → 2x2

Conv2D → 64 filters, 3x3 kernel, ReLU activation

MaxPooling2D → 2x2

Conv2D → 64 filters, 3x3 kernel, ReLU activation

Flatten → Dense(64, ReLU) → Dense(10, Softmax)

Optimizer: Adam

Loss: Sparse Categorical Crossentropy

Metrics: Accuracy
Results

Training Accuracy & Validation Accuracy: accuracy.png

Training Loss & Validation Loss: loss.png

Sample Predictions: sample_predictions.png

Classification Report: classification_report.txt

Confusion Matrix: confusion_matrix.npy (can be plotted)
Results

Training Accuracy & Validation Accuracy: accuracy.png

Training Loss & Validation Loss: loss.png

Sample Predictions: sample_predictions.png

Classification Report: classification_report.txt

Confusion Matrix: confusion_matrix.npy (can be plotted)

Conclusion

This project demonstrates a working deep learning pipeline for image classification.
It includes training, evaluation, visualizations, and a saved model ready for inference.
