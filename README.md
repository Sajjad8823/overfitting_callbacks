# MNIST Digit Classifier with Keras

This project builds a basic neural network to classify handwritten digits from the MNIST dataset using TensorFlow and Keras. It also compares the effects of L1 and L2 regularization on the training loss and accuracy.

## Dataset
The project uses the classic MNIST dataset, which contains 70,000 grayscale images of handwritten digits (0-9) sized at 28x28 pixels.

## Model Details
The model is a Sequential network consisting of:
* An input flatten layer (784 features)
* Three Dense hidden layers (128, 64, and 32 neurons) using ReLU activation
* A final Dense output layer (10 neurons) using Softmax activation

The script tests two variants of the network to manage model weights and prevent overfitting:
1. **L2 Regularization:** Adds a penalty equivalent to the square of the magnitude of coefficients (`l2(0.002)`).
2. **L1 Regularization:** Adds a penalty equivalent to the absolute value of the magnitude of coefficients (`l1(0.002)`).

## Dependencies
Make sure you have these packages installed before running the notebook:
* tensorflow
* numpy
* matplotlib
* pandas
* seaborn
* scikit-learn

## How to Run
1. Open the Jupyter Notebook.
2. Run the cells sequentially to load the dataset, preprocess the data (normalize pixel values by dividing by 255), build the model, and view the training history plots.
