# Simple MNIST Neural Network from Scratch

This repository contains the implementation of a simple feedforward neural network to recognize handwritten digits using the MNIST dataset. The neural network is implemented from scratch in Python without using any deep learning libraries like TensorFlow or PyTorch.

## Table of Contents

- [Simple MNIST Neural Network from Scratch](#simple-mnist-neural-network-from-scratch)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
  - [Dataset](#dataset)
  - [Neural Network Architecture](#neural-network-architecture)
  - [Training the Neural Network](#training-the-neural-network)
  - [Testing the Neural Network](#testing-the-neural-network)
  - [Evaluation](#evaluation)
  - [Conclusion](#conclusion)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

## Introduction

The MNIST dataset is a large collection of handwritten digits that is commonly used for training various image processing systems. The dataset contains 60,000 training images and 10,000 testing images, each of which is a 28x28 pixel grayscale image of a single digit (0-9).

In this project, we implement a simple feedforward neural network to classify these handwritten digits. The neural network is implemented from scratch without using any deep learning libraries, which provides a good opportunity to understand the underlying mechanics of neural networks.

## Getting Started

Follow these instructions to get the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.x
- NumPy
  
## Dataset

The MNIST dataset is used in this project. It contains 60,000 training images and 10,000 testing images, each of which is a 28x28 pixel grayscale image of a single digit (0-9). The dataset is automatically downloaded and loaded using the `load_data()` function in the `mnist.py` module.

## Neural Network Architecture

The neural network implemented in this project is a simple feedforward neural network with one hidden layer. The input layer has 784 nodes, corresponding to the 28x28 pixels in the input image. The hidden layer has 128 nodes, and the output layer has 10 nodes, corresponding to the 10 possible digits (0-9). The activation function used in the hidden layer is the sigmoid function, and the output layer uses the softmax function to convert the network output into probabilities.

## Training the Neural Network

The neural network is trained using the stochastic gradient descent (SGD) algorithm. The training data is divided into mini-batches, and the network parameters are updated based on the gradient of the loss function with respect to the parameters. The loss function used is the cross-entropy loss, which measures the difference between the predicted probabilities and the true class labels.

## Testing the Neural Network

After training, the neural network is tested on the test dataset to evaluate its performance. The test dataset is used to compute the accuracy of the neural network, which is the percentage of correctly classified images.

## Evaluation

The performance of the neural network can be evaluated by computing various metrics such as accuracy, precision, recall, and F1 score. These metrics provide a comprehensive view of the model's performance and can be used to compare different models.

## Conclusion

In this project, we implemented a simple feedforward neural network to recognize handwritten digits using the MNIST dataset. The neural network was implemented from scratch in Python without using any deep learning libraries. We trained the neural network using the stochastic gradient descent algorithm and evaluated its performance on the test dataset.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen
