# Deep Learning Implementations in TensorFlow

This repository contains a collection of Python scripts that implement various deep learning models and concepts using TensorFlow and its Keras API. Each script is a self-contained example designed to demonstrate a specific technique or architecture.

---

## Scripts Overview

The repository is organized into standalone scripts, each focusing on a distinct topic.

### Core Models

These scripts cover fundamental neural network implementations.

* `neural_network.py`: Implements a basic **feed-forward neural network** (Dense network) to demonstrate the core structure of fully connected layers.
* `regression_model.py`: Focuses on building a model to predict **continuous numerical values**, covering concepts such as **Mean Squared Error (MSE)** loss.
* `classification_problem.py`: A dedicated script for **classification tasks** where the goal is to predict discrete class labels, exploring concepts like **categorical cross-entropy** loss and **softmax** activation.

---

### Computer Vision

This section contains scripts for solving problems with image data using Convolutional Neural Networks (CNNs).

* `CNN_problem.py`: Implements a standard **Convolutional Neural Network (CNN)** from scratch, demonstrating the use of core layers like `Conv2D`, `MaxPooling2D`, and `Flatten` for image classification.
* `data_augmentation.py`: Explores techniques for **artificially expanding an image dataset** by applying transformations like random rotations, zooms, and flips to training data to improve model generalization and reduce overfitting.
* `transfer_learning.py`: Demonstrates the technique of **transfer learning**, which involves using a pre-trained model (e.g., VGG16, ResNet) and fine-tuning it for a new, specific task to achieve high performance with less data.
* `transpose_convolution.py`: Explores **transpose convolutions** (also known as "deconvolutions"). This script builds models that upsample feature maps, a key operation in generative architectures like Autoencoders or Generative Adversarial Networks (GANs).

---

### Advanced Architectures and Customization

These scripts cover more complex models and advanced framework features.

* `Transformer.py` & `transformer1.py`: These scripts contain implementations of the **Transformer architecture**. This model, which revolutionized Natural Language Processing (NLP), is based on **self-attention mechanisms**. The files explore key components like multi-head attention, positional encodings, and encoder-decoder stacks.
* `custom_layers.py`: This script showcases how to build **custom layers** in Keras by subclassing the `tf.keras.layers.Layer` class. This is an advanced feature that allows for the creation of unique, non-standard model architectures.

---

## Key Concepts Covered

This repository demonstrates hands-on implementation of a wide range of essential deep learning topics, including:

* **Fundamental Models**: Regression and Classification.
* **Convolutional Neural Networks (CNNs)** for Computer Vision.
* **Advanced Vision Techniques**: Transfer Learning, Data Augmentation, and Transpose Convolutions.
* **Modern NLP Architectures**: The Transformer model and its self-attention mechanism.
* **Advanced Keras Functionality**: Creating custom, reusable layers.
* **Core Deep Learning Workflow**: Data preprocessing, model building, training, and evaluation using TensorFlow.