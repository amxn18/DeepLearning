# 🧠 CNN Model Optimization using Keras Tuner 🔍

This project demonstrates the process of optimizing a Convolutional Neural Network (CNN) using **Keras Tuner**. The model is designed for image classification tasks such as digit recognition or similar grayscale image datasets.

## 📁 Project Overview

- Built a CNN with two convolutional layers and one dense layer.
- Tuned the number of filters, kernel sizes, dense units, and learning rate using hyperparameter search.
- Used **Random Search** to explore different combinations of hyperparameters.
- Tracked results using Keras Tuner’s in-built logging system.

## ⚙️ Model Components

- Two convolutional layers using ReLU activation.
- One fully connected dense layer with softmax activation.
- Adam optimizer with tunable learning rate.
- Sparse categorical crossentropy as the loss function.
- Input shape is assumed to be (28, 28, 1).

## 🔍 Tuning Strategy

- Hyperparameters tuned include:
  - Number of filters in each Conv2D layer
  - Kernel sizes for convolutions
  - Number of units in the dense layer
  - Learning rate for the optimizer

- Used **validation accuracy** as the optimization objective.
- Limited to 5 trials to keep the tuning lightweight and fast.

## 🗂 Output

- Best performing hyperparameter configuration can be extracted from the tuner object.
- Keras Tuner creates an `output/` directory with tuning logs and saved trials.
- You can use the best model configuration for final training or inference.

## ✅ Notes

- Ensure the input data is normalized and shaped correctly before training.
- Keras Tuner automatically logs the training process and displays the top models found.
- This setup is ideal for quick experimentation and finding performant CNN architectures.

---

This setup gives a great starting point for tuning image models in a structured and reproducible way.
