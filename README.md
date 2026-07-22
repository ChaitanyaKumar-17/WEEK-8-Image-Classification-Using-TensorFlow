# 🖼️ Image Classification System with TensorFlow

A robust Python deep learning project designed to ingest image data, preprocess pixels, and build a multi-class image classification model using a Fully Connected Deep Network[cite: 1]. This project emphasizes neural network best practices in supervised learning, dropout regularization, and evaluating model performance visually and mathematically.

---

## 💡 Overview

Modern computer vision applications rely on deep learning to identify and categorize objects within images. This project automates a workflow for extracting patterns from the CIFAR-10 dataset—a foundational task in AI-driven image recognition[cite: 1]. By normalizing pixel data, defining a sequential neural network architecture, and visualizing learning curves, the model transforms raw image matrices into identifiable categorical classes like 'Automobile', 'Bird', and 'Ship'[cite: 1].

---

## ✨ Key Features

*   **Data Acquisition & Preprocessing:** Designed to ingest the CIFAR-10 dataset, scale pixel values to a 0-1 range (float32), and apply one-hot encoding to target labels for categorical classification[cite: 1].
*   **Overfitting Prevention:** Utilizes `tensorflow.keras.layers.Dropout` at rates of 0.2 and 0.3 across hidden layers to ensure the model generalizes well to unseen data[cite: 1].
*   **Deep Neural Network Architecture:** Flattens 32x32x3 images into vectors of 3072 neurons and passes them through dense layers (256, 128, and 64 neurons) using ReLU activations[cite: 1].
*   **Comprehensive Evaluation:** Computes standard classification metrics using `scikit-learn`, including Accuracy, Precision, Recall, and F1-score (weighted average)[cite: 1].
*   **Visual Diagnostics:** Automatically generates side-by-side plots for training versus validation loss and accuracy, alongside random visualizations of both correct and misclassified images[cite: 1].
*   **Model Persistence:** Features built-in functionality to save the trained model as an `.h5` file (`cifar10_ann.h5`) and reload it for future inference[cite: 1].

---

## 🛠️ Prerequisites

Ensure you have the following installed before running the project:
*   Python 3.8 or higher
*   A standard Python IDE (VS Code, PyCharm) or Jupyter Notebook
*   Core Scientific and Deep Learning Libraries: `tensorflow`, `numpy`, `scikit-learn`, `matplotlib`[cite: 1]

---

## 🚀 Quick Start Guide

1.  Clone this repository to your local machine.
2.  Open your terminal or command prompt and install the necessary dependencies:
    ```bash
    pip install tensorflow numpy scikit-learn matplotlib
    ```
3.  Launch your Python environment or execute the script directly:
    ```bash
    python app.py
    ```

---

## 📊 Expected Output

Upon successful execution, the script will process the CIFAR-10 image data and output statistical metrics and visual insights, including[cite: 1]:

*   **Model Summary:** A terminal printout detailing the Sequential model's layer structure and parameter counts[cite: 1].
*   **Learning Curves:** Figure plots displaying Training vs. Validation Loss and Training vs. Validation Accuracy over 15 epochs[cite: 1].
*   **Evaluation Metrics:** A console readout displaying the final Accuracy, Precision, Recall, and F1-Score to 4 decimal places[cite: 1].
*   **Prediction Visualizations:** Grid plots showing randomly selected test images, labeled with their True (T) and Predicted (P) class names, color-coded green for correct and red for incorrect classifications[cite: 1].

---

## 🧩 Pipeline Architecture: How It Works

This script serves as a practical application of standard supervised deep learning pipelines tailored for image classification:

1.  **Data Ingestion:** The script loads the CIFAR-10 dataset, instantly splitting it into training (`x_train`, `y_train`) and testing sets (`x_test`, `y_test`)[cite: 1].
2.  **Standardization:** Image pixels are scaled by dividing by 255.0, bringing all features to a normalized 0-1 scale, while categorical labels are one-hot encoded to 10 distinct classes[cite: 1].
3.  **Model Construction:** The normalized data flows into a Sequential Keras model[cite: 1]. The data is flattened, processed through multiple dense hidden layers equipped with ReLU activations and Dropout, and culminates in a 10-neuron Softmax output layer for probability distribution[cite: 1].
4.  **Compilation & Training:** The network is compiled using the `adam` optimizer and `categorical_crossentropy` loss, then trained in batches of 64 over 15 epochs[cite: 1].
5.  **Metric Calculation:** Predictions are reduced via `np.argmax` and compared against the flattened true labels to compute weighted accuracy, precision, recall, and F1 scores[cite: 1].
6.  **Visual Evaluation & Saving:** The script visualizes learning progression and prediction accuracy on actual images before saving the fully trained model to disk for future deployment[cite: 1].