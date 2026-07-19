# Data Loading and Preprocessing
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values (0-1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode labels for categorical_crossentropy
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Class names for visualization later
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


# Defining the Fully Connected Deep Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

model = Sequential([
    # Flatten images to vectors (Input layer: 3072 neurons)
    Flatten(input_shape=(32, 32, 3)),
    
    # Hidden layers with ReLU activations and Dropout to prevent overfitting
    Dense(256, activation='relu'),
    Dropout(0.3),
    
    Dense(128, activation='relu'),
    Dropout(0.3),
    
    Dense(64, activation='relu'),
    Dropout(0.2),
    
    # Output layer: 10 neurons with Softmax
    Dense(10, activation='softmax')
])

model.summary()


# Compilation and Training
# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train_cat, 
                    epochs=15, 
                    batch_size=64, 
                    validation_data=(x_test, y_test_cat))

# Visualize Learning Curves (Loss)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Visualize Learning Curves (Accuracy)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Evaluation Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Get model predictions
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test.flatten()

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")


# Visualizing Predictions and Misclassifications
# Find indices of correct and incorrect predictions
correct_indices = np.where(y_pred == y_true)[0]
incorrect_indices = np.where(y_pred != y_true)[0]

def plot_images(indices, title):
    plt.figure(figsize=(10, 4))
    for i, idx in enumerate(np.random.choice(indices, 5, replace=False)):
        plt.subplot(1, 5, i + 1)
        plt.imshow(x_test[idx])
        plt.title(f"T: {class_names[y_true[idx]]}\nP: {class_names[y_pred[idx]]}", 
                  fontsize=10, color="green" if y_true[idx] == y_pred[idx] else "red")
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Visualize Random Predictions
plot_images(correct_indices, "Correct Predictions")

# Visualize Misclassified Images
plot_images(incorrect_indices, "Misclassified Images")


# Saving and Reloading the Model
import os

# Save the model
model_path = 'cifar10_ann.h5'
model.save(model_path)
print(f"Model saved to {model_path}")

# Reload and reuse the model
reloaded_model = tf.keras.models.load_model(model_path)

# Verify the reloaded model works by evaluating it
loss, acc = reloaded_model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Reloaded Model Accuracy: {acc:.4f}")