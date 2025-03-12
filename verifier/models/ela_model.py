import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import os
import cv2
import pickle
import kaggle

# Download CASIA 2.0 dataset using Kaggle API
dataset = "divg07/casia-20-image-tampering-detection-dataset"
data_dir = "CASIA2.0/"
kaggle.api.dataset_download_files(dataset, path=data_dir, unzip=True)

categories = ["authentic", "tampered"]  # Adjust folder names as per dataset structure

data = []
labels = []
img_size = 128  # Resize images for consistency

# Load images and apply preprocessing
for category in categories:
    path = os.path.join(data_dir, category)
    class_label = categories.index(category)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            img = img / 255.0  # Normalize
            data.append(img)
            labels.append(class_label)

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification (Authentic/Tampered)
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
epochs = 10
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

# Save trained model as .pkl
with open("ela_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved as ela_model.pkl")
