import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import matplotlib.pyplot as plt

# Set up categories
data_dir = r"C:\Users\Sweyam\Desktop\difference\model\imgdogandcat"  # Update this path
categories = ["cat", "dog"]

# Function to load images and assign labels
def load_images(data_dir, categories, img_size=150):
    data = []
    for category in categories:
        path = os.path.join(data_dir, category)
        print(f"Processing category: {category}, path: {path}")
        if not os.path.exists(path):
            print(f"Directory {path} does not exist!")
            continue

        label = categories.index(category)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img_array is None:
                    print(f"Failed to load image {img_path}")
                    continue

                resized_img = cv2.resize(img_array, (img_size, img_size))
                data.append([resized_img, label])
            except Exception as e:
                print(f"Error loading image {img} from {path}: {e}")
    return data

# Load dataset
data = load_images(data_dir, categories)

# Separate features and labels
X, y = zip(*data)
X = np.array(X) / 255.0  # Normalize
y = np.array(y)

# Print class distribution
print("Class distribution:", Counter(y))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Compute class weights to handle imbalance
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')  # Output layer
])

# Optional: Transfer Learning using MobileNetV2
# Uncomment this block to use a pre-trained model
from tensorflow.keras.applications import MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()

# Add early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with augmentation
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=25,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stopping]
)

# Evaluate model performance
y_pred = np.argmax(model.predict(X_test), axis=-1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=categories))

# Save the trained model
model.save('cat_dog_classifier.keras')

# Function to analyze a new image
def upload_and_analyze_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image at {image_path}")
        return
    
    img_resized = cv2.resize(img, (150, 150)) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_resized)
    predicted_category = categories[np.argmax(prediction)]
    confidence = np.max(prediction)
    print(f"Predicted Category: {predicted_category} ({confidence * 100:.2f}%)")

# Example usage
upload_and_analyze_image(r"C:\Users\Sweyam\Documents\jaba\ex8.jpeg")
