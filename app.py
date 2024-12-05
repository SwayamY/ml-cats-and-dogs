from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import cv2
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('dog_cat_classifier.keras')  # Use the updated model file
categories = ["dog", "cat"]  # Update categories

app = Flask(__name__)

# Ensure the uploaded images directory exists
UPLOAD_FOLDER = 'static/uploaded'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the directory 
from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import cv2
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('dog_cat_classifier.keras')  # Use the updated model file
categories = ["dog", "cat"]  # Update categories

app = Flask(__name__)

# Ensure the uploaded images directory exists
UPLOAD_FOLDER = 'static/uploaded'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the directory if it doesn't exist
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Process the uploaded image
    img = cv2.imread(filepath)
    if img is None:
        return render_template('index.html', error="Failed to process the uploaded image.")

    img_resized = cv2.resize(img, (150, 150)) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    # Make a prediction
    prediction = model.predict(img_resized)
    predicted_category = categories[np.argmax(prediction)]
    confidence = np.max(prediction)

    return render_template('index.html', prediction=predicted_category, confidence=f"{confidence * 100:.2f}%", filepath=filepath)

if __name__ == '__main__':
    app.run(debug=True)
