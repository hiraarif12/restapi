import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import zipfile

app = Flask(__name__)

# Define the directory for the extracted model
model_dir = 'extracted_model'

# Function to extract the model file
def extract_model():
    with zipfile.ZipFile('best_model.zip', 'r') as zip_ref:
        zip_ref.extract('best_model.h5', model_dir)

# Check if model file exists, if not extract from ZIP
if not os.path.isfile(os.path.join(model_dir, 'best_model.h5')):
    extract_model()

# Load the pre-trained emotion detection model
model_path = os.path.join(model_dir, 'best_model.h5')
model = load_model(model_path)

# Define emotion labels
label_map = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Ensure the "uploads" directory exists
uploads_dir = 'uploads'
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

# Define a route for predicting emotions
@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in request'})

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded file temporarily
    filename = secure_filename(image_file.filename)
    file_path = os.path.join(uploads_dir, filename)
    image_file.save(file_path)

    # Load the saved image using OpenCV (cv2)
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img_array = np.expand_dims(img, axis=0)
    img_array = img_array / 255.0

    # Predict emotion
    prediction = model.predict(img_array)
    prediction = np.argmax(prediction)
    final_prediction = label_map[prediction]

    # Remove the temporarily saved file
    os.remove(file_path)

    return jsonify({'prediction': final_prediction})

if __name__ == '__main__':
    app.run(debug=True)
