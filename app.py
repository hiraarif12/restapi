import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from keras.preprocessing import image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the pre-trained emotion detection model
model = load_model('model.h5')

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

    # Load the saved image using Keras
    img = image.load_img(file_path, target_size=(48, 48), color_mode='grayscale')
    img_array = image.img_to_array(img)
    resized_image = np.expand_dims(img_array, axis=0)
    resized_image = resized_image / 255.0

    # Predict emotion
    prediction = model.predict(resized_image)
    prediction = np.argmax(prediction)
    final_prediction = label_map[prediction]

    # Remove the temporarily saved file
    os.remove(file_path)

    return jsonify({'prediction': final_prediction})

if __name__ == '__main__':
    app.run(debug=True)
