from flask import Flask, request, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the PCOS detection model from the current directory
model_path = os.path.join(os.path.dirname(__file__), 'PCOS_model.keras')
model = tf.keras.models.load_model(model_path)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    
    # Open and preprocess the image
    img = Image.open(file).convert('RGB')  # Ensure image is in RGB format
    img = img.resize((224, 224))  # Resize to match model input size (224x224)
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 224, 224, 3)

    # Make prediction
    prediction = model.predict(img_array)
    result = "PCOS Detected" if prediction[0][0] > 0.5 else "No PCOS Detected"  # Threshold at 0.5

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)