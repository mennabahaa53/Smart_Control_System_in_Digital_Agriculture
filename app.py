#!/usr/bin/env python
# coding: utf-8

# In[18]:


import flask
import io
import string
import time
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, request 


# In[19]:


# Load the VGG16 model
model = tf.keras.models.load_model('Mymodel.h5')


# In[20]:


def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img


def predict_result(img):
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)__Common_rust', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape__Esca(Black_Measles)', 'Grape___healthy', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Orange__Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,bell__Bacterial_spot', 'Pepper,bell__healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']
    predicted_class_name = class_names[predicted_class]

    response = {
        'prediction': predicted_class_name
    }

    return jsonify(response)


# In[21]:


app = Flask(__name__)


# In[22]:


@app.route('/predict', methods=['POST'])
def infer_image():
    # Catch the image file from a POST request
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"
    
    file = request.files.get('file')

    if not file:
        return

    # Read the image
    img_bytes = file.read()

    # Prepare the image
    img = prepare_image(img_bytes)

    # Get the prediction
    prediction = predict_result(img)

    # Display the prediction
    print(prediction)

    # Return the prediction
    return prediction
    

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


# In[12]:


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')


# In[ ]:




