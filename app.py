from flask import Flask,render_template,request
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from PIL import Image , ImageOps # Used for opening and processing image files.

app = Flask(__name__)
uploadFolder = 'static/uploads'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = joblib.load('knn_model.pkl')

@app.route('/')       #
def hello_world():
    return render_template("index.html",prediction=None,image = None)

# @app.route('/products')    
# def products():
#     return 'This is nothing.'


def crop_and_center(img):
    # Invert the image to get a bounding box of the digit
    inverted = ImageOps.invert(img)
    bbox = inverted.getbbox()
    if bbox:
        img_cropped = img.crop(bbox)
        # Pad the cropped image to 28x28 while centering it
        img_centered = ImageOps.pad(img_cropped, (28, 28), color=0)
        return img_centered
    return img


@app.route('/upload',methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No file part'
    
    image = request.files['image']
    if(image.filename == ''):
        return 'No file selected'
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(filepath)

    # Preprocessing image to match digits dataset
    img = Image.open(filepath).convert('L') # gray-scale conversion 
    img = crop_and_center(img)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
   

    img_data = np.array(img) # converts to a numpy array
    img_data = img_data / 255.0  # normalize
    
    img_data = img_data.flatten().reshape(1,-1) # adds a dimension for model.predict

    # import matplotlib.pyplot as plt
    # plt.imshow(img_data.reshape(28, 28), cmap='gray')
    # plt.show() 


    prediction = model.predict(img_data)[0]  
    rel_path = os.path.join('uploads', image.filename).replace('\\', '/')
    return render_template('index.html', prediction=prediction, image=rel_path)


if __name__ == "__main__":
    app.run(debug=True,threaded=True)


#request.files	: Contains all uploaded files from form (dictionary-like)
# Its keys are name attributes of the input button in html form  

#request.files['image'] : Accesses the file with name "image" from the form and returns filestorage object.


