import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib


# Samples : 70000 total images (len(digits.data) or len(digits.target))
# Classes : 10(digits from 0 to 9)
# Image Size : 28x28 pixels (grayscale)
# Total Features : 784 per image (28 × 28 = 784 intensity values, flattened into a vector)

# mnist.data : 2-d Matrix (70000 * 784) - each image (ie. row of 28*28 pixel) is flattened into 1-d vector of size 784.
# mnist.target : array of labels (0 to 9)
# mnist.images : 28*28 pixel images as 2-d arrays 

mnist = fetch_openml('mnist_784',version=1)
X = mnist.data
y = mnist.target.astype('int')

X = X/255.0

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, y_train)

# from sklearn.metrics import accuracy_score, classification_report

# Predict on test set


# # Evaluate performance
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))



joblib.dump(model,'knn_model.pkl')
# When you train an ML model, you want to reuse it without training again every time.

# When any image is uploaded , it calculates it euclidean distance with every trained instances (ie. 784 flattend image) 
# images and finds nearest 7 neigbours from it and outputs maximum of 7 labels from it.




