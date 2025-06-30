# Handwritten Digit Recognition Web App

This is a web application that recognizes handwritten digits (0-9) from an uploaded image. It uses a K-Nearest Neighbors (KNN) machine learning model trained on the MNIST dataset.

![Screenshot of the application](static/images/image.png)

## Features

-   **Image Upload:** Upload an image of a handwritten digit.
-   **Prediction:** The app will predict the digit in the image.
-   **Image Preview:** See the uploaded image and the prediction result.
-   **Simple UI:** A clean and simple user interface.

## Tech Stack

-   **Backend:** Python, Flask
-   **Frontend:** HTML, CSS, JavaScript
-   **Machine Learning:** Scikit-learn, Pandas, Numpy
-   **Model:** K-Nearest Neighbors (KNN)

## How to Run the Project Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Pratik740/Handwritten-Digit-Recognition.git
    cd Handwritten-Digit-Recognition
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv env
    .\env\Scripts\activate

    # For macOS/Linux
    python3 -m venv env
    source env/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Flask application:**
    ```bash
    python app.py
    ```

5.  **Open your browser** and go to `http://127.0.0.1:5000/`.

## How it Works

The user uploads an image. The Flask backend receives the image and preprocesses it to match the format of the MNIST dataset (28x28 pixels, grayscale). The K-Nearest Neighbors model then predicts the digit, and the result is displayed to the user. 