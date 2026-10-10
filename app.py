import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# Load trained CNN model
model = tf.keras.models.load_model("mnist_cnn.keras")


# Application title
st.title("MNIST Digit Classifier")

st.write("Upload a handwritten digit image")


# Upload image
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)


# Execute only after user uploads an image
if uploaded_file is not None:

    # Open uploaded image using Pillow
    image = Image.open(uploaded_file)

    # Convert image to grayscale
    image = image.convert("L")

    # Resize image to 28 x 28
    image = image.resize(
        (28, 28),
        Image.Resampling.LANCZOS
    )

    # Display image
    st.image(
        image,
        caption="Uploaded Image"
    )

    # Convert image into NumPy array
    image_array = np.array(image)

    # Normalize pixel values
    image_array = image_array / 255.0

    # Reshape according to CNN input shape
    image_array = image_array.reshape(
        1, 28, 28, 1
    )

    # Make prediction
    prediction = model.predict(image_array)

    # Find predicted digit
    predicted_digit = np.argmax(prediction)

    # Find confidence
    confidence = np.max(prediction)

    # Display prediction
    st.success(
        f"Predicted Digit: {predicted_digit}"
    )

    # Display confidence
    st.write(
        f"Confidence: {confidence:.2%}"
    )
