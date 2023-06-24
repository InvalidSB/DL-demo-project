import streamlit as st
import tensorflow as tf
# import tensorflow.keras.preprocessing.image as image
import numpy as np
import PIL.Image as Image
# from cv2 import (VideoCapture, namedWindow, imshow,
#                  waitKey, destroyWindow, imwrite)

classes = ['Class1', 'Class2', 'Class3', 'Class4']
# model = tf.keras.models.load_model('saved_model.pb')
model = tf.keras.models.load_model('./model')

image_size = (128, 128)


def process_image(model, image_path, classes):

    labels = ['Alternaria_Leaf_Spot', ' Black Rot',
              'Cabbage Aphid', 'Cabbage Looper', ' Healthy Leaf']

    image = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(512, 512))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image/255.0
    image = tf.expand_dims(image, 0)

    pred = model.predict(image)
    confidence = pred.max() * 100
    predicted_class = labels[pred.argmax(axis=-1)[0]]
    # setting up the threshhold
    if(confidence >= 80):
        return "Sorry couldnot recognize it as cauliflower leaf"

    # print(predicted_class)
    return predicted_class


def suggest_remedies(disease):
    remedies = {
        'Alternaria_Leaf_Spot': 'Remedy for Alternaria Leaf Spot',
        'Black Rot': 'Remedy for Black Rot',
        'Cabbage Aphid': 'Remedy for Cabbage Aphid',
        'Cabbage Looper': 'Remedy for Cabbage Looper',
        'Healthy Leaf': 'No remedies needed. The leaf is healthy.'
    }

    return remedies.get(disease, 'No remedies found.')


st.set_page_config(page_title="Cauliflower disease classification...",
                   page_icon=":camera:", layout="centered")
st.title("Cauliflower disease classification..")
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if file:
    st.image(file, caption='Uploaded Image.', width=200)
    if st.button("Submit"):
        result = process_image(model, file, classes)
        st.success(
            "The image has been gone through classification process successfully.")
        st.markdown('<h1 style="font-size: 30px; text-align: center">Result :</h1>',
                    unsafe_allow_html=True)
        Result_text = '<p style="font-family:Courier; color:yellow;font-weight:bold; font-size: 60px; text-align: center">{}</p>'
        st.markdown(Result_text.format(result), unsafe_allow_html=True)

        suggested_remedy = suggest_remedies(result)
        st.markdown(f"Remedy: {suggested_remedy}")
