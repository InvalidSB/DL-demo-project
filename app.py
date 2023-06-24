import streamlit as st
import tensorflow as tf
import numpy as np
import PIL.Image as Image

# model = tf.keras.models.load_model('saved_model.pb')
model = tf.keras.models.load_model('./model')

image_size = (128, 128)


def process_image(model, image_path):

    labels = ['Alternaria Leaf Spot', 'Black Rot',
              'Cabbage Aphid', 'Cabbage Looper', 'Healthy Leaf']

    image = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(512, 512))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image/255.0
    image = tf.expand_dims(image, 0)

    pred = model.predict(image)
    confidence = pred.max() * 100
    predicted_class = labels[pred.argmax(axis=-1)[0]]

    # print(predicted_class)
    return predicted_class, confidence


def suggest_remedies(disease):
    remedies = {
        'Alternaria Leaf Spot': '''<h3>Chemical controls for Alternaria leaf spot are:</h3>
                                <ul>
                                <li>
                                    Cabrio EG (Group 11) at 12 to 16 oz/A on 7- to 14-day intervals.
                                </li>
                                <li>
                                                            Miravis Prime (Group 7 + 12) at 11.4 fl oz/A on 7-day intervals.
                                </li>
                                </ul>''',
        'Black Rot': '''<h3>Chemical controls for black rot are:</h3>
                            <ul>
                            <li>
                                        Actigard at 0.5 to 1 oz/A every 7 days for up to four (4) applications per
                                                season.
                            </li>
                            <li>
                                            Champ WG (Group M1) at 1.06 lb/A on 7- to 10-day intervals.
                            </li>
                            </ul>''',

        'Cabbage Aphid': '''<h3>Chemical controls for cabbage aphid are:</h3>
                                <ul>
                                <li>
                                Apply when aphid first appear; repeat at 8- to 10-day intervals.
                                </li>
                                <li>
                                Acetamiprid (Assail 30SG) at 0.038 to 0.075 lb ai/A. PHI 7 days. REI 12 hr.
                                </li>
                                <li>
                                Borate complex (Prev-Am Ultra) as 0.8% solution. REI 12 hr. Spray to
                                    complete coverage.
                                </li>
                                </ul>
                                    ''',

        'Cabbage Looper': '''<h3>Chemical controls for cabbage looper are:</h3>
                                <ul>
                                <li>
                                Acephate (Acephate 90WDG) at 1.0 lb ai/A. PHI 14 days.
                                </li>
                                <li>
                                Azadirachtin (Neemix 4.5) at 0.14 to 0.35 lb ai/A. PHI 0 days. REI 4 hr.
                                                    OMRI-listed for organic use.
                                </li>
                                </ul>
                                                ''',
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
        result, confidence = process_image(model, file)

        st.success(
            "The image has been gone through classification process successfully.")
        if confidence >= 75.0:
            st.markdown('<h1 style="font-size: 30px; text-align: center">Result :</h1>',
                        unsafe_allow_html=True)
            Result_text = '<p style="font-family:Courier; color:yellow;font-weight:bold; font-size: 60px; text-align: center">{}</p>'
            st.markdown(Result_text.format(result), unsafe_allow_html=True)
            st.markdown(
                f'<p style=" color:green;font-weight:bold; font-size: 20px;">Classified with the accuracy of {round(confidence, 2)}%</p>', unsafe_allow_html=True)

            suggested_remedy = suggest_remedies(result)
            st.markdown(f"{suggested_remedy}", unsafe_allow_html=True)
        else:
            st.warning(
                "Could not classify the image with sufficient confidence.")
