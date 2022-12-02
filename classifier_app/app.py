import streamlit as st
from PIL import Image
from clf import predict
from munch import munchify
import yaml

conf = 'configs/default.yaml'
with open(conf, 'r') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
cfg = munchify(cfg)

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Diabetic Retinopathy Detection from retinal fundus")
st.write("")

file_up = st.file_uploader("Upload an image", type="png")

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second...")
    labels = predict(file_up, cfg)

    # print out the top 5 prediction labels with scores
    for i in labels:
        st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])
