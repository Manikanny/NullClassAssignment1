import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image


model = tf.keras.models.load_model('Age_Sex_Detection.keras') # Loading our the pre-trained model

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def get_activation_maps(model, img_array, layer_name):
    intermediate_layer_model = tf.keras.models.Model(inputs=model.input,
                                                     outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(img_array)
    return intermediate_output[0]

def display_activation_maps(activation_maps):
    num_filters = activation_maps.shape[-1]
    size = activation_maps.shape[1]
    cols = 8 
    rows = num_filters // cols if num_filters % cols == 0 else (num_filters // cols) + 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(30, 30))
    axes=axes.flatten()
    for i in range(num_filters):
        ax = axes[i]
        ax.matshow(activation_maps[:, :, i], cmap='viridis')
        ax.axis('off')
    st.pyplot(fig)

st.title("Activation Map Viewer")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

layer_name = st.text_input("Enter the layer name (e.g., conv2d_9):")

if uploaded_file is not None and layer_name:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image')
        
        img_path = uploaded_file.name
        img_array = load_and_preprocess_image(uploaded_file)

        activation_maps = get_activation_maps(model, img_array, layer_name)
        
        st.subheader(f"Activation Maps for Layer: {layer_name}")
        display_activation_maps(activation_maps)
    except Exception as e:
        st.error(f"Error: {e}")
