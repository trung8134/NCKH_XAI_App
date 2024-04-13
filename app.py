import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import numpy as np
from XAI_process import visual_GradCAM
from XRAI_process import visual_XRAI
import matplotlib.pyplot as plt
import io
import os
import shutil


model = load_model('VGG16-Plant Disease-90.96.h5')
class_names = ['Corn Common Rust', 'Corn Gray Leaf Spot', 
               'Corn Healthy', 'Corn Northern Leaf Blight', 
               'Rice Brown Spot', 'Rice Healthy',
               'Rice Leaf Blast', 'Rice Neck Blast',
               'Wheat Brown Rust', 'Wheat Healthy', 'Wheat Yellow Rust']

def process_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image
    
def predict(image):
    image = process_image(image)
    prediction = model.predict(image)
    return class_names[np.argmax(prediction, axis=1)[0]]

dir = 'data'
def delete_files(dir):
        for filename in os.listdir(dir):
            file_path = os.path.join(dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('')
                
                
def main():
    st.set_page_config(page_icon=':books:')
    st.header('Plant Disease Classifier')
    delete_files(dir=dir) 
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        file_path = os.path.join('data', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.read())
        
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.success("Classifying well done.")
        
        # predict
        prediction = predict(image)
        st.write(f"Prediction: {prediction}")
        
        # show Grad-CAM image heatmap
        heatmap_grad = visual_GradCAM(model, 'block5_conv3', img_path=file_path, img_size=(224, 224))
        # show Grad-CAM image heatmap
        heatmap_xrai = visual_XRAI(model, file_path=file_path)
        
        with st.spinner("Processing Visualize..."):
            # Create a figure with two subplots
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            # Plot the Grad-CAM heatmap
            axs[0].imshow(heatmap_grad)
            axs[0].axis('off')
            axs[0].set_title('Grad-CAM Heatmap')

            # Plot the XRAI heatmap
            axs[1].imshow(heatmap_xrai)
            axs[1].axis('off')
            axs[1].set_title('XRAI Heatmap')

            # Save the plot to a BytesIO object
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            # Display the plot in Streamlit
            st.image(buf, caption='Heatmaps!', use_column_width=True)

            st.write("In the above two methods. Grad-CAM retrieves all potential influential regions in the image, while XRAI extracts the top 30% most important regions.")

if __name__ == '__main__':
    main()


    
    