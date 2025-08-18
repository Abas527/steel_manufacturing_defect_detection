# streamlit_app.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from predict import predict, model_path
from detect.train_yolo import predict_yolo
import streamlit as st
from PIL import Image
from yolo_pred import draw_bounding_boxes
from models.gradcam_utils import gradcam_on_image


def main():



    st.title("Manufacturing Defect Detection App")
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded_Image", use_container_width=True)
        
        if st.button("Predict"):
            result = predict(model_path, uploaded_file)
            st.success(f"Prediction: {result}")
            if result == 1:
                bounding_box= predict_yolo(uploaded_file)
                if bounding_box:
                    image = Image.open(uploaded_file).convert("RGB")
                    drawn_image,labels=draw_bounding_boxes(image, bounding_box)
                    st.image(drawn_image, use_container_width=True)
                    st.write("Detected Defects:")
                    for label in labels:
                        st.write(label)
                else:
                    st.warning("No defects detected.")
        
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
        
                overlayed_img, pred_class = gradcam_on_image(image, model_path)
                st.image(overlayed_img, caption=f"Grad-CAM (Predicted class: {pred_class})")
            else:
                st.warning("Please upload an image first.")
                
                
if __name__ == "__main__":
    main()