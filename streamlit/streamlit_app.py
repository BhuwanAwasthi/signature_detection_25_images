import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Function to register a custom dataset
def register_custom_dataset():
    dataset_name = "signature_detection"
    # Avoid re-registering the dataset
    if dataset_name not in MetadataCatalog.list():
        MetadataCatalog.get(dataset_name).set(thing_classes=["signature"])
        MetadataCatalog.get(dataset_name).set(evaluator_type="coco")

# Register the custom dataset
register_custom_dataset()

# Load YOLOv5 model
import torch

try:
    yolo_model = torch.hub.load(
        'ultralytics/yolov5', 'custom', path='/workspaces/signature_detection_small_dataset/trained_model/best.pt'
    )
except Exception as e:
    st.error(f"Error loading YOLOv5 model: {e}")
    st.stop()

# Configure Detectron2
try:
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'  # Use CPU for inference
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # For the "signature" class
    cfg.merge_from_file("detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "/workspaces/signature_detection_small_dataset/trained_model/detectron2_signature_model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set confidence threshold
    predictor = DefaultPredictor(cfg)
except Exception as e:
    st.error(f"Error configuring Detectron2: {e}")
    st.stop()

# Streamlit UI
st.title("Signature Detection with YOLOv5 and Detectron2")
st.write("Upload an image to detect signatures using two models.")

# Upload Image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_image:
    try:
        # Load image
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        # YOLOv5 Inference
        st.write("### YOLOv5 Predictions")
        try:
            yolo_results = yolo_model(image)  # Use the correct YOLO inference method
            yolo_boxes = yolo_results.xyxy[0].cpu().numpy()
            yolo_annotated_image = image.copy()
            for box in yolo_boxes:
                x1, y1, x2, y2, conf, cls = box
                label = f"Signature: {conf:.2f}"  # Add label with confidence
                cv2.rectangle(yolo_annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    yolo_annotated_image,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
            st.image(yolo_annotated_image, caption="YOLOv5 Detection", use_container_width=True)
        except Exception as e:
            st.error(f"Error during YOLOv5 inference: {e}")

        # Detectron2 Inference
        st.write("### Detectron2 Predictions")
        try:
            outputs = predictor(image)
            v = Visualizer(
                image[:, :, ::-1],
                MetadataCatalog.get("signature_detection"),  # Use custom dataset name
                scale=1.2,
            )
            detectron_annotated_image = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            st.image(
                detectron_annotated_image.get_image()[:, :, ::-1],
                caption="Detectron2 Detection",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Error during Detectron2 inference: {e}")

    except Exception as e:
        st.error(f"Error processing the uploaded image: {e}")

