
## **Signature Detection Using YOLOv5 and Detectron2**

This project focuses on detecting signatures in images using two advanced deep learning models: YOLOv5 and Detectron2. The objective was to create a robust signature detection system capable of working with a small dataset. We had to chose between Yolo and Detectron2 but since I haven't worked with Detectron2 in the past, I have used both of them to learn more about Detectron2.

---

### **Project Overview**
We tackled the challenge of detecting signatures using two approaches:
1. **YOLOv5**: Known for its speed and accuracy with minimal computational power requirements.
2. **Detectron2**: A flexible and efficient object detection framework by Facebook AI.

The project involved:
- Collecting and preparing a small dataset of 25 signature images.
- Annotating the dataset for both YOLOv5 and Detectron2 using the LabelImg tool.
- Training both models in Google Colab, leveraging its GPU resources for faster computations.
- Visualizing and validating the models on test images.
- Deploying the models using a **Streamlit** application for real-time predictions.

---

### **Dataset Details**

#### **Source**
The dataset was manually created and annotated, focusing on detecting signatures in images. It was stored in a Google Drive folder for easy access and organization.

#### **Structure**
The dataset was divided into two primary folders:
1. **`signature_images`**: Contains the raw images of signatures used for training and validation.
2. **`signature_annotations`**: Contains the corresponding `.txt` annotation files for each image, created using the LabelImg tool.

These folders can be downloaded from the following:
Annotaions: https://drive.google.com/drive/folders/1brMgRrG-uoMnzqbD4eHPLJ3DOTX9O9jq?usp=sharing
Images: https://drive.google.com/drive/folders/1VZl-KaW2YazxTSbSDH9rKYb9g19Md37b?usp=sharing

#### **Format Requirements**
- **YOLOv5**:
  - Requires images and their corresponding label files in YOLO format.
  - Each label file contains bounding box coordinates in YOLO format, with each line representing a detected object.
- **Detectron2**:
  - Requires the dataset in COCO format with `.json` annotation files.
  - The annotation files were converted from YOLO format to COCO format using Python scripts.

#### **Challenges**
1. **Small Dataset Size**:
   - Initially, the dataset consisted of only **25 images**. This was insufficient for training robust machine learning models. To address this:
     - **Data augmentation** techniques such as flipping, rotation, brightness adjustment, and Gaussian blur were applied, significantly increasing the dataset size.
2. **Annotation Inconsistency**:
   - During the annotation process, the LabelImg tool assigned a default class ID of `15` to all bounding boxes. This caused incompatibility issues with both YOLOv5 and Detectron2, which expect the class IDs to start from `0`.
   - The class IDs in all annotation files were corrected programmatically to ensure compatibility.

By addressing these challenges and augmenting the dataset, we ensured that both YOLOv5 and Detectron2 models had sufficient and well-annotated data for training. 

---

### **How the Models Were Trained**
#### **YOLOv5**
- **Training Steps**:
  1. Annotated the dataset using LabelImg and converted labels to YOLO format.
  2. Performed data augmentation to increase dataset size.
  3. Trained the YOLOv5x model using the expanded dataset on Google Colab.

- **Training Results**:
  - Achieved a high precision and recall on the validation set.
  - mAP@50 (Mean Average Precision at IoU=0.5): `99.5%`

#### **Detectron2**
- **Training Steps**:
  1. Converted annotations to COCO format.
  2. Configured the Detectron2 framework for single-class detection (`signature`).
  3. Trained the model on Google Colab and evaluated using the COCOEvaluator.

- **Training Results**:
  - Average Precision (AP) @ IoU=0.5: `96.7%`

---

### **Streamlit Application**
After training, both models were integrated into a Streamlit application for easy deployment and testing. Key features of the app:
- Upload an image and run predictions using both YOLOv5 and Detectron2.
- Visualize bounding boxes and confidence scores for detected signatures.
- Compare results from both models.

#### **Sample Predictions**
Below are some examples of signature detection results:

| Input Images:
|![image](https://github.com/user-attachments/assets/0e564eb9-1159-49e7-9b44-f78446517bb7)
YOLOv5 Prediction: ![image](https://github.com/user-attachments/assets/ccab8f11-a925-41a8-b284-f4e3884c68a6)
Detectron2 Prediction: ![image](https://github.com/user-attachments/assets/f8d1d529-e826-4ff8-9afb-dbe69173c816)

---

### **How to Use This Project**
#### **1. Google Colab Setup**
1. Open the https://colab.research.google.com/drive/1FalLjU5NfDDp8ZjmmKU19pyFMDduJ1UA?usp=sharing.
2. Create a copy in your Google Drive.
3. Download the dataset from Google Drive Link and upload it to Colab or drive wherever you feel.
4. Update paths in the Colab notebook to match your file locations.
5. Follow the steps in the notebook to preprocess the dataset, train the models, and download the trained weights.

#### **2. Streamlit Application**
1. Clone this repository.
2. Place the downloaded YOLOv5 and Detectron2 model weights in the `streamlit/saved_models/` folder.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app (only after adjusting the paths, otherwise it might fail):
   ```bash
   streamlit run streamlit/streamlit_app.py
   ```
5. Open the application in your browser and upload images for prediction.

---

### **Key Files and Folders**
- `jupyternbk/`: Contains the training notebook for YOLOv5 and Detectron2.
- `streamlit/`: Includes the Streamlit app script and saved models.
- `README.md`: This file, detailing the project.

---
