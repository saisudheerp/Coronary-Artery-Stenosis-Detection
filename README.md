# Coronary Artery Stenosis Detection Using YOLOv8

## Overview

This project focuses on the development and implementation of a real-time coronary artery stenosis (CAS) detection system using the YOLOv8 deep learning model. YOLOv8 was chosen for its balance between high accuracy and fast inference speed, making it ideal for real-time clinical applications. CAS is a life-threatening condition caused by the narrowing of coronary arteries, which can lead to critical cardiovascular events. By leveraging machine learning, this project provides a non-invasive, efficient, and cost-effective solution to aid in early detection and diagnosis.

---

## Features

- **High Accuracy**: Achieved a precision of 95.3% and recall of 93.6%.
- **Real-time Capability**: Inference time of 12.8 milliseconds per image.
- **User-Friendly Workflow**: Provides easy-to-use scripts for training, validation, and prediction.
- **Data Visualization**: Includes tools to visualize results and confusion matrices.

---

## Dataset

- **Training Set**: 6700 images
- **Validation Set**: 1542 images
- **Test Set**: 800 images
- **Annotation Format**: YOLO-compatible labels were used.

The dataset comprises high-quality coronary angiography images annotated for stenosis regions.

---

## Methodology

1. **Data Preprocessing**: The dataset was thoroughly cleaned by removing duplicate and low-quality images to ensure optimal model performance. Annotations were converted to the YOLOv8-compatible format, and additional preprocessing steps included resizing images to 800x800 pixels and normalizing pixel values to a range of 0-1. Data augmentation techniques, such as random rotations, flipping, and scaling, were applied to improve model generalization.
2. **Model Selection**: YOLOv8 was chosen for its balance between accuracy and inference speed, providing a lightweight yet powerful architecture suitable for real-time applications.
3. **Training**: The model was fine-tuned for 25 epochs using a learning rate of 0.001 and a batch size of 16. Pre-trained weights from YOLOv8s were used to initialize the model, allowing it to adapt quickly to the coronary angiography dataset. The training process monitored both object detection and classification loss to optimize performance.
4. **Validation and Testing**: Model performance was rigorously evaluated using metrics such as precision, recall, and mean Average Precision (mAP) at various IoU thresholds. Validation included analyzing confusion matrices and bounding box predictions to ensure robustness and reliability.

---

## Results

- **Precision**: 95.3%
- **Recall**: 93.6%
- **mAP@50**: 96.9%
- **mAP@50-95**: 46.9%
- **Inference Time**: 12.8 ms per image

Qualitative results show accurate detection of stenosis regions with minimal false positives. When compared to other models, YOLOv8 outperformed Faster-RCNN Inception ResNet V2 and SSD MobileNet V2 in terms of speed and achieved competitive accuracy. Specifically, Faster-RCNN Inception ResNet V2 achieved an mAP of 95% but had a slow inference speed of 3 fps, while SSD MobileNet V2 was faster with 38 fps but had a lower accuracy of 83%. This highlights YOLOv8's effectiveness as a balanced solution for real-time and accurate detection.

---

## Future Scope

1. Extend the dataset with more diverse images to improve generalization, which is critical for the model to perform effectively across different patient populations and imaging conditions.
2. Integrate the system into clinical workflows as a real-time decision support tool, enabling faster diagnoses and aiding clinicians in making timely, data-driven decisions to improve patient outcomes.
3. Explore hybrid models to improve both precision and inference speed.

---
