
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

## Requirements

- Python 3.10
- PyTorch 2.5.1
- CUDA-compatible GPU
- Required Python Libraries:
  - ultralytics (v8.2.103)
  - numpy
  - matplotlib

Install the dependencies using:

```bash
pip install -r requirements.txt
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-repo-name/coronary-stenosis-detection.git
```

2. Navigate to the project directory:

```bash
cd coronary-stenosis-detection
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Training

To train the model on your dataset:

```bash
!yolo task=detect mode=train model=yolov8s.pt data=path/to/data.yaml epochs=25 imgsz=800 plots=True
```

### Validation

To validate the model:

```bash
!yolo task=detect mode=val model=path/to/best.pt data=path/to/data.yaml
```

### Prediction

To run predictions on test images:

```bash
!yolo task=detect mode=predict model=path/to/best.pt conf=0.25 source=path/to/test/images save=True
```

---

## Visualizations

- **Confusion Matrix**: Located at `runs/detect/train/confusion_matrix.png`
- **Validation Results**: Located at `runs/detect/train/results.png`
- **Predictions**: Saved in the `runs/detect/predict` directory.

---

## Future Scope

1. Enhance the model for better performance at stricter IoU thresholds, as this would improve the system's ability to handle complex cases and ensure reliable detection in challenging scenarios.
2. Extend the dataset with more diverse images to improve generalization, which is critical for the model to perform effectively across different patient populations and imaging conditions.
3. Integrate the system into clinical workflows as a real-time decision support tool, enabling faster diagnoses and aiding clinicians in making timely, data-driven decisions to improve patient outcomes.
4. Explore hybrid models to improve both precision and inference speed.

---

## Acknowledgments

- Inspired by prior work on stenosis detection, including Faster-RCNN, SSD MobileNet V2, and StenUNet.
- Special thanks to KLE Technological University for their support and resources.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact

For any inquiries or collaboration opportunities, please contact:

- **Name**: P Sai Sudheer
- **Email**: [[your-email@example.com](mailto\:your-email@example.com)]
- **GitHub**: [https://github.com/your-profile](https://github.com/your-profile)
