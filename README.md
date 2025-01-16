# Traffic Sign Recognition using Computer Vision and CNN

## Introduction

### What is Computer Vision?

Computer Vision is a field of artificial intelligence that enables computers to interpret and process visual data from the world. By mimicking the human vision system, computer vision enables applications such as object detection, image recognition, and video analysis. In the context of self-driving cars, computer vision plays a pivotal role in understanding the environment by identifying traffic signs, pedestrians, road lanes, and other vehicles.

### What is CNN?

Convolutional Neural Networks (CNNs) are a class of deep learning algorithms designed for analyzing visual data. They consist of convolutional layers that extract spatial features from images, pooling layers to reduce dimensionality, and fully connected layers for classification. CNNs are highly effective in recognizing patterns in images, making them the backbone of many computer vision applications.

### Why Traffic Sign Recognition?

In the world of self-driving cars, recognizing traffic signs is critical for ensuring road safety and compliance with traffic laws. Traffic sign recognition systems provide vehicles with the ability to:

- Interpret speed limits.
- Recognize stop signs and pedestrian crossings.
- Detect warnings such as sharp curves or slippery roads.

This project demonstrates how CNNs and computer vision techniques can be used to build a robust traffic sign recognition system. With additional integration of technologies like YOLO (You Only Look Once) for real-time object detection, the system can be extended to enhance its real-world application.

---

## Features

- **Traffic Sign Detection:** Recognizes 43 classes of traffic signs with a accuracy of ----\*\*\*.
- **Preprocessing:** Includes grayscaling, histogram equalization, and normalization for consistent input.
- **Data Augmentation:** Uses transformations like rotation, zooming, and shifting to enhance training data.
- **Web Deployment:** Serves the trained model via a Flask-based web application for image uploads and predictions.



---

## Installation

### Prerequisites

- Python 3.7 or higher
- TensorFlow
- Keras
- Flask
- OpenCV
- Pandas
- NumPy
- Matplotlib

---

## Usage

### Training the Model

1.Dataset is in the `Dataset/` directory. The structure matches the following format (43 subfolders, each corresponding to a class).
2. Run `main.py` to train the model:
   ```bash
   python main.py
   ```
   - Outputs: A trained model saved as `model.keras`.

### Running the Web Application

1. Launch the Flask application:
   ```bash
   python app.py
   ```
2. Open in browser and go to `http://127.0.0.1:5000/`.
3. Upload an image of a traffic sign to receive the predicted class.

---

## Model Architecture

The CNN used for this project consists of:

- **Convolutional Layers:** Extract spatial features using filters.
- **Pooling Layers:** Reduce feature map dimensionality to prevent overfitting.
- **Dropout Layers:** Prevent overfitting by randomly deactivating neurons during training.
- **Fully Connected Layers:** Perform classification.

---

## Future Enhancements

1. **Integration with YOLO:** Real-time detection of traffic signs in video feeds.
2. **Dynamic Dataset Expansion:** Use YOLO to collect and label new traffic signs for improving the model.
3. **Deployment on Edge Devices:** Optimize the model for deployment on devices like Raspberry Pi for real-world usage.
4. **Multilingual Support:** Recognize traffic signs in various languages and regions.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements.

---

## Acknowledgments

- Kaggle for providing traffic sign datasets.
- TensorFlow and Keras for deep learning frameworks.
- OpenCV for image processing tools.


