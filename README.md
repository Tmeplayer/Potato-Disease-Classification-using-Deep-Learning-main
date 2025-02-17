### **README for GitHub Repository**  

---

# **Plant Disease Classification Using CNN**  

This repository contains a **Convolutional Neural Network (CNN)** model for classifying plant diseases using the **PlantVillage dataset**. The model is built using **TensorFlow** and **Keras**, and it achieves high accuracy in detecting diseases such as **Potato Early Blight**, **Potato Late Blight**, and **Healthy Potato** leaves.  

---

## **Table of Contents**  
1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [Model Architecture](#model-architecture)  
6. [Results](#results)  
7. [Contributing](#contributing)  
8. [License](#license)  

---

## **Project Overview**  
The goal of this project is to build a deep learning model that can automatically classify plant diseases from images. The model is trained on the **PlantVillage dataset**, which contains labeled images of healthy and diseased plants. The trained model can predict the class of a new image (e.g., "Potato Late Blight") with high confidence.  

---

## **Dataset**  
The dataset used in this project is the **PlantVillage dataset**, which is publicly available. It contains images of various plants, including potatoes, tomatoes, and peppers, with labels for healthy and diseased conditions.  

- **Dataset Structure**:  
  ```
  dataset/PlantVillage/
  ├── Potato___Early_Blight/
  ├── Potato___Late_Blight/
  └── Potato___Healthy/
  ```  

- **Image Size**: All images are resized to **256x256 pixels**.  
- **Classes**: The model is trained to classify 3 classes (e.g., Early Blight, Late Blight, Healthy).  

---

## **Installation**  
To run this project, you need to install the required Python libraries.  

1. **Clone the repository**:  
   ```bash
   git clone https://github.com/your-username/plant-disease-classification.git
   cd plant-disease-classification
   ```  

2. **Install dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```  

   **Requirements**:  
   - TensorFlow  
   - NumPy  
   - Matplotlib  
   - Pillow (PIL)  

---

## **Usage**  

### **1. Training the Model**  
To train the model, run the following command:  
```bash
python train.py
```  

- The script will:  
  - Load the dataset.  
  - Preprocess the images (resize, normalize, augment).  
  - Train the CNN model for 35 epochs.  
  - Save the trained model as `model.h5`.  

### **2. Making Predictions**  
To classify a new image, use the `predict.py` script:  
```bash
python predict.py --image_path path/to/your/image.jpg
```  

- Example:  
  ```bash
  python predict.py --image_path test_images/potato_late_blight.jpg
  ```  

- Output:  
  ```
  Predicted: Potato___Late_Blight (95.34% confidence)
  ```  

---

## **Model Architecture**  
The CNN model consists of the following layers:  

1. **Preprocessing Layers**:  
   - Resizing (256x256)  
   - Rescaling (normalize pixel values to [0, 1])  
   - Data Augmentation (random flips and rotations)  

2. **Convolutional Base**:  
   - 2x Conv2D layers (32 and 64 filters)  
   - MaxPooling2D layers to reduce spatial dimensions  

3. **Classifier**:  
   - Flatten layer to convert 2D features to 1D  
   - Dense layer (64 neurons)  
   - Output layer (softmax activation for multi-class classification)  

---

## **Results**  
- **Training Accuracy**: ~98%  
- **Validation Accuracy**: ~95%  
- **Test Accuracy**: ~94%   

---

## **Contributing**  
Contributions are welcome! If you'd like to improve this project, please follow these steps:  

1. Fork the repository.  
2. Create a new branch (`git checkout -b feature/YourFeatureName`).  
3. Commit your changes (`git commit -m 'Add some feature'`).  
4. Push to the branch (`git push origin feature/YourFeatureName`).  
5. Open a pull request.  

---

## **License**  
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE.txt) file for details.  

---
---

## **requirements**  
This project is requirement some python lib . See the [requirements](requirements.txt) file for details.  

---

## **Acknowledgments**  
- **PlantVillage Dataset**: [https://plantvillage.psu.edu/](https://plantvillage.psu.edu/)  
- **TensorFlow**: [https://www.tensorflow.org/](https://www.tensorflow.org/)  
- **Keras**: [https://keras.io/](https://keras.io/)  

---
For any questions or feedback, please contact:  
Email: aa01102017878@gmail.com
GitHub: Ali 

--- 

