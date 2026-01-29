# Cartoon Character Image Classification using CNN & Transfer Learning

---

## Project Overview
This project presents the development and evaluation of an image classification system for cartoon characters using Convolutional Neural Networks (CNNs). The work follows an iterative deep learning approach, beginning with a custom CNN built from scratch and later improving performance using transfer learning with MobileNetV2.

The project demonstrates a complete deep learning workflow:
- Dataset preparation and preprocessing
- Data augmentation
- CNN model design from scratch
- Model evaluation and error analysis
- Performance improvement using transfer learning
- Prediction on unseen images

---

## Objectives
- To implement a CNN from scratch for multi-class image classification
- To evaluate class-wise performance using detailed metrics
- To improve accuracy and generalization using transfer learning
- To compare the effectiveness of custom CNNs versus pretrained models

---

## Dataset Description
- Source: Cartoon Images Dataset.zip
- Total Images: 1292
- Images organized into class-specific folders
- Labels and image paths indexed using a Pandas DataFrame

### Cartoon Characters (10 Classes)
1. Mickey Mouse
2. Donald Duck
3. Naruto
4. Conan
5. Olaf
6. Mr. Bean
7. SpongeBob
8. Doraemon
9. Shinchan
10. Minions

Images were not tightly bounded. Instead, samples were selected such that the target character occupied most of the frame, introducing natural background variability and making the task more realistic.

---

## Data Preprocessing and Augmentation
- All images resized to 224 × 224 pixels
- Pixel values normalized to the [0, 1] range
- Stratified dataset split:
  - Training: 70%
  - Validation: 15%
  - Testing: 15%

### Data Augmentation (Training Set Only)
Applied using ImageDataGenerator to reduce overfitting:
- Rotation
- Width and height shifting
- Shear transformation
- Zooming
- Horizontal flipping

Validation and test sets were only rescaled to ensure unbiased evaluation.

---

## Data Generators
The flow_from_dataframe() method was used to create:
- Training generator (augmented and shuffled)
- Validation generator (rescaled only)
- Test generator (rescaled only, not shuffled)

This enabled efficient batch-wise loading and preprocessing of images.

---

## Model 1: Custom CNN (From Scratch)

### Architecture
- Two Conv2D + ReLU + MaxPooling2D blocks
- Flatten layer
- Dense hidden layer with 128 neurons (ReLU)
- Output layer with 10 neurons and Softmax activation

### Compilation and Training
- Optimizer: Adam
- Learning Rate: 0.001
- Loss Function: Categorical Crossentropy
- Metric: Accuracy
- Epochs: 10

### Performance
- Test Accuracy: 74.07%
- Test Loss: 1.2891

The model learned basic visual features but showed confusion between visually similar characters due to limited dataset size and absence of bounding boxes.

---

## Model 2: Transfer Learning with MobileNetV2

To improve performance, transfer learning was implemented using MobileNetV2 pretrained on the ImageNet dataset.

### Model Design
- MobileNetV2 convolutional base loaded without top layers
- All pretrained layers frozen
- Custom classification head added:
  - GlobalAveragePooling2D
  - Dense layer with 128 neurons (ReLU)
  - Dense output layer with 10 neurons (Softmax)

### Compilation and Training
- Optimizer: Adam
- Learning Rate: 0.0001
- Epochs: 10
- Only the classification head was trained

---

## Transfer Learning Model – Detailed Evaluation

### Overall Performance
- Accuracy: 92%
- Macro Average F1-score: 0.92
- Weighted Average F1-score: 0.92
- Total Test Samples: 189

### Class-wise Observations
- Excellent performance for Mr. Bean, Minions, Doraemon, and SpongeBob with near-perfect recall.
- Significant improvement for Mickey Mouse, Naruto, Shinchan, and Conan compared to the custom CNN.
- Donald Duck and Olaf showed relatively lower recall, likely due to visual similarity and limited samples.

These results confirm the effectiveness of pretrained feature extraction for small, multi-class image datasets.

---

## Pretrained Model Download
The trained MobileNetV2-based classification model & CNN model is available for download here:

https://drive.google.com/drive/folders/1OGIKpdx0At9YYiwVpp3-Dm2FsWLVS3KU?usp=sharing

You can download the model files and load them locally to run inference or extend the project for your own experiments.


## Model Comparison

| Model | Test Accuracy | F1-score |
|------|--------------|----------|
| Custom CNN (From Scratch) | 74.07% | ~0.74 |
| MobileNetV2 (Transfer Learning) | 92% | 0.92 |

---

## Prediction on New Images
Both models support prediction on unseen images:
1. Image resized to 224 × 224
2. Pixel values normalized
3. Image passed through the trained model
4. Output includes predicted class and confidence score

The MobileNetV2-based model provides more reliable and confident predictions.

---

## Conclusion
This project demonstrates the design, evaluation, and improvement of a cartoon character image classification system. While the custom CNN provided a strong foundation, the adoption of transfer learning with MobileNetV2 led to a substantial improvement in accuracy and class-level performance.

The project highlights:
- Limitations of training CNNs from scratch on small datasets
- Effectiveness of pretrained models for feature extraction
- Importance of detailed evaluation beyond overall accuracy

---

## Future Work
- Fine-tuning deeper layers of MobileNetV2
- Increasing dataset size for visually similar classes
- Applying regularization techniques such as Dropout and Batch Normalization
- Experimenting with other pretrained architectures like ResNet and EfficientNet

---

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib / Seaborn
