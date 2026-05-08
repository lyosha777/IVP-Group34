# IVP Group 34 – Handwritten Hindi Digit Recognition

## Overview

This project was developed for the **KEN3238 – Introduction to Image and Video Processing (IIVP)** course at Maastricht University.

The goal of the project is to classify handwritten Hindi digits (0–9) using a **Convolutional Neural Network (CNN)** implemented in **TensorFlow/Keras**.

The model is trained on grayscale digit images and generates predictions for unseen test images in a format compatible with the Kaggle competition submission system.

---

# Group Information

**Group 34**

- Mirzoyan Allen
- Alexander Šajánek

---

# Project Structure

```text
IVP-Group34-main/
│
├── data/
│   ├── train/
│   │   ├── 0/
│   │   ├── 1/
│   │   ├── ...
│   │   └── 9/
│   │
│   ├── test/
│   │   └── test images (.png)
│   │
│   ├── test.csv
│   └── sample_submission.csv
│
├── models/
│   └── hindi_digit_cnn.keras
│
├── src/
│   ├── train_cnn.py
│   └── sumbission_test.py
│
├── predictions.csv
├── README.md
└── .gitignore
```

---

# Folder Explanation

## `data/`

Contains all dataset-related files.

### `train/`
Contains the training dataset organized into folders by digit class:

```text
train/
├── 0/
├── 1/
├── ...
└── 9/
```

Each folder contains grayscale images representing the corresponding Hindi digit.

---

### `test/`

Contains unlabeled test images used for final prediction and Kaggle submission.

---

### `test.csv`

Contains the IDs of test images that must be predicted by the model.

---

### `sample_submission.csv`

Example format for the final Kaggle submission file.

---

## `models/`

Stores trained deep learning models.

### `hindi_digit_cnn.keras`

Saved CNN model after training.

---

## `src/`

Contains the Python source code used for training and prediction.

---

# Python Files Explanation

## `train_cnn.py`

Main training script for the CNN model.

### What this file does

- Loads the training dataset
- Splits data into training and validation sets
- Applies preprocessing
- Builds the CNN architecture
- Trains the model
- Saves the trained model

---

### Main Components

#### Dataset Loading

Uses:

```python
image_dataset_from_directory()
```

to automatically load images and labels from the folder structure.

---

#### Preprocessing

- Converts images to grayscale
- Resizes images to `32x32`
- Normalizes pixel values using:

```python
layers.Rescaling(1./255)
```

- Uses caching, shuffling, and prefetching for faster training

---

#### CNN Architecture

The model contains:

- 3 convolutional layers
- ReLU activation functions
- MaxPooling layers
- Dense fully connected layer
- Dropout regularization
- Softmax output layer

Architecture overview:

```text
Input Image
   ↓
Conv2D + ReLU
   ↓
MaxPooling
   ↓
Conv2D + ReLU
   ↓
MaxPooling
   ↓
Conv2D + ReLU
   ↓
Flatten
   ↓
Dense(128)
   ↓
Dropout(0.3)
   ↓
Softmax Output (10 classes)
```

---

#### Training

The model uses:

- Adam optimizer
- Categorical cross-entropy loss
- Accuracy metric
- Early stopping callback to reduce overfitting

---

#### Output

After training, the model is saved as:

```text
models/hindi_digit_cnn.keras
```

---

# `sumbission_test.py`

Prediction and submission generation script.

> Note: The filename contains a typo (`sumbission_test.py` instead of `submission_test.py`).

---

### What this file does

- Loads the trained CNN model
- Reads test image IDs
- Loads and preprocesses test images
- Predicts digit classes
- Generates a Kaggle submission CSV file

---

### Prediction Pipeline

For each image:

1. Open image using PIL
2. Convert image to grayscale
3. Resize to `32x32`
4. Convert image to NumPy array
5. Run CNN prediction
6. Select class with highest probability using:

```python
np.argmax()
```

---

### Output

Creates:

```text
predictions.csv
```

with the format:

```text
Id,Category
10001,4
10002,7
...
```

---

# Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- PIL (Python Imaging Library)

---

# How to Run the Project

## 1. Train the Model

```bash
python src/train_cnn.py
```

This trains the CNN and saves the model.

---

## 2. Generate Predictions

```bash
python src/sumbission_test.py
```

This creates:

```text
predictions.csv
```

for Kaggle submission.

---

# Methodology Summary

The project uses a deep learning CNN approach for image classification. CNNs are effective for handwritten digit recognition because they automatically learn visual patterns such as edges, curves, and shapes directly from image pixels.

The model was trained using grayscale Hindi digit images resized to `32x32` pixels. To improve generalization and reduce overfitting, dropout regularization and early stopping were applied during training.

---

# References

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-based learning applied to document recognition.*
- Krizhevsky, A., Sutskever, I., & Hinton, G. (2012). *ImageNet classification with deep convolutional neural networks.*
- Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.*
