# Facial Verification with Siamese Network

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)

A robust facial verification system built with a Siamese Neural Network architecture using TensorFlow. This system determines whether two facial images belong to the same person, providing a foundation for biometric authentication, access control systems, and identity verification applications.

## 📑 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Overview

This project implements a facial verification system using Siamese Neural Networks and contrastive learning. Rather than classifying faces into predetermined categories, this approach learns a similarity metric between pairs of images, making it suitable for verifying identities even for individuals not seen during training.

## Features

- **Siamese Network Implementation**: Built using TensorFlow's Functional API
- **Custom L1 Distance Layer**: Efficiently computes embedding similarity
- **Real-time Verification**: Live processing through webcam input using OpenCV
- **Preprocessing Pipeline**: Comprehensive image preprocessing and augmentation
- **Model Evaluation**: Performance metrics including accuracy, precision, and recall
- **Visualization Tools**: t-SNE visualization of face embeddings
- **Inference API**: Simple interface for integrating verification into other applications

## Architecture

The Siamese architecture consists of:

- **Twin CNN Networks** with shared weights that transform face images into embedding vectors
- **L1 Distance Layer** measuring the similarity between embeddings
- **Dense Classification Layer** with sigmoid activation for binary prediction

### Model Details:
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam (learning rate: 0.0001)
- **Embedding Dimension**: 128
- **Input Shape**: 100x100x3 (RGB images)

## Project Structure

```
Facial-Verification-Siamese-Network/
│
├── data/                              # Processed image data used for training
│   ├── anchor/                        # Anchor images (reference faces)
│   ├── positive/                      # Positive images (same identity)
│   └── negative/                      # Negative images (different identity)
│
├── application_data/                 # Real-time application related data
│   └── input_image/                  # Images captured live for verification
│
├── lfw_funneled/                     # Original LFW dataset after extraction
│   └── ...                           # All subfolders from LFW
│
├── model/                            # Saved Siamese model files
│   └── siamese_model.h5              # Trained model file
│
├── utils/                            # Utility scripts (optional: for preprocessing, model, etc.)
│   ├── preprocessing.py              # Image loading, resizing, labeling
│   ├── distance_layer.py             # Custom L1 Distance layer
│   └── dataset_generator.py          # Data loader and augmentation functions
│
├── Facial Verification with a Siamese Network - Final.ipynb  # Main Jupyter Notebook (training + inference)
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation

```

## Dataset

This project uses the **Labeled Faces in the Wild (LFW)** dataset:

- **Scale**: 13,233 images of 5,749 individuals
- **Diversity**: 1,680 subjects with two or more images
- **Characteristics**: Unconstrained, real-world images with variations in pose, lighting, expression
- **Format**: 250x250 pixel JPEG images
- **License**: Available for research purposes

### Data Preprocessing

Images undergo the following preprocessing steps:
1. Face detection and alignment
2. Resizing to 100x100 pixels
3. Normalization (pixel values scaled to [0,1])
4. Augmentation (random rotation, brightness adjustment, horizontal flipping)

Download the LFW dataset [here](https://www.kaggle.com/datasets/atulanandjha/lfwpeople).

### Getting Started

1. Clone the Repository
```
git clone https://github.com/yourusername/facial-verification-siamese.git
cd facial-verification-siamese
```

2. Install Dependencies
```
pip install -r requirements.txt
```

4. Download the Dataset
Download the LFW dataset from the link above and extract it into the root directory as lfw_funneled/.
```
Facial-Verification-Siamese-Network/
├── lfw_funneled/
│   └── (all dataset folders here)
```
5. Run the Notebook
Launch the Jupyter notebook and execute all cells:
```
jupyter notebook "Facial Verification with a Siamese Network - Final.ipynb"
```

This will:
- Preprocess the dataset
- Generate training pairs
- Train the Siamese network
- Save the model
- Enable real-time webcam-based facial verification

## Results

The model achieves:
- **Accuracy**: 96.2% on LFW test set
- **Precision**: 94.8%
- **Recall**: 95.1%
- **F1 Score**: 95.0%
- **Equal Error Rate (EER)**: 3.7%

## Future Improvements

- [ ] Implement triplet loss for more discriminative embeddings
- [ ] Add attention mechanisms to focus on key facial features
- [ ] Integrate with a more efficient face detector for faster processing
- [ ] Expand testing to more diverse datasets (e.g., racial diversity, different age groups)
- [ ] Deploy model to edge devices with TensorFlow Lite
- [ ] Add anti-spoofing mechanisms
