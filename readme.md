# Distorted Visual Sequence Pattern Recognition

A deep learning solution for recognizing distorted alphanumeric sequences from grayscale images using a CRNN (CNN + BiLSTM + CTC) architecture.

## Overview

This project addresses the problem of optical character recognition (OCR) on heavily distorted image sequences. The model predicts a sequence of characters directly from an input image without requiring character-level segmentation.

The solution uses:

* CNN-based feature extraction
* Bidirectional LSTM sequence modeling
* Connectionist Temporal Classification (CTC) Loss
* Beam Search decoding
* Test-Time Augmentation (TTA)

## Dataset

* Training Images: 20,000
* Test Images: 5,000
* Image Size: 200 × 100 pixels
* Character Vocabulary: 31 alphanumeric characters

During dataset inspection, two corrupted labels were identified and removed from training to maintain a clean vocabulary.

## Model Architecture

The recognition pipeline follows a CRNN architecture:

Input Image
→ CNN Backbone
→ Sequence Features
→ BiLSTM Layers
→ Linear Projection
→ CTC Decoder

### Components

* Multi-layer CNN feature extractor
* 2-layer Bidirectional LSTM
* Linear classification layer
* CTC Loss for alignment-free sequence learning

## Training

### Preprocessing

* Grayscale image loading
* CLAHE contrast enhancement
* Resize to 64 × 256
* Pixel normalization

### Data Augmentation

* Rotation
* Perspective distortion
* Affine transformation
* Gaussian noise
* Gaussian blur
* Brightness/contrast adjustment

### Optimization

* AdamW Optimizer
* Mixed Precision Training (AMP)
* ReduceLROnPlateau Scheduler
* Gradient Clipping

## Results

The model converged successfully and achieved a very low Character Error Rate (CER) on the validation set.

Best Validation CER:

```text
0.0005
```

## Project Structure

```text
.
├── dataset.py
├── model.py
├── train.py
├── inference.py
├── utils.py
├── requirements.txt
├── submission.csv
└── README.md
```

## Running Training

```bash
python train.py
```

## Running Inference

```bash
python inference.py
```

## Output

The inference pipeline generates:

```text
submission.csv
```

containing predictions for all test images.

## Author

Lakshya Kumar

Mechanical Engineering

IIT Roorkee
