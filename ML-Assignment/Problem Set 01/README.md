# Problem Set 01: Chest X-Ray Image Classification using CNN

## Overview
This project classifies pediatric chest X-ray images into two categories: NORMAL and PNEUMONIA using a Convolutional Neural Network (CNN).

## Dataset Summary
- Total Images: 5,863 JPEG images
- Dataset Splits: Train (5,216), Validation (16), Test (624)
- Classes: NORMAL, PNEUMONIA

## Approach & Methodology
1. Preprocessing and normalization: Images are resized to 150x150 pixels and normalized to the [0, 1] range.
2. Handling class imbalance: Calculated class weights were applied during training to reduce the effect of class imbalance in the dataset.
3. Data augmentation: RandomFlip, RandomRotation(0.1), and RandomZoom(0.1) were used to improve generalization and reduce overfitting.
4. Model architecture: A sequential CNN with convolution and pooling blocks, a flattened layer, a dense layer with 128 units, dropout regularization, and a sigmoid output layer was implemented.

## Results
- The CNN was trained to classify chest X-ray images into NORMAL and PNEUMONIA categories.
- Training incorporated normalization, augmentation, and class weighting to improve robustness.
- Test-set evaluation was used to assess model performance.

## Limitations
- Each directory must include the code and a README.md file explaining the approach, methodology, and findings.
