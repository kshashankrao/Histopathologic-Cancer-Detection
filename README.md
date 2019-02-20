# Histopathologic-Cancer-Detection
Detection of meta state cancer in small image patches taken from larger digital pathology scans. 
This competition is hosted on Kaggle.

This repository contains algorithm to train and test a convolution neural network to classify cancer 
cells with the dataset hosted on Kaggle for Histopathologic Cancer detection.

1) In net.py you can find the neural network architecture.
2) Also in main.py you can modify the code to finetune Vgg16 network.

Trained for 10 Epochs
Train Accuracy: 92%
Test Accuracy: 90%

Usage:
1) Change the directory name accordingly in main.py and test.py
2) Train script: python main.py 
3) Test script: python test.py
