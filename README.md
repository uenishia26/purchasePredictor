# SVM-Based Purchase Prediction Model

This project is a simple machine learning model created using Support Vector Machine (SVM) to predict whether a user will purchase an item or not based on their age and salary.

## Overview

The model uses the SVM algorithm with different kernel types to classify users into two categories:
- **1**: Purchased the item
- **0**: Did not purchase the item

## Features

The model takes into consideration the following features:
- **Age**: The age of the user.
- **Salary**: The estimated salary of the user.

## Model Performance

### Linear Kernel
- **Accuracy**: 90%
- **Kernel Type**: Linear

### RBF Kernel
- **Accuracy**: 93%
- **Kernel Type**: RBF (Radial Basis Function)

The RBF kernel showed a 3% improvement in accuracy compared to the linear kernel.

## Conclusion

This project demonstrates how different kernel types in SVM can affect the accuracy of the model. The linear kernel produced 90% accuracy, while the RBF kernel improved the accuracy by 3%, achieving 93%.
Feel free to explore and tweak the model further to see how it performs with your data!
