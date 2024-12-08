# LPRNet Optimization and Analysis

This repository contains the implementation and optimization of **LPRNet** (License Plate Recognition Network), a project conducted as part of the CSC591/791 coursework by Chirag Bheemaiah Palanganda Karumbaiah (cpalang@ncsu.edu) and Mukund Logasundar (mlogasu@ncsu.edu). The focus of this project was to evaluate and enhance the performance of LPRNet using techniques such as quantization, pruning, and Machine Learning Compiler (MLC) optimizations. The primary goal was to achieve an optimal balance of accuracy, inference speed, and model size, making the model efficient and suitable for deployment on resource-constrained devices.

## Table of Contents

1. [Introduction](#introduction)  
2. [Experiments and Results](#experiments-and-results)  
3. [Optimization Techniques](#optimization-techniques)  
   - Quantization  
   - Pruning  
   - MLC Optimizations  
4. [Setup and Installation](#setup-and-installation)  
5. [Usage](#usage)  
6. [Acknowledgments](#acknowledgments)  

---

## Introduction

LPRNet is a lightweight and efficient model designed for license plate recognition tasks. This project evaluates LPRNet's performance under various optimization strategies, including:

- **Quantization**: Using weights-only quantization and Post-Training Static Quantization (PTSQ).  
- **Pruning**: Applying structured and unstructured pruning strategies.  
- **MLC Optimizations**: Leveraging TVM for auto-tuning and kernel optimization.  

The experiments were conducted on **Google Colab**, leveraging its Intel Xeon CPU for benchmarking.

---

## Experiments and Results


### Models Summary

| Optimization Technique              | Accuracy (%) | Inference Speed (ms) | Model Size (KB) |
|-------------------------------------|--------------|-----------------------|-----------------|
| Base Model                          | 90.1         | 211.56               | 1816.73         |
| Quantization - Weights Only         | 89.9         | 43.14                | 533.57          |
| Quantization - Post-Training Static | 83.0         | 25.45                | 637.38          |
| Pruned Model I                      | 89.9         | 30.6                 | 1816.73*        |
| Pruned Model II                     | 89.1         | 31.25                | 1816.73*        |
| Pruned + PTSQ                       | 82.9         | 26.86                | 637.38          |
| MLC Optimized - Manual              | X            | X                    | -               |
| MLC Optimized - Auto                | 90.1         | 39.88                | 2905            |
| All Optimizations Combined          | 90.0         | 35.50                | 2905            |

> Note: For pruned models, the size reflects ONNX limitations and does not account for the structural pruning improvements.

---

## Optimization Techniques

### 1. Quantization  
Weights-only quantization reduced the model size and inference time while maintaining near-original accuracy. PTSQ further improved efficiency at the cost of a minor accuracy tradeoff.

### 2. Pruning  
Structured and unstructured pruning was applied using heuristics to prune less at lower feature levels, ensuring performance retention.

### 3. Machine Learning Compiler (MLC) Optimizations  
Auto-tuning via TVM optimized kernel operations, achieving significant speed-ups without sacrificing accuracy.

---

## Setup and Installation

### Prerequisites
- Python 3.8+
- PyTorch
- TVM
- ONNX
- Google Colab (recommended for testing)

# Validation

## Run Notebooks

To validate our work and observe the impact of various optimization techniques, run the following Jupyter notebooks. Each notebook focuses on a specific optimization approach or combines multiple techniques.

### 1. Weights-Only Quantization
- **Notebook**: `Weights_only_Quantization.ipynb`
- **Description**: Implements weights-only quantization to reduce model size and improve inference speed while maintaining acceptable accuracy.

### 2. Post-Training Static Quantization
- **Notebook**: `Post_Training_Static_Quantization.ipynb`
- **Description**: Applies post-training static quantization using PyTorch's quantization utilities to reduce memory footprint and increase computational efficiency.

### 3. Pruning
- **Notebook**: `PruningLPRNet.ipynb`
- **Description**: Demonstrates structured and unstructured pruning techniques to eliminate redundant parameters and reduce model complexity.

### 4. MLC Optimization
- **Notebook**: `MLC_LPRNet.ipynb`
- **Description**: Uses TVM's auto-tuning capabilities to generate optimized computation schedules tailored to the LPRNet architecture.

### 5. Combined Optimization
- **Notebook**: `Optimized_Model_Inference.ipynb`
- **Description**: Integrates quantization, pruning, and MLC optimization to achieve the best balance of accuracy, speed, and size in the final model.


