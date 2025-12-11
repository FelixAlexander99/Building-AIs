# Building AIs.
California Housing Price Predictor (PyTorch).
An end-to-end Deep Learning regression model built from scratch using PyTorch. This project implements a full production-ready training pipeline, moving from raw CSV data processing to model inference, designed to predict housing prices based on the standard California Housing dataset.

--- Project Overview: 
* The goal of this project was to move beyond high-level APIs and understand the low-level mechanics of training neural networks. It handles real-world data challenges including categorical encoding, feature scaling, and hardware optimization.

--- Tech Stack & Concepts: 
* Core: Python 3.10+, PyTorch (Neural Networks, Autograd)
* Data Processing: Pandas, NumPy (Z-score normalization, One-Hot Encoding)
* Training: Custom Training Loop, Validation Splits, Checkpointing
* Hardware: CUDA (GPU) acceleration with Windows-specific multiprocessing optimizations.

--- Key Features: 
* Custom Dataset Class: Implements a robust torch.utils.data.Dataset wrapper that handles raw CSV loading, cleaning (handling NaNs), and feature engineering on the fly.
* Optimized Data Loading: Utilizes DataLoader with parallel workers (num_workers) and memory pinning to maximize GPU throughput, tuned specifically to avoid Windows multiprocessing bottlenecks.
* Dynamic Architecture: A Multi-Layer Perceptron (MLP) that automatically adjusts its input layer based on the dataset's feature count.
* Activation: implemented LeakyReLU to prevent "dead neuron" issues.
* Optimization: Uses Adam optimizer for faster convergence compared to standard SGD.
* Robust Training Pipeline:
* 80/20 Train/Validation split to monitor overfitting.
* Automatic model checkpointing (saves the model only when Validation Loss improves).
* Real-time loss logging.

--- Methodology: 
* Data Preprocessing:
* Dropped irrelevant features (configurable).
* Applied One-Hot Encoding to the ocean_proximity categorical feature.
* Applied Z-Score Normalization to numerical features to ensure stable gradient descent.
* Scaled target values to prevent gradient explosion.
  
--- Model Architecture:
* Input Layer -> Hidden Layer (64 neurons) -> LeakyReLU -> Hidden Layer (32 neurons) -> LeakyReLU -> Output.
  
--- Performance:
* Achieved significant convergence by resolving broadcasting shape mismatches in the loss calculation.
* Loss function: Mean Squared Error (MSE).

--- How to Run: 
* Clone the repository:
* code
* Bash
* git clone https://github.com/yourusername/housing-price-predictor.git
* cd housing-price-predictor
Install dependencies:
* code
* Bash
* pip install torch pandas numpy
Run the training script:
* code
* Bash
* python main.py
The script will automatically detect CUDA/CPU, train the model, save the best weights to best_house_model.pth, and perform a sample inference test at the end.

--- Key Learnings: 
* Broadcasting Logic: Addressed critical bugs regarding tensor shapes (e.g., [Batch, 1] vs [Batch]) that initially caused the model to converge on the dataset mean.
* Windows Multiprocessing: Implemented if __name__ == '__main__': guards to safely enable multi-threaded data loading on Windows environments.
* Feature Importance: Observed the massive impact of including vs. dropping geospatial data (Latitude/Longitude) on model accuracy.
