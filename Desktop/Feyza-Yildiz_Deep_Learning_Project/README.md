# Deep Learning Artwork Classification

This repository contains the code for a deep learning project that compares different neural network architectures for artwork style classification using a WikiArt-like dataset.

## Project Overview

The goal of this project is to classify paintings into different art styles using image classification models. A custom Convolutional Neural Network (CNN) is implemented from scratch and compared against pre-trained models from the `timm` library via transfer learning. The models are evaluated based on accuracy, F1-score, precision, recall, and model complexity (number of parameters).

## Main Features

- Artwork style classification using a curated image dataset.
- End-to-end pipeline implemented in a Jupyter/Colab notebook.
- Custom **SimpleCNN** model implemented from scratch.
- Support for multiple pre-trained models via `timm` (transfer learning).
- Comprehensive evaluation: accuracy, F1, precision, recall, confusion matrix.
- Training curves visualization (loss and accuracy).
- Grad-CAM visualizations to interpret model decisions.

## Technologies Used

- Python
- PyTorch, Torchvision, `timm`
- NumPy, Pandas
- scikit-learn
- Matplotlib, Seaborn
- OpenCV
- Grad-CAM

## Notebook Structure

The notebook is organized into the following main sections:

1. **Setup and Global Configuration**  
   Environment setup, imports, random seed configuration, and hyperparameters.

2. **Drive Mount and Paths**  
   Google Drive mount (for Colab) and path definitions for images, metadata, and splits.

3. **Dataset Loading and Preprocessing**  
   Dataset loading, image transformations, dataset and dataloader definitions.

4. **Model Definitions**  
   - `SimpleCNN`: a CNN model implemented from scratch.  
   - Additional models using pre-trained architectures from `timm` for transfer learning.

5. **Training and Validation**  
   Training loop, validation loop, metric calculation, and model checkpointing.

6. **Evaluation**  
   - Evaluation on the test set (accuracy, F1-score, precision, recall).  
   - Confusion matrix plotting and training curve visualization.

7. **Explainability with Grad-CAM**  
   Grad-CAM heatmaps to highlight which image regions the model focuses on for its predictions.

## How to Run

1. **Clone the repository or download the notebook**

   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
