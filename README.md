# Code for: Modeling the inverse MEG problem in neuro-imaging using Physics Informed Neural Networks

This repository contains the source code and reproduction scripts for the paper "Modeling the inverse MEG problem in neuro-imaging using Physics Informed Neural Networks".

## Repository Contents

*   **`fenics_forward_poisson.py`**: Finite Element Method (FEM) forward solver implementing the Poisson equation for electrostatics in the brain. This script generates the ground-truth synthetic data used for training and validation.
*   **`benchmark_pinns_only_v25.ipynb`**: The main Jupyter Notebook for the Physics-Informed Neural Network (PINN). It handles:
    *   Loading the dataset.
    *   Defining the PINN architecture (Hybrid model with Maxwell's equations).
    *   Training the model.
    *   Evaluating performance (localization error).
*   **`optimize_mne_inverse.py`**: Implementation of the Minimum Norm Estimation (MNE) inverse solver with depth weighting and noise regularization.
*   **`benchmark_mne_batch.py`**: Batch processing script that runs the MNE solver on the validation dataset (N=1000) to generate the comparative statistics and error histogram presented in the paper.
*   **`COLAB_WORKFLOW_GUIDE.md`**: Instructions for running these scripts in Google Colab (if applicable).

## Requirements

*   Python 3.8+
*   FEniCS (for forward modeling)
*   PyTorch (for PINN training)
*   MNE-Python
*   NumPy, SciPy, Matplotlib

## Reproduction Steps

1.  **Generate Data**: Run `fenics_forward_poisson.py` to create the synthetic dataset (or use the provided `.npz` files if available).
2.  **Train PINN**: Execute `benchmark_pinns_only_v25.ipynb` to train the neural network and assess its localization accuracy.
3.  **Run Benchmark**: Execute `benchmark_mne_batch.py` to calculate the MNE baseline performance.
    ```bash
    python benchmark_mne_batch.py
    ```
4.  **Compare**: The script will output the mean error and generate the `mne_validation_histogram.png` plot.
