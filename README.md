# Root-TransUNet: High-Precision Arabidopsis Root Segmentation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)

This is the official repository for **Root-TransUNet**, an optimised hybrid deep learning architecture designed for the high-precision segmentation of *Arabidopsis thaliana* root systems, specifically addressing the complex topological challenges during nematode infection.

## 📂 Repository Structure

* `environment.yml` & `requirements.txt`: Environment configuration files.
* `image_cropping_and_stitching.ipynb`: Script for preprocessing (cropping large petri dish images into patches) and post-processing (stitching predicted patches back into a full image).
* `Root-TransUNet_predict.ipynb`: Core inference script to generate segmentation masks for the cropped image patches.

---

## 🛠️ Environment Setup

We provide two ways to set up the environment. **Conda is highly recommended** to automatically manage CUDA dependencies.

**Option 1: Using Conda (Recommended)**
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
conda env create -f environment.yml
conda activate root-transunet  # Replace with the actual name in your yml if different
