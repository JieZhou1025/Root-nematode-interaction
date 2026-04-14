# Root-TransUNet: High-Precision Arabidopsis Root Segmentation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)

This is the official repository for **Root-TransUNet**, an optimised hybrid deep learning architecture designed for the high-precision segmentation of *Arabidopsis thaliana* root systems, specifically addressing the complex topological challenges during nematode infection.

---

## 📂 Repository Structure

| File / Folder | Description |
|---|---|
| `environment.yml` | Conda environment configuration (recommended) |
| `requirements.txt` | Pip environment configuration |
| `image_cropping_and_stitching.ipynb` | Preprocessing (crop large petri dish images into patches) and post-processing (stitch predicted patches back into a full image) |
| `Root-TransUNet_predict.ipynb` | Core inference script to generate segmentation masks for cropped image patches |

---

## 🛠️ Environment Setup

We provide two ways to set up the environment. **Conda is highly recommended** to automatically manage CUDA dependencies.

### Option 1: Using Conda (Recommended)

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
conda env create -f environment.yml
conda activate root-transunet  # Replace with the actual name in your yml if different
```

### Option 2: Using Pip

```bash
pip install -r requirements.txt
```

---

## 🚀 Inference Pipeline

Because the original petri dish scans are extremely high-resolution, feeding them directly into the model will cause memory overflow and loss of detail. Our pipeline follows a **Crop ➡️ Predict ➡️ Stitch** strategy.

### Step 1: Image Cropping (Preprocessing)

1. Open the **first section** of `image_cropping_and_stitching.ipynb`.
2. Define the input folder containing your original large petri dish images.
3. Run the cropping cells to generate $512 \times 512$ overlapping image patches.
4. The cropped patches will be saved in your designated patch folder.

### Step 2: Model Prediction

1. **Download Weights**: Download the pre-trained weights (`Root_TransUNet.pth`) from the [GitHub Releases](../../releases) page.
2. Place the downloaded `.pth` file into a `model_weights/` directory inside this project.
3. Open `Root-TransUNet_predict.ipynb`.
4. Point the input directory to the folder containing the cropped patches from Step 1.
5. Run the notebook to generate binary segmentation masks for each patch.

### Step 3: Image Stitching (Post-processing)

1. Return to `image_cropping_and_stitching.ipynb` and use the **Stitching** section.
2. Point the input folder to the directory containing the predicted masks from Step 2.
3. Run the stitching script. The script will automatically parse the filenames, reconstruct the coordinates, and merge the patches back into the full-scale petri dish segmentation map.

---

## 📖 Citation

If you find our model or code useful in your research, please consider citing our paper:

```bibtex
@article{RootTransUNet,
  title={need to fill in},
  author={need to fill in},
  journal={need to fill in},
  year={2026},
  publisher={need to fill in}
}
```
