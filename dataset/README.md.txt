# Breast Cancer Ultrasound Dataset (Processed for Multi-task Model)

This repository contains the **processed BUSI dataset** (Breast Ultrasound Images) for multi-task deep learning: **tumor classification** and **tumor segmentation**.  

> **Note:** The original BUSI dataset is available on Kaggle and is **not included here** due to licensing. This README explains how to obtain, preprocess, and organize the dataset for this project.

---

## 1. Dataset Source

- **Name:** Breast Ultrasound Images Dataset (BUSI)  
- **Author:** Saba Hesaraki  
- **Kaggle Link:** [BUSI Dataset](https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset)  
- **Classes:** `normal`, `benign`, `malignant`  
- Each image may include a corresponding mask for tumor region (ground truth segmentation).

---

## 2. Folder Structure (Required)

After preprocessing, the dataset should be organized as follows:

Processed/
├── train/
│ ├── normal/images
│ ├── normal/masks
│ ├── benign/images
│ ├── benign/masks
│ └── malignant/images
│ └── masks
├── val/
│ ├── normal/images
│ ├── normal/masks
│ ├── benign/images
│ ├── benign/masks
│ └── malignant/images
│ └── masks
└── test/
├── normal/images
├── normal/masks
├── benign/images
├── benign/masks
└── malignant/images
└── masks


> Each mask should be a **binary PNG image** (`0` = background, `1` = tumor).

---

## 3. Preprocessing Steps

For each image and mask:

1. **Resize** to 256×256 pixels.  
2. **Image preprocessing:**  
   - Median filter to reduce speckle noise  
   - CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement  
   - Gaussian smoothing  
   - Normalize pixel values to 0–1  
3. **Mask preprocessing:**  
   - Resize to 256×256  
   - Convert to **binary mask** (0/1)  
4. **Handle multiple masks:** If an image has more than one mask, merge them using a **pixel-wise maximum** to create a single mask per image.

---

## 4. How to Prepare the Dataset

1. Download the BUSI dataset from Kaggle.  
2. Organize images and masks in folders by class (`normal`, `benign`, `malignant`).  
3. Run the preprocessing script (`dataset_preprocess.py`) to generate the **Processed** folder in the structure above.  

> The repo includes a script to split the dataset into **train/validation/test** (70/15/15) while preserving class balance.

---

## 5. Notes

- **Patient privacy:** Only sample images or processed data should be included in the repository; full original data should be obtained from Kaggle.  
- **Compatibility:** This processed dataset is ready to use with the multi-task InceptionResNetV2 + ASPP model provided in this project.  

---

## 6. References

Saba Hesaraki, “Breast Ultrasound Images Dataset (BUSI),” Kaggle, 2018. [Link](https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset)
