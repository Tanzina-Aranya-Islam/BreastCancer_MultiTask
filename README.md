# BreastCancer_MultiTask
Multi-task deep learning model (InceptionResNetV2 + ASPP) for breast cancer classification &amp; segmentation.
Breast Cancer Detection and Segmentation using InceptionResNetV2 + ASPP
Project Overview

This project implements a multi-task deep learning model for breast cancer classification and tumor segmentation from ultrasound images. The model performs simultaneous segmentation of tumor regions and classification into normal, benign, or malignant categories.

Dataset
Dataset used: Breast Ultrasound Images (BUSI) Dataset with Ground Truth

Structure:
Dataset_BUSI_with_GT/
├── normal/
│   ├── image1.png
│   ├── image1_mask.png
│   └── ...
├── benign/
│   ├── image1.png
│   ├── image1_mask.png
│   └── ...
└── malignant/
    ├── image1.png
    ├── image1_mask.png
    └── ...


Each class folder contains images and their corresponding binary masks.
Preprocessing included resizing to 256×256, median filtering, CLAHE, Gaussian smoothing, and normalization.

Split:
Train: 70%
Validation: 15%
Test: 15%
(Masks and images are saved in separate folders under each split for compatibility with the multi-task pipeline.)

Model Architecture
Backbone: InceptionResNetV2 (pretrained on ImageNet)
Segmentation Head: ASPP (Atrous Spatial Pyramid Pooling) + convolutional layers + resizing → output mask (1 channel, sigmoid)
Classification Head: GlobalAveragePooling + Dense layers → output softmax probabilities for 3 classes

Multi-task loss:
binary_crossentropy for segmentation
Class-weighted sparse_categorical_crossentropy for classification

Total model output:
segmentation_output → tumor mask
classification_output → normal / benign / malignant
Training

Hyperparameters:
Batch size: 8
Image size: 256×256
Epochs: 50
Optimizer: Adam (learning rate = 1e-4)
Loss weights: segmentation 0.4, classification 1.0

Callbacks:
ModelCheckpoint (monitor classification accuracy)
ReduceLROnPlateau
EarlyStopping

Class weights: computed from training labels to handle class imbalance.
Data Augmentation: random flips, rotations, applied to both images and masks.

Results
Test set metrics:
Class	Precision	Recall	F1-score	Support
normal	0.85	0.85	0.85	20
benign	0.89	0.85	0.87	66
malignant	0.74	0.81	0.77	31
Accuracy			0.84	117
Macro Avg	0.82	0.83	0.83	117
Weighted Avg	0.84	0.84	0.84	117

Segmentation accuracy: ~93–94% on validation/test sets.
Plots: Training/validation loss & accuracy curves, confusion matrix, ROC-AUC curves.

Usage
Clone the repository:
git clone <your-repo-url>
cd <repo-folder>
Install dependencies (tested with TensorFlow 2.x):
pip install tensorflow numpy pandas opencv-python scikit-learn matplotlib seaborn tqdm
Mount Google Drive or place dataset in Dataset_BUSI_with_GT/ folder.
Run preprocessing script to generate train/val/test splits:
dataset.ipynb


Train model and evaluate model and generate plots:
model.ipynb
