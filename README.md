# Wheat Leaf Disease Identification System
MNCM: Hybrid Attention MobileNet for Wheat Disease Identification
# Project Overview
This project constructs the lightweight MNCM deep learning model based on MobileNetV3-Large backbone for automatic classification of wheat leaf diseases. The proposed model embeds the Channel Prior Convolutional Attention (CPCA) module to optimize local feature extraction and eliminate background noise, as well as Multi-Head Self-Attention (MHA) module to capture global long-range dependencies of multi-scale features. Combined with layer-wise learning rate optimization strategy, MNCM achieves high-precision wheat disease identification with low computational cost and lightweight architecture, which can be deployed on edge devices for intelligent field monitoring and early warning of crop diseases. The dataset contains five leaf categories including healthy wheat, leaf rust, blight, powdery mildew and septoria, with a total of 7493 images.
# Project Structure
```
MNCM/
├── MNCM_best.pth              # Saved optimal model weight file after training
├── train.py                   # Main training script for MNCM
├── backbone.py                # Code of MobileNetV3-Large backbone network
├── CPCA.py                    # Implementation of Channel Prior Convolutional Attention module
├── MHA.py                     # Implementation of Multi-Head Self-Attention module
├── predict.py                 # Inference script for single/batch disease image prediction
└── dataset/
    ├── train/                 # Training dataset folder
    ├── val/                   # Validation dataset folder
    └── test/                  # Test dataset folder
```
# Core Technologies
1.Optimized MobileNetV3-Large Backbone: Constructed by depthwise separable convolution, retaining Hard-Swish activation and basic SE structure. The first nine layers of backbone are frozen for generic low-level features, and the remaining six layers are fine-tuned.
2.Hybrid Dual-Attention Fusion Mechanism
CPCA Attention: Realizes feature optimization from channel and spatial dimensions. Replaces fully-connected layers in SE with 1×1 convolution and adopts dilated convolution to extract local lesion texture, suppressing background interference such as soil reflection, weed shadow and uneven illumination via dual weighting.
MHA Attention: Selects multi-scale features C3 (shallow), C5 (middle) and C7 (high-level) from backbone. Features are fused with weighted upsampling to capture long-distance feature correlation and adapt to lesions of varying sizes.
3.Layer-wise Learning Rate & Layer Freezing Strategy: Pre-trained backbone uses small learning rate while newly added CPCA and MHA modules adopt higher learning rates. Cosine annealing learning rate scheduler is applied to accelerate convergence and prevent overfitting.
# Recommended Operating Environment
Windows 11
Python == 3.10.16
PyTorch == 2.7.0
IDE: Pycharm2025.1
RAM ≥ 32GB
# Dataset Introduction & Directory Structure
The dataset consists of field-collected raw images, Pix2Pix augmented samples and public LWDCD2020 dataset. Data split ratio: Train:Val:Test = 3:1:1 with total 7493 pictures.
```
dataset/
├── train/
│   ├── healthy/        # Healthy wheat leaves
│   ├── blight/         # Blight disease
│   ├── leaf_rust/      # Leaf rust disease
│   ├── powdery_mildew/# Powdery mildew disease
│   └── septoria/       # Septoria disease
├── val/
│   ├── healthy/
│   ├── blight/
│   ├── leaf_rust/
│   ├── powdery_mildew/
│   └── septoria/
├── test/
│   ├── healthy/
│   ├── blight/
│   ├── leaf_rust/
│   ├── powdery_mildew/
│   └── septoria/
```
# Model Training
Modify dataset path inside train.py, then run the following command:
```
python train.py
```
# Training Hyperparameter Configuration
The program automatically saves optimal checkpoint MNCM_best.pth. After training completion, comprehensive quantitative evaluation is performed on test set. The optimal parameter setup is listed below:
| Initial LR	| Total Epoch	| Batch Size	| Optimizer |	LR Decay Mode | 
|:0.001|:	100	|:16	|:SGD|:	Cosine Annealing
# Performance Evaluation
After training, the model outputs overall accuracy, per-class Precision, Recall, F1-score and Specificity. Confusion matrix, indicator correlation heatmap and radar chart are generated; Grad-CAM is adopted to visualize the model’s focus area on lesions.
The overall test accuracy of MNCM reaches 97.49%, outperforming OverLoCK-B (95.78%), ConvNeXt (93.93%) and MobileMamba (88.98%). Individual category accuracy: Blight 97.00%, Healthy 98.87%, Leaf rust 96.69%, Powdery mildew 97.60%, Septoria 96.53%.
# References and contact information
The paper is in the submission stage and will update the BiBTeX citation format after its official publication. Currently, it can be temporarily cited:
```
@article{mncm_wheat2026,
  title={MNCM: Hybrid attention MobileNet for wheat (Triticum aestivum L.) disease identification},
  author={Ning Li, Yangxuan Liu, Madineh Bijani, Longguo Wu, Xiaojie Du, Laixiang Xu},
  journal={Physiological and Molecular Plant Pathology},
  year={2026},
  note={Manuscript submitted for publication} 
  }
```
# Contact Information
If you encounter code running issues or academic exchange needs, please contact: 
Email: liuyangxuan@huuc.edu.cn
GitHub Repository: https://github.com/dyy603/xulaixiang-liuyangxuan-Wheat-Disease-Identification
Issues: Submit questions directly on GitHub page
