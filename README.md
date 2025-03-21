
# LGPS: Lightweight GAN-Based Polyp Segmentation

This repository contains the official implementation of **LGPS**, a lightweight GAN-based framework for polyp segmentation in colonoscopy images. LGPS achieves state-of-the-art performance with only **1.07 million parameters**, making it highly suitable for real-time clinical applications.

![LGPS_smallest_6](https://github.com/user-attachments/assets/a3676ae3-ffcd-4db3-864a-682e77bed462)


## Table of Contents
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Datasets](#datasets)
6. [Results](#results)
7. [Citation](#citation)
8. [License](#license)
9. [Contact](#contact)


## Introduction
Colorectal cancer (CRC) is a major global cause of cancer-related deaths, with early polyp detection and removal during colonoscopy being crucial for prevention. LGPS is a lightweight GAN-based framework designed to address challenges such as:
- High computational costs.
- Difficulty in segmenting small or low-contrast polyps.
- Limited generalizability across datasets.

LGPS incorporates:
- A **MobileNetV2 backbone** enhanced with modified residual blocks and Squeeze-and-Excitation (SE) modules.
- **Convolutional Conditional Random Fields (ConvCRF)** for precise boundary refinement.
- A **hybrid loss function** combining Binary Cross-Entropy, Weighted IoU Loss, and Dice Loss.

For more details, please refer to our [paper](#citation).

---

## Key Features
- **Lightweight Design:** Only **1.07 million parameters**, making it **17x smaller** than the smallest existing model.
- **State-of-the-Art Performance:** Achieves a **Dice coefficient of 0.7299** and an **IoU of 0.7867** on the challenging PolypGen dataset.
- **Robust Generalization:** Validated on five benchmark datasets, including Kvasir-SEG, CVC-ClinicDB, ETIS, CVC-300, and PolypGen.
- **Real-Time Applicability:** Suitable for deployment on resource-constrained devices.

---
## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Falmi/LGPS.git
2. Install dependencies:
   pip install -r requirements.txt
## Usage
1. Preprocess your dataset.
2. Train the model:
   python train.py
3. Evaluate the model:
  python evaluate.py
## Datasets
Kvasir-SEG
CVC-ClinicDB
## Citation
@article{tesema2024lgps,
  title={LGPS: A Lightweight GAN-Based Approach for Polyp Segmentation in Colonoscopy Images},
  author={Tesema, Fiseha B. and Manzanares, Alejandro Guerra and Author, Third C.},
  journal={IEEE Transactions on Medical Imaging},
  year={2024}
}
## License
