
# LGPS: Lightweight GAN-Based Polyp Segmentation

This repository contains the official implementation of **LGPS**, a lightweight GAN-based framework for polyp segmentation in colonoscopy images. LGPS achieves state-of-the-art performance with only **1.07 million parameters**, making it highly suitable for real-time clinical applications.

![model for github](https://github.com/user-attachments/assets/7eccf8fa-4c3c-475e-a0c3-c097fa4ba9f9)



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
### Prerequisites
### steps
1. Clone this repository:
   ```bash
   git clone https://github.com/Falmi/LGPS.git
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
## Datasets
- Kvasir-SEG: [Download](https://datasets.simula.no/kvasir-seg/)
- CVC-ClinicDB: [Download](https://polyp.grand-challenge.org/CVCClinicDB/)
- ETIS: Download [Download](https://polyp.grand-challenge.org/ETISLarib/)
- CVC-300: [Download](http://pages.cvc.uab.es/CVC-Colon/)
- PolypGen: [Download](https://drive.google.com/drive/folders/16uL9n84SrMt7IiQFzTUQNaJ9TbHJ8DhW)

Place the datasets in the data/ directory.
## Usage
1. Preprocess your dataset.
   ```bash
   cd data
   python Preprocess_SEG.py 
   python Preprocess_CVC_CliniCDB.py 
2. Train the model:
   ```bash
   python train.py
4. Evaluate the model:download the pretrained model from [here](https://drive.google.com/uc?export=download&id=1HI42ASPDcfjW5mNvDlQuLBzjoKVK3DYE)
   ```bash
   python Test.py --data_path "data/CVC-ClinicDB" --model_path "XXX.h5"
## Results
![image](https://github.com/user-attachments/assets/596362f3-38ed-4aa3-ba6f-a6f0c840cb2f)

## Citation
- @misc{tesema2025lgpslightweightganbasedapproach,
      title={LGPS: A Lightweight GAN-Based Approach for Polyp Segmentation in Colonoscopy Images}, 
      author={Fiseha B. Tesema and Alejandro Guerra Manzanares and Tianxiang Cui and Qian Zhang and Moses Solomon and Sean He},
      year={2025},
      eprint={2503.18294},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.18294}, 
}
- Submited to IEEE Transactions on Medical Imaging.
## License
