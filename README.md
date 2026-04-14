# Workshop on Computer Vision for Image Segmentation

Author: Vladimir PIMONOV  
Contact: [foxlightmanstudes@gmail.com]  
Affiliation: [team PNEC / ILM]

This repository contains the teaching material, notebooks, helper scripts, and model definitions used in a workshop on computer vision for image segmentation.

The practical work is centered on a complete segmentation workflow for microscopy images:

- preparing masks from annotations,
- building and balancing a patch-based dataset,
- preparing PyTorch datasets and dataloaders,
- training U-Net family models,
- running inference on patches and full images,
- visualizing and post-processing predictions.

The notebooks are written so they can be followed during the workshop, but they are also intended to remain usable independently afterward.

---

## Repository content

The repository is organized around four practical notebooks and a set of reusable helper scripts. It also includes a `requirements.txt` file listing the required packages and their versions, as well as a `readme.txt` file with a practical guide for installing the environment needed for this workshop.

### Practical notebooks

- **TP01** – Dataset preparation  
  Extraction of masks from annotation files, patch description in a dataframe, balancing strategy, and export of the final image/mask patches.

- **TP02** – Dataframe preparation / PyTorch data pipeline  
  Definition of the custom dataset, transformations, augmentations, and dataloaders.

- **TP03** – U-Net training  
  Model selection, loss functions, optimizer and scheduler setup, training loop, validation, and checkpoint loading.

- **TP04** – Inference  
  Full-image tiled inference, patch inference, contour visualization, and simple confidence-based post-processing.

---

## Project structure

```text
repo_root/
│
├── TP01. Dataset for training 512 v4.1.ipynb
├── TP02. Dataframe preparation v4.0.ipynb
├── TP03. U-Net training v4.0.ipynb
├── TP04. Inference v4.0.ipynb
│
├── readme.txt
├── requirements.txt
├── README.md
│
└── scripts/
    ├── __init__.py
    ├── engine.py
    ├── inference.py
    ├── loss_functions.py
    ├── transforms.py
    │
    └── Models/
        ├── UNetResNet34.py
        ├── UNetResNetDeep.py
        ├── UNetResNetDeep_legacy.py
        ├── UNetPPResNet34.py
        ├── UNetPPResNetDeep.py
        ├── UNetPPResNetDeep_legacy.py
        ├── UNet3PlusResNet34.py
        └── UNet3PlusResNet50Deep.py