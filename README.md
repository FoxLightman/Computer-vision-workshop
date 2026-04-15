# Practical Workshop on Image Segmentation: Data Preparation, Model Training, and Inference

Author: Vladimir PIMONOV  
Contact: [foxlightmanstudes@gmail.com]  
Affiliation: [team PNEC / ILM]

This repository contains the material for a practical workshop on image segmentation with deep learning.

The repository covers the full workflow required to adapt a segmentation model to a microscopy dataset:

- extraction of masks from annotations,
- dataset preparation and balancing,
- patch generation,
- construction of PyTorch datasets and dataloaders,
- model training and validation,
- and inference on patches and full images.

The notebooks are intended both for guided use during the workshop and for independent use afterward as technical reference material.

---

## Repository content

The repository is organized around four practical notebooks and a set of reusable helper scripts.  
It also includes a `requirements.txt` file listing the required packages and their versions, as well as a `readme.txt` file with a practical guide for installing the environment needed for this workshop.

### TP01 – Dataset preparation

This notebook prepares the segmentation dataset from annotated microscopy images.

It includes:

- search for annotation files,
- extraction of masks from annotation layers,
- matching source images with their masks,
- definition of patching rules depending on magnification,
- description of candidate patches in a dataframe,
- characterization of patches as empty, almost empty, touching borders, or containing relevant objects,
- balancing of the future training set,
- export of the final image and mask patches.

The main purpose of this notebook is to construct a controlled training dataset before writing the cropped patches to disk.

### TP02 – Dataframe preparation and PyTorch data pipeline

This notebook prepares the exported dataset for model training.

It includes:

- definition of a custom PyTorch dataset,
- implementation of paired image and mask transformations,
- definition of augmentation rules,
- loading of the dataframe containing the train, validation, and test split,
- creation of datasets and dataloaders,
- visual inspection of augmentations and loaded samples.

The main purpose of this notebook is to connect the saved dataset to the PyTorch training pipeline.

### TP03 – Model training and validation

This notebook contains the training workflow.

It includes:

- model loading and configuration,
- preparation of train, validation, and test dataloaders,
- estimation of class imbalance,
- definition of losses, optimizers, and schedulers,
- training and validation loops,
- metric tracking,
- checkpoint saving and loading,
- first prediction checks on test patches and larger images.

The main purpose of this notebook is to train and evaluate a segmentation model on the prepared dataset.

### TP04 – Inference and post-processing

This notebook contains the inference workflow.

It includes:

- loading trained checkpoints,
- full-image tiled inference,
- patch inference,
- probability-map visualization,
- thresholded segmentation overlays,
- contour visualization,
- comparison of several confidence thresholds,
- confidence-based filtering of weak isolated detections.

The main purpose of this notebook is to apply a trained model to new images and inspect the predicted results.

---

## Available models

The repository contains several segmentation architectures from the U-Net family, implemented with ResNet backbones.

Available models include:

- **U-Net**
- **Deep U-Net**
- **U-Net++**
- **Deep U-Net++**
- **U-Net 3+**
- **Deep U-Net 3+**

Legacy versions are also included for compatibility and reference.

Two implementation families are available:

- **Classic models**, corresponding to the standard forms described in the literature,
- **Deep models**, which extend the depth of the network and increase the receptive field.

The deep variants require larger input patches because repeated downsampling reduces the spatial size of the feature maps more strongly.

The model definitions provided in this repository are intended for training and inference within the workshop workflow.  
The corresponding training data and fully trained model weights are available separately through the Zenodo repository:

[Zenodo repository link]

This repository therefore contains the code and workflow, while the dataset and trained checkpoints are distributed externally.

---

## Project structure

```text
repo_root/
│
├── TP01. Dataset for training 512 v4.1.ipynb   # dataset preparation and patch export
├── TP02. Dataframe preparation v4.0.ipynb      # PyTorch dataset and dataloader setup
├── TP03. U-Net training v4.0.ipynb             # model training and validation
├── TP04. Inference v4.0.ipynb                  # inference and post-processing
│
├── readme.txt                                  # practical environment installation guide
├── requirements.txt                            # required packages and versions
├── README.md                                   # repository description
│
├── Models/                                     # trained checkpoints; distributed via Zenodo
│   └── ...
│
├── Aggregates for training TP/                 # example data; distributed via Zenodo
│   └── ...
│
└── scripts/
    ├── __init__.py
    ├── engine.py                               # training and evaluation loops
    ├── inference.py                            # patch and tiled inference functions
    ├── loss_functions.py                       # loss functions for segmentation
    ├── transforms.py                           # dataset transforms and augmentations
    │
    └── Models/                                 # model definitions used by the notebooks
        ├── UNetResNet34.py                     # Classic U-Net; weights available via Zenodo
        ├── UNetResNetDeep.py                   # Deep implementation of U-Net
        ├── UNetResNetDeep_legacy.py            # Legacy Deep U-Net; weights available via Zenodo
        ├── UNetPPResNet34.py                   # Classic U-Net++; weights available via Zenodo
        ├── UNetPPResNetDeep.py                 # Deep implementations of U-Net++
        ├── UNetPPResNetDeep_legacy.py          # Legacy Deep U-Net++; weights available via Zenodo
        ├── UNet3PlusResNet34.py                # Classic U-Net3+; weights available via Zenodo 
        └── UNet3PlusResNet50Deep.py            # Deep implementation of U-Net3+; weights available via Zenodo