======================Simple way========================

Install via pip the requirements from requirements.txt
 using following command in conda prompt or just python terminal

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126


===================Cleaner approach======================

Or create the separate environment in conda prompt by following the steps:

1) Copypaste next line and adjust the name of your environment (prefixed by -n)

conda create -n TEM_inference python=3.10.9

2) Activate the environment

conda activate ai_mask_env

3) Install the conda packages (WINDOWS!)

conda install numpy=1.23.5 scipy=1.10.0 tifffile=2021.7.2 matplotlib=3.7.0 pandas=1.5.3 scikit-image=0.19.3 pillow=11.2.1 tqdm=4.64.1 notebook=6.5.4 "typing-extensions>=4.10"

4) Install the psd-tools package to open GIMP and Photoshop files with .psd resolution

pip install psd-tools==1.9.34

5) Install pytorch to work with models

pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126