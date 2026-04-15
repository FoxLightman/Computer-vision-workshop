# Installation of the software environment for the computer vision workshop

This document describes the installation of the software environment for the computer vision workshop. The software environment is based on Python (version 3.10.9 required, other versions may work but have not been tested) and uses the following libraries:

- `PyTorch` (version 2.6 required to load legacy models)
- `torchvision`
- `matplotlib`
- `numpy`
- `scikit-image`
- `pandas`
- `scipy`
- `tifffile`
- `pillow`
- `psd-tools`
- `tqdm`
- `typing-extensions`
- `notebook`

It can be installed in three distinct ways, all of which make use of virtual environments. The three methods are:

1. [Using `uv` (recommended)](#using-uv-recommended)
2. [Using `venv` and `pip` (fallback)](#using-venv-and-pip-fallback)
3. [Using `conda`](#using-conda)

On Windows, it is recommended to use **PowerShell**.

## Using `uv` (recommended)

> This is the recommended method for installing the software environment. It uses `uv`, a tool for managing Python virtual environments and dependencies.

`uv` is a modern alternative to `pip` and `conda` that simplifies the management of Python projects. It allows you to create isolated environments and manage dependencies with ease.

To install `uv`, you can follow the instructions on the [official `uv` documentation](https://docs.astral.sh/uv/getting-started/installation).

Once you have `uv` installed, you just need to run the following command in the root directory of the project (where the `pyproject.toml` file is located):

```bash
uv sync
```

This command will create a virtual environment with the correct Python version and install all the dependencies specified in the `pyproject.toml` file. After running this command, you can activate the virtual environment and start working on the project.

Besides being easy to use and extremely fast, the main upside of this approach is that `uv` will automatically install the GPU version of PyTorch if a GPU is available, and will fall back to the CPU version if not, without needing to specify it manually.

To activate the virtual environment created by `uv`, you can use the following command:

```bash
source .venv/bin/activate
```

On Windows (PowerShell), use:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Using `venv` and `pip` (fallback)

> This method is not recommended as it requires you to have the correct version of Python installed on your system and to manually specify the correct architecture for PyTorch, which can be error-prone. However, it is a good fallback option if you do not want to use `uv`.

This method involves using the built-in `venv` module to create a virtual environment and `pip` to install the required packages.

To create a virtual environment using `venv` with the required Python version, you can run the following command:

```bash
python3.10 -m venv .venv
```

On Windows, use:

```powershell
py -3.10 -m venv .venv
```

This command will create a virtual environment in a directory named `.venv`. After creating the virtual environment, you need to activate it. You can do this with the following command:

```bash
source .venv/bin/activate
```

On Windows, use:

```powershell
.\.venv\Scripts\Activate.ps1
```

Once the virtual environment is activated, you can install the required packages using `pip`.

First, you need to determine the appropriate architecture for PyTorch based on your system. You can set the `ARCH` variable as follows:

- For CPU-only systems:

```bash
export ARCH=cpu
```

On Windows (PowerShell), use:

```powershell
$env:ARCH = "cpu"
```

- For NVIDIA GPU systems: identify the CUDA version (e.g., `cu118` for CUDA 11.8) by running `nvidia-smi` in the terminal, and then replace `XXX` in the following:

```bash
export ARCH=cuXXX
```

On Windows (PowerShell), use:

```powershell
$env:ARCH = "cuXXX"
```

- For AMD GPU systems: identify the ROCm version (e.g., `rocm5.4` for ROCm 5.4) by running `rocminfo` in the terminal, and then replace `Y.Y` in the following:

```bash
export ARCH=rocmY.Y
```

On Windows (PowerShell), use:

```powershell
$env:ARCH = "rocmY.Y"
```

You can finally run the following command to install all the dependencies:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/${ARCH}
```

On Windows (PowerShell), use:

```powershell
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/$env:ARCH
```

## Using `conda`

> This method is not recommended as it requires you to manually specify the correct architecture for PyTorch, which can be error-prone. It also requires a mixed installation with `pip` for some packages. Finally, it is less reliable than the other methods, as it may lead to dependency conflicts (typical of `conda` when trying to install packages in a specific version).

This method involves using `conda`, a popular package manager and environment management system, to create a virtual environment and install the required packages.

To create a virtual environment using `conda` with the required Python version, you can run the following command:

```bash
conda create -n computer-vision-workshop python=3.10.9
```

This command will create a virtual environment named `computer-vision-workshop` with Python version 3.10.9. After creating the virtual environment, you need to activate it. You can do this with the following command:

```bash
conda activate computer-vision-workshop
```

Once the virtual environment is activated, you can install the required packages using `conda` and `pip`.

You can then install the required packages using the following command:

```bash
conda install numpy=1.23.5 scipy=1.10.0 tifffile=2021.7.2 matplotlib=3.7.0 pandas=1.5.3 scikit-image=0.19.3 pillow=11.2.1 tqdm=4.64.1 notebook=6.5.4 "typing-extensions>=4.10" packaging=22.0 conda-forge::nccl
```

For `psd-tools`, you can install it using `pip`:

```bash
pip install psd-tools==1.9.34
```

For PyTorch, you need to determine the appropriate architecture for PyTorch based on your system. You can set the `ARCH` variable as follows:

- For CPU-only systems:

```bash
export ARCH=cpu
```

On Windows (PowerShell), use:

```powershell
$env:ARCH = "cpu"
```

- For NVIDIA GPU systems: identify the CUDA version (e.g., `cu118` for CUDA 11.8) by running `nvidia-smi` in the terminal, and then replace `XXX` in the following:

```bash
export ARCH=cuXXX
```

On Windows (PowerShell), use:

```powershell
$env:ARCH = "cuXXX"
```

- For AMD GPU systems: identify the ROCm version (e.g., `rocm5.4` for ROCm 5.4) by running `rocminfo` in the terminal, and then replace `Y.Y` in the following:

```bash
export ARCH=rocmY.Y
```

On Windows (PowerShell), use:

```powershell
$env:ARCH = "rocmY.Y"
```

You can run the following command to install all the dependencies:

```bash
pip install torch==2.6.0 torchvision==0.21.0 --extra-index-url https://download.pytorch.org/whl/${ARCH}
```

On Windows (PowerShell), use:

```powershell
pip install torch==2.6.0 torchvision==0.21.0 --extra-index-url https://download.pytorch.org/whl/$env:ARCH
```
