# Methyl-GP
This repository is the implementation of 'Methyl-GP: interpretable prediction models for DNA methylation prediction based on language model and representation learning'.
## 1. Environment setup
We recommend you to build a python virtual environment with [Anaconda](https://docs.anaconda.com/anaconda/install/). We conducted all experiments on 1 NVIDIA GeForce RTX 4090. If you use GPU with other specifications and memory sizes, consider adjusting your batch size accordingly.
## 1.1 Create and activate a new virtual environment
    conda create -n methylgp python=3.9
    conda activate methylgp
## 1.2 Install the package and other requirements
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

    git clone https://github.com/OotakK/Methyl-GP
    cd Methyl-GP
    python -m pip install -r requirements.txt
