# Methyl-GP
This repository is the implementation of 'Methyl-GP: interpretable prediction models for DNA methylation prediction based on language model and representation learning'. We provide a complete files for readers to replicate our work. To reproduce our work, please follow the steps given below.
## 1. Environment setup
We recommend you to build a python virtual environment with [Anaconda](https://docs.anaconda.com/anaconda/install/). We conducted all experiments on 1 NVIDIA GeForce RTX 4090. If you use GPU with other specifications and memory sizes, consider adjusting your batch size accordingly.
### 1.1 Create and activate a new virtual environment
    conda create -n methylgp python=3.9
    conda activate methylgp
### 1.2 Install the package and other requirements
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

    git clone https://github.com/Hao010418/Methyl-GP
    cd Methyl-GP
    python -m pip install -r requirements.txt
## 2. Basic dictionaries
* The parameters can be modified in `config_init.py`. Generally, you only need to change the `learning rate (lr)`, `weight decay (wd)`, and `batch size (batch_size)` for replicating. 
* You should change the `pretrainpath` in `models/DNABERT.py` for different types of methylation. Here, we provide the fine-tuned DNABERT models at 
## 3. Get started
Once all preparations are complete, initiate the training by executing `train.py`.
## 4. Acknowledgement
Thanks for the authors of [Deep6mA](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008767), [BERT6mA](https://academic.oup.com/bib/article/23/2/bbac053/6539171?login=false), [iDNA-ABF](https://link.springer.com/article/10.1186/s13059-022-02780-1), and [DNABERT](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680?login=false). Their work has greatly facilitated our research, apreciate them!
