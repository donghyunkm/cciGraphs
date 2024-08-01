
## Methods to understand cell types through spatial peptidergic communication networks

Peptidergic communication networks in spatial transcriptomic data can be viewed as multilayer graphs. This repository explores graph embedding approaches to obtain embeddings for such graphs, and develops simulations to investigate the utility of such embeddings to understand the structure of spatial peptidergic networks.  


### Environment Setup
```
conda create -n cci python=3.9
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install jupyterlab scikit-learn seaborn
conda install pyg -c pyg
conda install -c conda-forge scanpy python-igraph leidenalg

pip install -e . 
# Minimal python packaging for local development 
# https://github.com/rhngla/minpypack 

```