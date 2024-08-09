
## Methods to understand cell types through spatial peptidergic communication networks

Peptidergic communication networks in spatial transcriptomic data can be viewed as multilayer graphs. This repository explores graph embedding approaches to obtain embeddings for such graphs, and develops simulations to investigate the utility of such embeddings to understand the structure of spatial peptidergic networks.  


### Environment Setup

For PyG `2.5.1`:

```bash
conda create -n cci python=3.11

# order of installations is important. See https://github.com/pyg-team/pytorch_geometric/discussions/7866#discussioncomment-7970609
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 cpuonly -c pytorch
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch_geometric

# installs package from local directory and other dependencies.
# see https://github.com/rhngla/minpypack 
pip install -e .
```


### Quarto

The main commands are to render and preview the documents. 

```bash
quarto render
quarto preview
```
