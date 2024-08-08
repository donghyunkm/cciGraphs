
## Methods to understand cell types through spatial peptidergic communication networks

Peptidergic communication networks in spatial transcriptomic data can be viewed as multilayer graphs. This repository explores graph embedding approaches to obtain embeddings for such graphs, and develops simulations to investigate the utility of such embeddings to understand the structure of spatial peptidergic networks.  


### Environment Setup

For PyG `2.5.1`:

```bash
conda create -n cci python=3.9

# order of installations is important: https://github.com/pyg-team/pytorch_geometric/discussions/7866#discussioncomment-7970609
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 cpuonly -c pytorch
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch_geometric

# other dependencies
pip install jupyterlab scikit-learn pandas matplotlib seaborn
pip install ipywidgets rich tqdm timebudget toml
pip install 'scanpy[leiden]'

# installs package from local directory for development.
# see https://github.com/rhngla/minpypack 
pip install -e .
```

For PyG `2.5.3`

```bash
conda create n cci python=3.12
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 -c pytorch
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
pip install torch_geometric
```