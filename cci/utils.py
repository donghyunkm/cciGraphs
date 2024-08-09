from collections import defaultdict

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import toml
from sklearn import linear_model
from sklearn.dummy import DummyClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ligands = [
    ["Npy"],
    ["Sst"],
    ["Vip"],
    ["Tac2"],
    ["Cck"],
    ["Penk"],
    ["Crh"],
    ["Cort"],
    ["Tac1"],
    ["Pdyn"],
    ["Pthlh"],
    ["Pnoc"],
    ["Trh"],
    ["Grp"],
    ["Rln1"],
    ["Adcyap1"],
    ["Nts"],
    ["Nmb"],
]
receptors = [
    ["Npy1r", "Npy2r", "Npy5"],
    ["Sstr1", "Sstr2", "Sstr3", "Sstr4"],
    ["Vipr1", "Vipr2"],
    ["Tacr3"],
    ["Cckbr"],
    ["Oprd1", "Oprm1"],
    ["Crhr1", "Crhr2"],
    ["Sstr1", "Sstr2", "Sstr3", "Sstr4"],
    ["Tacr1"],
    ["Oprd1", "Oprk1", "Oprm1"],
    ["Pth1r"],
    ["Oprl1"],
    ["Trhr", "Trhr2"],
    ["Grpr"],
    ["Rxfp1", "Rxfp2", "Rxfp3"],
    ["Adcyap1r1", "Vipr1", "Vipr2"],
    ["Ntsr1", "Ntsr2"],
    ["Nmbr"],
]


def get_available_lr(adata):
    ligand_index = defaultdict(list)
    for i, ligand_category in enumerate(ligands):
        for ligand in ligand_category:
            series = adata.var.gene_symbol[adata.var.gene_symbol == ligand]
            if not series.empty:
                ligand_index[i].append(series.index[0])

    receptor_index = defaultdict(list)
    for i, receptor_category in enumerate(receptors):
        for receptor in receptor_category:
            series = adata.var.gene_symbol[adata.var.gene_symbol == receptor]
            if not series.empty:
                receptor_index[i].append(series.index[0])

    return ligand_index, receptor_index


def get_new_gene_subsets(adata, percents):
    gene_list = set(adata.var.index)
    gene_list_length = len(gene_list)
    subsets = []
    for percent in percents:
        subset = np.random.choice(list(gene_list), int(gene_list_length * percent), replace=False)
        gene_list = gene_list - set(subset)
        subsets.append(subset)

    sets = [subsets[0]]

    for i in range(len(subsets) - 1):
        sets_temp = np.hstack((sets[i], subsets[i + 1]))
        sets.append(sets_temp)
    set_return = []
    for i in range(len(sets)):
        set_return.append(list(sets[i]))
    return set_return


def get_adata(identifier):
    paths = toml.load("/allen/programs/celltypes/workgroups/mousecelltypes/Donghyun/cciGraphs/data/config.toml")
    adata = ad.read_h5ad(paths[identifier])
    adata.obsm["ccf"] = np.concatenate(
        (
            np.expand_dims(np.array(adata.obs["x_ccf"]), axis=1),
            np.expand_dims(np.array(adata.obs["y_ccf"]), axis=1),
            np.expand_dims(np.array(adata.obs["z_ccf"]), axis=1),
        ),
        axis=1,
    )
    adata.var.set_index("gene_symbol", inplace=True, drop=False)

    return adata
