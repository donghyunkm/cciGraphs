from pathlib import Path
import warnings
import numpy as np
from torch_geometric.data import Data
import pickle
from scipy.sparse import csr_array
from sklearn.model_selection import StratifiedKFold
from torch_geometric.utils import from_scipy_sparse_matrix
from cci.utils import get_adata
from anndata import ImplicitModificationWarning

warnings.filterwarnings("ignore", category=ImplicitModificationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def dataset_application_poster(case=None):
    """This dataset is used to generate results to compare embeddings:
    1. Squashed LR graph: one Node2Vec. Identity of LR is lost.
    2. Independent LR graphs: concatenated Node2Vec embeddings.
    3. Joint LR graph: 1 embedding overall (MultilayerNode2Vec).
    4. Distance graph: 1 embedding overall (Node2Vec).
    The dataset is a single section from VISp with the largest number of cells.

    Args:
        case (str): "squashed", "independent", "joint" or "distance".
    """

    adata = get_adata("VISp")

    # print(adata.obs[["brain_section_label", "z_section"]].sort_values("z_section").value_counts().to_frame().head(4))
    one_sec = adata[adata.obs["z_section"] == 5.0, :]
    subclass_color_dict = one_sec.obs[["subclass_color", "subclass"]].drop_duplicates()
    subclass_color_dict = subclass_color_dict.set_index("subclass").to_dict()["subclass_color"]
    df = one_sec.obs.copy()

    # graph construction hyperparameters and graph properties
    graph_hparams = dict(
        case=case,
        d=40 / 1000,  # (in mm)
        L_thr=0.0,
        R_thr=0.0,
        lr_gene_pairs=[["Tac2", "Tacr3"], ["Penk", "Oprd1"], ["Pdyn", "Oprd1"], ["Pdyn", "Oprk1"], ["Grp", "Grpr"]],
    )
    graph_hparams["n_layers"] = len(graph_hparams["lr_gene_pairs"])
    graph_hparams["num_nodes"] = df.shape[0]

    edge_index_list = [None] * graph_hparams["n_layers"]
    A_csr = [None] * graph_hparams["n_layers"]
    df["participant"] = np.zeros(graph_hparams["num_nodes"], dtype=bool)

    # get distance matrix
    Dx = (df["x_reconstructed"].values.reshape(-1, 1) - df["x_reconstructed"].values.reshape(1, -1)) ** 2
    Dy = (df["y_reconstructed"].values.reshape(-1, 1) - df["y_reconstructed"].values.reshape(1, -1)) ** 2
    D = np.sqrt(Dx + Dy)
    del Dx, Dy

    # loop over ligand-receptor pairs that make up each layer of the multi-layer graph
    for i in range(graph_hparams["n_layers"]):
        ligand, receptor = graph_hparams["lr_gene_pairs"][i]
        df["L"] = one_sec[:, one_sec.var["gene_symbol"] == ligand].X.toarray().ravel()
        df["R"] = one_sec[:, one_sec.var["gene_symbol"] == receptor].X.toarray().ravel()

        df["L"] = (df["L"] > graph_hparams["L_thr"]).astype(bool)
        df["R"] = (df["R"] > graph_hparams["R_thr"]).astype(bool)

        A = df["L"].values.reshape(-1, 1) @ df["R"].values.reshape(1, -1)

        # cells are connected only if within distance d
        A[D > graph_hparams["d"]] = 0

        # participant should have more than one connection (this should not change across cases)
        df["participant"] = df["participant"] + (A.sum(axis=1) > 1)
        A_csr[i] = csr_array(A)

    if case == "distance":
        A = D <= graph_hparams["d"]
        graph_hparams["n_layers"] = 1
        edge_index_list = [None] * graph_hparams["n_layers"]
        edge_index_list[0], _ = from_scipy_sparse_matrix(csr_array(A))

    if case == "squashed":
        # edge values are only allowed to be 0 or 1, not the sum of all layers (as in a multigraph)
        A = np.zeros((graph_hparams["num_nodes"], graph_hparams["num_nodes"])).astype(bool)
        for i in range(graph_hparams["n_layers"]):
            A = A + A_csr[i].toarray()  # A_csr[i].toarray() is a boolean matrix

        # overwrite layer count only after individual graphs have been consumed.
        graph_hparams["n_layers"] = 1
        edge_index_list = [None] * graph_hparams["n_layers"]
        edge_index_list[0], _ = from_scipy_sparse_matrix(csr_array(A))

    elif case == "independent":
        for i in range(graph_hparams["n_layers"]):
            edge_index_list[i], _ = from_scipy_sparse_matrix(A_csr[i])

    elif case == "joint":
        for i in range(graph_hparams["n_layers"]):
            edge_index_list[i], _ = from_scipy_sparse_matrix(A_csr[i])
        graph_hparams["n_layers"] = 1

    elif case == "distance":
        graph_hparams["n_layers"] = 1
        edge_index_list = [None] * graph_hparams["n_layers"]
        edge_index_list[0], _ = from_scipy_sparse_matrix(csr_array(A))

    # get stratified splits based on subclass label
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    keep = df["participant"].values
    train_idx, test_idx = next(skf.split(np.arange(df[keep].shape[0]), df["subclass"][keep].values))

    labels = df["subclass"].cat.codes.values
    keep_idx = np.where(keep)[0]
    train_idx = keep_idx[train_idx]
    test_idx = keep_idx[test_idx]

    # this is simply a container for data; does not follow standard PyG graph patterns
    # this is okay because node2vec constructs random walks with a distinct mechanism.
    # random walk construction was also modified for the multilayer case.
    data = Data(x=None, edge_index_list=edge_index_list, labels=labels, train_idx=train_idx, test_idx=test_idx)
    return data, df, graph_hparams


def save_application_experiment(
    fname, epoch, z, df, graph_hparams, n2v_params=None, leiden_partitions=None, z_tsne=None
):
    path = Path("/allen/programs/celltypes/workgroups/mousecelltypes/Donghyun/cciGraphs/data/application/")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    savedict = {
        "z": z,
        "df": df,
        "graph_hparams": graph_hparams,
        "n2v_params": n2v_params,
        "leiden_partitions": leiden_partitions,
        "z_tsne": z_tsne,
    }
    with open(path / fname, "wb") as f:
        pickle.dump(savedict, f)
    return


def load_application_experiment(fname):
    path = Path("/allen/programs/celltypes/workgroups/mousecelltypes/Donghyun/cciGraphs/data/application/")
    with open(path / fname, "rb") as f:
        data = pickle.load(f)
    return data
