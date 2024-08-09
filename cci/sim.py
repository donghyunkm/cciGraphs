import numpy as np
import pandas as pd


class Interactions:
    """Class to generate adjacency matrices for testing."""

    def __init__(self, n_labels=8, n_blocks_per_graph=2, n_nodes_per_label=10, n_channels=3, labels=None, seed=0):
        self.n_labels = n_labels
        self.n_blocks_per_graph = n_blocks_per_graph
        self.n_nodes_per_label = n_nodes_per_label
        self.n_channels = n_channels
        self.labels = labels
        self.labels = self.set_labels()
        self.seed = seed

    def set_labels(self):
        """Generates a list of string labels.

        Returns:
            labels: list of labels
        """
        if self.labels is None:
            # small letters starting from 'a'
            labels = np.array([chr(i) for i in range(97, 97 + self.n_labels)])
        else:
            labels = self.labels
        return labels

    def get_block_members(self):
        """Generates a dictionary of block members for each block in the graph.

        Returns:
            block_members: dict with a list of labels in each key.
        """
        np.random.seed(self.seed)
        self.seed = self.seed + 1
        n = self.n_labels // self.n_blocks_per_graph
        block_members = {}
        remaining_label_set = set(self.labels)
        for b in range(self.n_blocks_per_graph):
            if b == 0:
                block_members[b] = np.random.choice(list(remaining_label_set), n, replace=False)
            elif b < self.n_blocks_per_graph - 1:
                block_members[b] = np.random.choice(list(remaining_label_set), n, replace=False)
            else:
                block_members[b] = list(remaining_label_set)

            if len(remaining_label_set) > 0:
                remaining_label_set = remaining_label_set - set(block_members[b])

        return {k: sorted(list(v)) for k, v in block_members.items()}

    def make_adj(self, block_members=None):
        """Generates an adjacency matrix for a given block structure (label x label).

        Args:
            block_members: dict with a list of labels in each key.

        Returns:
            block_members: dict with a list of labels in each key.
            adj_df: adjacency matrix
        """
        if not block_members:
            block_members = self.get_block_members()

        adj_df = pd.DataFrame(np.zeros((self.n_labels, self.n_labels)), index=self.labels, columns=self.labels)
        for b in block_members:
            for i in range(len(block_members[b])):
                for j in range(i + 1, len(block_members[b])):
                    adj_df.loc[block_members[b][i], block_members[b][j]] = 1
                    adj_df.loc[block_members[b][j], block_members[b][i]] = 1

        return block_members, adj_df

    def get_adj_per_channel(self):
        """Generates adjacency matrices (label x label) for each channel.

        Returns:
            blocks_per_channel: dict with a list of labels in each key.
            adj_per_channel: dict with adjacency matrices
        """
        blocks_per_channel = {}
        adj_per_channel = {}
        for c in range(self.n_channels):
            block_members, A_df = self.make_adj()
            blocks_per_channel[c] = block_members
            adj_per_channel[c] = A_df
        return blocks_per_channel, adj_per_channel

    def get_node_labels(self):
        """Generates a list of node labels.

        Returns:
            node_labels: list of node labels
        """
        node_labels = []
        for l_ in range(self.n_labels):
            node_labels = node_labels + self.n_nodes_per_label * [self.labels[l_]]

        return node_labels

    def get_block_labels(self, blocks_per_channel):
        """Generates a list of block labels for each node.

        Returns:
            block_labels: list of block labels for each node
        """
        block_labels = []

        for label in [chr(i) for i in range(97, 97 + self.n_labels)]:
            if label in blocks_per_channel[0]:
                block_labels = block_labels + self.n_nodes_per_label * [0]
            else:
                block_labels = block_labels + self.n_nodes_per_label * [1]

        return block_labels

    def get_sample_adj(self, adj_df):
        """Generates adjacency matrix (node x node) for a list of node labels

        Args:
            adj_df: (label x label) adjacency matrix as a dataframe, with labels as index and columns.

        Returns:
            node_labels: list of node labels
            adj: adjacency matrix
        """
        node_labels = self.get_node_labels()
        adj = np.zeros((len(node_labels), len(node_labels)))

        # for now this inefficient way is fine.
        for i, label_i in enumerate(node_labels):
            for j, label_j in enumerate(node_labels):
                adj[i, j] = adj_df.loc[label_i[0], label_j[0]]
        return node_labels, adj
