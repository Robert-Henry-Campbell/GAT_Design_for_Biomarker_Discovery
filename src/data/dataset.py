import os.path as osp
import torch
from torch_geometric.data import Dataset


class SCDataset(Dataset):
    """Dataset that appends a precomputed call rate feature to each node."""

    def __init__(self, root, edge_index, graph_idx_list, device, synth_data,
                 transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.edge_index = edge_index
        self.graph_idx_list = graph_idx_list
        self.device = device
        self.synth_data = synth_data
        # call rate is calculated once for the whole dataset
        self.call_rate = self.calculate_call_rate_by_node().to(device)

    @property
    def raw_file_names(self):
        return [f"noedgeidx_{i}.pt" for i in self.graph_idx_list]

    @property
    def processed_file_names(self):
        return self.raw_file_names

    def len(self):
        return len(self.raw_file_names)

    def get_without_call_rates(self, idx):
        data = torch.load(osp.join(self.root, self.raw_file_names[idx]))
        data.edge_index = self.edge_index.to(self.device)
        if data.x.device != self.device:
            data = data.to(self.device)
        return data

    def get(self, idx):
        data = torch.load(osp.join(self.root, self.raw_file_names[idx]))
        data = data.to(self.device)
        call_rate = self.call_rate
        data.x = torch.cat((data.x, call_rate.unsqueeze(-1)), dim=-1)
        data.edge_index = self.edge_index.to(self.device)
        if data.x.device != self.device:
            data = data.to(self.device)
        return data

    def calculate_call_rate_by_node(self):
        graph_count = self.len()
        node_nonzero_count = torch.zeros(
            self.get_without_call_rates(0).num_nodes,
            dtype=torch.float,
            device=self.device,
        )
        if not self.synth_data:
            for idx in range(graph_count):
                data = self.get_without_call_rates(idx)
                if data.x.is_sparse:
                    node_indices = data.x._indices()[0]
                    node_nonzero_count.index_add_(
                        0,
                        node_indices.to(self.device),
                        torch.ones(node_indices.size(0), device=self.device),
                    )
                else:
                    input(f"calculating call rate, but graph {idx} isn't sparse!")
            call_rate = node_nonzero_count / graph_count
        else:
            for idx in range(graph_count):
                data = self.get_without_call_rates(idx)
                node_indices = (data.x[:, 0] != 0).nonzero(as_tuple=True)[0]
                node_nonzero_count.index_add_(
                    0,
                    node_indices.to(self.device),
                    torch.ones(node_indices.size(0), device=self.device),
                )
            call_rate = node_nonzero_count / graph_count

        non_zero_indices = call_rate.nonzero(as_tuple=False).t()
        sparse_call_rate = torch.sparse_coo_tensor(
            indices=non_zero_indices,
            values=call_rate[call_rate.nonzero(as_tuple=True)],
            size=call_rate.size(),
            device=self.device,
        )
        return sparse_call_rate
