import subprocess
import sys
import types
import torch
from torch_geometric.data import Data, Batch


def _load_gat_with_stub():
    class SAGPoolStub(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def forward(self, x, edge_index, edge_attr=None, batch=None):
            if batch is None:
                batch = x.new_zeros(x.size(0), dtype=torch.long)
            perm = torch.arange(x.size(0), device=x.device)
            score = torch.ones(x.size(0), device=x.device)
            return x, edge_index, edge_attr, batch, perm, score, score

    stub_module = types.ModuleType("sag_pool_custom")
    stub_module.SAGPoolingCustom = SAGPoolStub
    sys.modules["sag_pool_custom"] = stub_module
    from src.models.gat import GAT
    return GAT


def test_gat_forward_backward():
    subprocess.run("python -m py_compile $(git ls-files '*.py')", shell=True, check=True)

    GAT = _load_gat_with_stub()
    x1 = torch.rand(3, 2)
    edge_index1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    y1 = torch.tensor([0])

    x2 = torch.rand(3, 2)
    edge_index2 = torch.tensor([[0, 2], [2, 1]], dtype=torch.long)
    y2 = torch.tensor([1])

    batch = Batch.from_data_list([
        Data(x=x1, edge_index=edge_index1, y=y1),
        Data(x=x2, edge_index=edge_index2, y=y2),
    ])

    model = GAT(in_channels=2, hidden_channels=4, num_heads=1, pooling_keep_count=2, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    out, _ = model(batch.x, batch.edge_index, batch.batch)
    loss = torch.nn.functional.cross_entropy(out, batch.y)
    loss.backward()
    optimizer.step()
