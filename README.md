# GAT Design for Biomarker Discovery

This repository contains research code from a master's thesis exploring **graph attention networks (GATs)** for classifying single--cell multi--omic data. The goal is to use attention scores to highlight genes that may act as biomarkers of disease.

The code is largely experimental and was originally developed for a high performance computing (HPC) environment. Several scripts write intermediate results and models to disk. Paths may need adjusting if you run the code on a different system.

## Contents

- `Complete_Model_HPC_callrate_v5_45_24.py` – main training script for real data
- `Synth_Model_v3_37_22.py` – simplified model for synthetic data experiments
- `sag_pool_custom.py` – custom self--attention graph pooling layer
- `dic_pickles/` – pickled metadata dictionaries for features and nodes
- `edge_indices/` – prebuilt adjacency matrices
- `graph_idx/` – graph index lists for training subsets
- `models/` and `synth_models/` – saved models, metrics and call rate files
- `testing_notes` and `version_notes.txt` – development notes

## Requirements

Python 3.8+ with the packages listed in `requirements.txt`. PyTorch and PyTorch Geometric are required for model training.

```
python -m pip install -r requirements.txt
```

## Quick Start

1. Prepare your feature dictionaries, edge index tensors and graph index lists as provided in the `dic_pickles/`, `edge_indices/` and `graph_idx/` folders.
2. Adjust any paths or parameters at the top of the desired training script.
3. Run the model:

```
python Complete_Model_HPC_callrate_v5_45_24.py
```

Synthetic data experiments can be executed with `Synth_Model_v3_37_22.py`.

## Testing

Unit tests live in the `tests/` directory. The file `tests/test_gat_small.py`
exercises a minimal GAT model. Run it with `pytest`:

```bash
pytest tests/test_gat_small.py
```

You can also verify that all Python files compile:

```bash
python -m py_compile $(git ls-files '*.py')
```

## License and Citation

No explicit license is provided. All rights remain with the original author. If you use this code in academic work, please cite this repository.

## Contact

Robert Campbell – [h.robert.campbell@gmail.com](mailto:h.robert.campbell@gmail.com)
