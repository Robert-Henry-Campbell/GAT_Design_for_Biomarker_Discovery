# Context

This is a legacy project from my MSC thesis and reprepresents my first project designing machine learning archetecture at scale on an HPC environment. The code is functional but does not use best practices (modular/scalable/testable), and is not how I would structure a similarly sized project today. For an example of an up-to-date project structure, consider my repo "Translational_Geroprotective_Molecule_Discovery" which covers my ongoing PhD work translating longevity therapies from mice to humans using causal genomic tools. 


---

# Single-Cell Multi-Omics Graph Attention Network

This repository explores using **Graph Attention Networks (GATs)** for classifying single-cell **RNA-seq/ATAC-seq** data with an emphasis on **attention-based pooling**. The main goal is to identify which gene expressions (and potential regulatory regions) define disease states.

## Research Motivation

- **Question**: Can the attention scores in a GAT pooling layer detect critical gene signals associated with disease states as effectively as standard approaches like TWAS (Transcriptome-Wide Association Study)?  
- **Focus**: Compare synthetic datasets (Aim 1) against an Alzheimer’s dementia dataset (Aim 2).  
- **Novelty**: Incorporating self-attention directly in the **pooling layer** (instead of the usual message-passing layer) to potentially yield more biologically interpretable results.

## Methods Overview

1. **Data Integration**:  
   - Multi-omics single-cell data combining RNA-seq and ATAC-seq.  
   - Synthetic datasets for method development and baseline comparisons.  
   - Alzheimer’s dementia dataset from [Cell Genomics](https://www.cell.com/cell-genomics/fulltext/S2666-979X(23)00019-8#sectitle0030).

2. **Model Architecture**:  
   - **Layer 1**: GAT message-passing layer for feature transformation.  
   - **Layer 2**: **Attention-based pooling**, where node scores guide which genes (nodes) remain in the graph.  
   - **Layer 3**: Fully connected classification layer predicting disease status.

3. **Performance Metrics**:  
   - **Macro F1**, precision, recall.  
   - **Attention metrics** (score separation, clustering) to assess interpretability.

## Repository Structure

```bash
drug_mr_project/
├── data/
│   ├── drug_a/                   
│   │   ├── strategy_a
│   │   │   ├── raw/               # CSVs for each gene's QTL data
│   │   │   └── processed/         # Final SNP sets after QC
├── results/
│   ├── figures/                   # Model plots
│   └── tables/                    # Numerical outputs
├── config/
│   ├── pipeline_params.yaml       # Pipeline params (p-value thresholds, etc.)
│   └── drug_confounders.yaml      # Per-drug confounder sets for queries
├── src/
│   ├── qc/
│   │   ├── filter_snps.py
│   │   ├── ld_clump.py
│   │   ├── compute_Fstats.py
│   │   ├── phenoscanner.py
│   │   └── ...
│   ├── harmonize/
│   │   └── harmonize_snps.py
│   ├── mr/
│   │   └── run_mr.py
│   ├── utils/
│   │   └── i_o.py
│   └── pipeline.py
├── dic_pickles/                   # Feature & node metadata dictionaries
├── edge_indices/                  # Edge index files for adjacency
├── graph_idx/                     # Graph index files for subsets
├── models/                        # Trained models & results
├── environment.yml                # Conda environment specification
└── README.md                      # Project documentation
```

## Setup & Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/vessacks/Translational_Geroprotective_Molecule_Discovery.git
   cd Translational_Geroprotective_Molecule_Discovery
   ```

2. **Create Conda Environment** (recommended):
   ```bash
   conda env create -f environment.yml
   conda activate drug_mr_project
   ```
   Or install required Python packages manually (e.g., `torch-geometric`, `pandas`, `matplotlib`).

3. **Data Preparation**:
   - Place your single-cell data, pickled dictionaries, and edge indices in their respective folders (see “Repository Structure”).
   - Adjust any file paths in `config/pipeline_params.yaml` or script arguments.

## Usage

- **Training**:  
  - Edit parameters at the top of `src/pipeline.py` (e.g., `num_epochs`, `batch_size`) to match your system or HPC settings.  
  - Run the model:
    ```bash
    python src/pipeline.py
    ```
  - The script saves metrics, model checkpoints, and plots in the `models/` folder.

- **Analysis**:  
  - Results can be found in CSV files in `models/` or in `results/tables/` for aggregated metrics.  
  - Figures (e.g., training curves) are automatically stored in `results/figures/`.

## Current Findings

- Preliminary results on synthetic data suggest that GAT pooling attention scores can highlight feature groups indicative of phenotypic differences.
- Real-world single-cell datasets require careful hyperparameter tuning (batch size, pooling ratio, etc.) due to large memory demands.

## Contributing

We welcome suggestions or pull requests. Contact:
```
Name: Robert Campbell
Email: h.robert.campbell@gmail.com
Institution: Nuffield Department of Population Health, Oxford University
```

## License

Please check the repository’s LICENSE file or add one if needed. In the absence of a separate license, all rights are reserved by the author(s).

---

Feel free to customize the above sections to fit your latest workflows, rename directories, or reorganize content as your project evolves.
