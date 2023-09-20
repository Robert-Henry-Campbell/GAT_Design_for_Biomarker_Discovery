# Single-Cell-atac-seq-and-rna-seq-Graph-Attention-Network-Classification-Model


## Research Question
Can the attention mechanism from a Graph Attention Network (GAT) be used in a pooling layer to identify gene expressions that characterize cellular disease state?
### Aim 1: Compare GAT Pooling to TWAS on synthetic dataset
### Aim 2: Test model on Alzheimer's Dementia Dataset

### Novelty
- Novel use of attention in the pooling layer rather than message passing layer of a GAT classification model 

## Methods 
**GAT Model**
### Layer 1: Message Passing
### Layer 2: Graph Pooling with Node Self-Attention
### Layer 3: Classification Layer

**Predictors**
Experimentally Integrated ATAC/RNA dataset
- Multi-omics 10x protocol
- Baseline: RNA-seq only
- Combined model: RNA + ATAC-seq

**Metrics**
- Macro metrics: F1 score, precision, recall
- Attention Metrics: Score separation, score clustering

**Statistical Model**
- Three layer GAT Classification model with attention mechanism in pooling layer. 
  
## Data
[AD Dataset](https://www.cell.com/cell-genomics/fulltext/S2666-979X(23)00019-8#sectitle0030) (Spa)


## Files
dic_pickles - pickled dictionaries relating feature names to node indexs, graph indices to graph metadata, and id to node metadata
edge_indices - edge index files with varying search window sizes
graph_idx - graph indices for selecting subsets of the dataset
models - trained models and model results from significant model runs 

