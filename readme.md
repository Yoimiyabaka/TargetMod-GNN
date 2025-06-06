# TargetMod-GNN
Target Module Identification via Graph Neural Networks
A hierarchical GNN-based framework for drug target module discovery by integrating protein structure and interaction networks. The project leverages the HIGH-PPI model and GNNExplainer to identify interpretable target modules and reveal underlying mechanisms of drug action.
## Dependencies
TargetMod-GNN runs on Python 3.7-3.9. To install all dependencies, directly run:
```
cd TargetMod-GNN-main
conda env create -f environment.yml
conda activate TargetMod-GNN
```
Download the following whl files to `./file/`: [torch-scatter](https://data.pyg.org/whl/torch-1.11.0%2Bcu102/torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl), [torch-sparse](https://data.pyg.org/whl/torch-1.11.0%2Bcu102/torch_sparse-0.6.13-cp39-cp39-linux_x86_64.whl), [torch-cluster](https://data.pyg.org/whl/torch-1.11.0%2Bcu102/torch_cluster-1.6.0-cp39-cp39-linux_x86_64.whl), [torch-spline-conv](https://data.pyg.org/whl/torch-1.11.0%2Bcu102/torch_spline_conv-1.2.1-cp39-cp39-linux_x86_64.whl).

```
cd ./file
pip install torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
pip install torch_sparse-0.6.13-cp39-cp39-linux_x86_64.whl
pip install torch_cluster-1.6.0-cp39-cp39-linux_x86_64.whl
pip install torch_spline_conv-1.2.1-cp39-cp39-linux_x86_64.whl
pip install torch-geometric
```
# Datasets
Three datasets (SHS27k, SHS148k and STRING) can be downloaded from the [Google Drive](https://drive.google.com/drive/folders/1Yb-fdWJ5vTe0ePAGNfrUluzO9tz1lHIF?usp=sharing):
* `protein.actions.SHS27k.STRING.pro2.txt`             PPI network of SHS27k
* `protein.SHS27k.sequences.dictionary.pro3.tsv`      Protein sequences of SHS27k
* `protein.actions.SHS148k.STRING.txt`             PPI network of SHS148k
* `protein.SHS148k.sequences.dictionary.tsv`         Protein sequences of SHS148k
* `9606.protein.action.v11.0.txt`         PPI network of STRING
* `protein.STRING_all_connected.sequences.dictionary.tsv`             Protein sequences of STRING
* `edge_list_12`             Adjacency matrix for all proteins in SHS27k
* `x_list`             Feature matrix for all proteins in SHS27k

# drug target Prediction

Example: predicting unknown drug target in SHS27k datasets with native structures:

## Using Processed Data for SHS27k Dataset

Download  `protein.actions.SHS27k.STRING.pro2.txt`, `protein.SHS27k.sequences.dictionary.pro3.tsv`, `edge_list_12`, `x_list` and `vec5_CTC.txt` to `./HIGH-PPI-main/protein_info/`.

## Data Processing for New Datasets (if applicable)
Prepare all related PDB files. Native protein structures can be downloaded in batches from the [RCSB PDB](https://www.rcsb.org/downloads), and predicted protein structures with errors can be downloaded from the [AlphaFold database](https://alphafold.ebi.ac.uk/). Put all of the PDB files in `./protein_info/`.

Generate adjacency matrix with native PDB files:
```
python ./protein_info/generate_adj.py --distance 12
```
Generate feature matrix:
```
python ./protein_info/generate_feat.py
```

## Training
To predict drug target, use 'model_train.py' script to train HIGH-PPI with the following options:
* `ppi_path`             str, PPI network information
* `pseq_path`             str, Protein sequences
* `p_feat_matrix`       str, The feature matrix of all protein graphs
* `p_adj_matrix`       str, The adjacency matrix of all protein graphs
* `split`       str, Dataset split mode
* `save_path`             str, Path for saving models, configs and results
* 'epoch_num'     int, Training epochs
```
python model_train.py --ppi_path ./protein_info/protein.actions.SHS27k.STRING.pro2.txt --pseq ./protein_info/protein.SHS27k.sequences.dictionary.pro3.tsv --split random --p_feat_matrix ./protein_info/x_list.pt --p_adj_matrix ./protein_info/edge_list_12.npy --save_path ./result_save --epoch_num 5
```
