import pandas as pd
import gseapy as gp
import matplotlib.pyplot as plt
from gseapy import Biomart
gene_list=pd.read_csv('result_save\gnn_2025-03-14-18-49-10\GNNExplainer\protein_name_283.csv')
queries ={'ensembl_peptide_id': gene_list["Protein_Name"].values}
print(queries)
bm = Biomart()
results = bm.query(dataset='hsapiens_gene_ensembl',
                   attributes=['ensembl_peptide_id', 'external_gene_name', 'entrezgene_id'],
                   filters=queries)

gene_symbols = results['external_gene_name'].dropna().unique().tolist()
print(gene_symbols)

from gseapy import enrichr

# 执行富集分析
enr_results = gp.enrichr(gene_list=gene_list,
                 gene_sets=['GO_Biological_Process_2023','KEGG_2021_Human'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                 cutoff=0.5
                )

# 查看前10显著通路
print(enr_results.results.head(10))