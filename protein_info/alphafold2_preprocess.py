import pandas as pd
f=pd.read_csv('protein_info/pdb_file/alpha_not_found.csv')
f['id']=f['id'].str.replace("9606.", "", regex=False)
f=f.sort_values(by='seq', key=lambda col: col.str.len())
f.to_csv('protein_info/alphafold2_seq_sorted.csv',index=False)