import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
import os,sys
from urllib3.util.retry import Retry
import subprocess

def convert_to_dataframe():
    file_path = 'protein_info/ensp_uniprot.txt'
    #file_path='protein_info/alphafold2_seq_sorted_3.csv'
    with open(file_path, 'r') as file:
        line = file.readline().strip()
    entries = line.split(', ')
    
    data = []
    for entry in entries:
        if isinstance(entry, list): 
            entry = entry[0]
        cleaned_entry = entry.strip("'").replace('\\t', '\t')
        data.append(cleaned_entry.split('\t'))

    df = pd.DataFrame(data, columns=["ENSP", "UniProt"])
    df["ENSP"] = df["ENSP"].str.replace("9606.", "", regex=False)
    return df




# 打开日志文件

ALPHAFOLD_URL_TEMPLATE = 'https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb'
LOG_FILE='protein_info/pdb_file/download_log_alphafold_2.txt'
OUTPUT_PDB_DIR='protein_info/pdb_file'
os.makedirs(OUTPUT_PDB_DIR, exist_ok=True)
try:
    log = open(LOG_FILE, 'w', encoding='utf-8')
except OSError as e:
    print(f"无法打开日志文件 {LOG_FILE}，错误: {e}")
    exit(1)



def log_message(message):
    print(message)
    log.write(message + '\n')

def download_alphafold(ensembl_id,uniprot_id, output_dir, session):
    """
    从AlphaFold数据库下载指定UniProt ID的结构文件
    """
    alpha_url = ALPHAFOLD_URL_TEMPLATE.format(uniprot_id=uniprot_id)
    try:
        response = session.get(alpha_url)
        if response.status_code == 200:
            filename = f"{ensembl_id}.pdb"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w') as file:
                file.write(response.text)
            log_message(f"下载成功: {filename} (来源: AlphaFold)")
            return 0
        else:
            log_message(f"!无法下载AlphaFold文件: {ensembl_id}, 状态码: {response.status_code}")
            log_message(f"响应内容: {response.text}")
            return 1
            #subprocess.run(["Rscript", "protein_info\download_pdb.R",ensembl_id])
            #log_message(f"下载成功: {ensembl_id}:{uniprot_id} (来源: pdb)")
            #return False
    except Exception as e:
        log_message(f"!下载AlphaFold文件 {ensembl_id} 时发生异常: {e}")
        return 1
        
        #return False

def create_session():
    """
    创建一个带有重试机制的requests Session
    """
    session = requests.Session()
    retry = Retry(
        total=5,  # 总共重试5次
        backoff_factor=1,  # 重试间隔因子
        status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的HTTP状态码
        allowed_methods=["GET", "POST"]  # 替换 method_whitelist 为 allowed_methods
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def main():
    #open_log()
    uniprot_ids=convert_to_dataframe()
    ensembl_data=pd.read_csv("dataset\protein.SHS148k.sequences.dictionary.tsv",sep='\t',header=None)
    ensembl_list=list(ensembl_data.iloc[:,0].str.replace("9606.", "", regex=False))
    gene_data=pd.read_csv("target_info\idmapping_148k.tsv",sep='\t')
    uniport_names=gene_data[["From","Entry","Entry Name"]]
    session = create_session()
    f=0
    for i in range(0,len(ensembl_list)):
        ensembl_id=ensembl_list[i]
        uniport_id=uniprot_ids[uniprot_ids['ENSP']==ensembl_id]['UniProt'].values
        if uniport_id.size > 0 and uniport_id!='Noneid':
            f1=download_alphafold(ensembl_id,uniport_id[0],OUTPUT_PDB_DIR,session)
            f=f+f1

            
        else:
            uniport_id=uniport_names[uniport_names['From']==ensembl_id]['Entry'].values
            if uniport_id.size > 0:
                f2=download_alphafold(ensembl_id,uniport_id[0],OUTPUT_PDB_DIR,session)
                f=f+f2
            else:
                log_message(f"!找不到pdb文件: {ensembl_id}")
                f=f+1
                #subprocess.run(["Rscript", "protein_info\download_pdb.R",ensembl_id])
                #log_message(f"下载成功: {ensembl_id} (来源: pdb)")
    print(f'{f}个没找到。')




def extract_ensp_ids():
    file_path = "protein_info\pdb_file\download_log_alphafold.txt"
    ensembl_data=pd.read_csv("dataset\protein.SHS148k.sequences.dictionary.tsv",sep='\t',header=None)
    ensembl_data.columns = ['ensembl_ids', 'seq']
    enriched_ids = []
    enriched_ids_seqs=dict()
    a=0
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("!"):
                parts = line.split(":")
                if len(parts) > 1:
                    ensp_part = parts[1].strip()
                    ensp_id = ensp_part.split(",")[0].strip()
                    enriched_ids.append(ensp_id)
                    a+=1
    print(a)
    for ensembl_id in enriched_ids[:]:
        ensembl_id='9606.'+ensembl_id
        if ensembl_id in ensembl_data['ensembl_ids'].values:
            #print(ensembl_id)
            seq=ensembl_data.loc[ensembl_data['ensembl_ids'] == ensembl_id, 'seq'].iloc[0]
            enriched_ids_seqs[ensembl_id]=seq
    df_ids_seqs = pd.DataFrame(list(enriched_ids_seqs.items()), columns=['id', 'seq'],index=None)
    df_ids_seqs.to_csv('protein_info/pdb_file/alpha_not_found.csv',index=False)          
    
if __name__=='__main__':
    main()
    #extract_ensp_ids()