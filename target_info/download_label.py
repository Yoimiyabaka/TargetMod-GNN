import pandas as pd

import pandas as pd

def convert_to_dataframe():
    file_path = 'protein_info/ensp_uniprot.txt'
    
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


def trans_ensembl_to_uniport():
    
    pro_info=pd.read_csv("dataset\protein.SHS148k.sequences.dictionary.tsv",sep='\t',header=None)
    ensembl_ids=list(pro_info.iloc[:,0])
    a=0
    with open("esembl_ids_148k.txt",'w') as f:
        for i in ensembl_ids[0:]:
            i=i[5:]
            f.write(f'{i}\t')
            a+=1
    print(i)
    print(a)
    print(len(ensembl_ids))
            
        
import requests
from bs4 import BeautifulSoup


def get_target_from_TTD():
    

    TTD_data = []
    TTD_current_record = {}
    with open("target_info\P2-01-TTD_uniprot_all.txt", 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('-'):  
                continue
            if line.startswith("TARGETID") or line.startswith("UNIPROID") or line.startswith("TARGNAME") or line.startswith("TARGTYPE"):
                key, value = line.split('\t', 1)
                TTD_current_record[key] = value
                if key == "TARGTYPE": 
                    TTD_data.append(TTD_current_record)
                    TTD_current_record = {}
    TTD_df = pd.DataFrame(TTD_data)
    return TTD_df

def merge_all_target():
    TTD_df=get_target_from_TTD()
    DC_df=get_target_from_DrugCentral()
    CE_df=get_target_from_CHEMBL()
    DB_df=get_target_from_DrugBank()
    gene_data=pd.read_csv("target_info\idmapping_27k.tsv",sep='\t')#基因名----TTD用
    uniport_names=gene_data[["From","Entry","Entry Name"]]
    ensembl_data=pd.read_csv("dataset\protein.SHS27k.sequences.dictionary.pro3.tsv",sep='\t',header=None)#PPI中所有ensembl
    ensembl_list=list(ensembl_data.iloc[:,0].str.replace("9606.", "", regex=False))
    uniprot_ids=convert_to_dataframe() #ensembl与uniprot对应
    
    drug_target={}
    t=0
    f=0
    for i in range(0,len(ensembl_list)):
        ensembl_id=ensembl_list[i]
        #uniprot_ids['ENSP'] = uniprot_ids['ENSP'].str.strip()
        #ensembl_id = ensembl_id.strip()
        #print(uniprot_ids[uniprot_ids['ENSP']==ensembl_id]['UniProt'])
        #break
        uniport_id=uniprot_ids[uniprot_ids['ENSP']==ensembl_id]['UniProt'].values
        gene_id=uniport_names[uniport_names['From']==ensembl_id]['Entry Name'].values
        if uniport_id.size > 0 and gene_id.size > 0:
            if  uniport_id[0] in DB_df["UniProt ID"].values or\
                gene_id[0] in TTD_df["UNIPROID"].values or\
                uniport_id[0] in DC_df["ACCESSION"].values or\
                uniport_id[0] in CE_df["UniProt Accessions"].values:
                drug_target[ensembl_id]=1
                t+=1
            else:
                drug_target[ensembl_id]=0
                f+=1
            
        elif uniport_id.size > 0 and gene_id.size <= 0:
            if  uniport_id[0] in DB_df["UniProt ID"].values or\
                uniport_id[0] in DC_df["ACCESSION"].values or\
                uniport_id[0] in CE_df["UniProt Accessions"].values:
                drug_target[ensembl_id]=1
                t+=1
            else:
                drug_target[ensembl_id]=0
                f+=1
            
        elif uniport_id.size <= 0 and gene_id.size > 0:
            uniport_id=uniport_names[uniport_names['From']==ensembl_id]['Entry'].values
            if uniport_id.size > 0:
                if  uniport_id[0] in DB_df["UniProt ID"].values or\
                    gene_id[0] in TTD_df["UNIPROID"].values or\
                    uniport_id[0] in DC_df["ACCESSION"].values or\
                    uniport_id[0] in CE_df["UniProt Accessions"].values:
                    drug_target[ensembl_id]=1
                    t+=1
                else:
                    drug_target[ensembl_id]=0
                    f+=1
                
            else:
                if  gene_id[0] in TTD_df["UNIPROID"].values:
                    drug_target[ensembl_id]=1
                    t+=1
                else:
                    drug_target[ensembl_id]=0
                    f+=1
                
        else:
            drug_target[ensembl_id]=0
            f+=1
        

        
    print(t)
    drug_target = {"ProteinID": list(drug_target.keys()), "IsTarget": list(drug_target.values())}
    drug_target_df=pd.DataFrame(drug_target)
    drug_target_df.to_csv("target_info\drug_target_ALL.csv",index=False)
    
def get_target_from_DrugCentral():
    DC_data=pd.read_csv("target_info\drug.target.interaction.tsv",sep="\t")
    return DC_data

def get_target_from_CHEMBL():
    CE_data=pd.read_csv("target_info\DOWNLOAD-WUZ0t7Mx0di-WUmeFZiq8vT5q_ckwAu1cFcarEGX3sQ=.tsv",sep="\t")
    CE_data = CE_data.dropna(subset=["UniProt Accessions"])
    return CE_data
def get_target_from_DrugBank():
    DB_data=pd.read_csv("target_info\DrugBank_all.csv")
    DB_data = DB_data.dropna(subset=["UniProt ID"])
    return DB_data

    
def get_target_from_drugbank():
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Referer": "https://go.drugbank.com",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    }

    uniport_ids=pd.read_csv("idmapping_2024_12_06.tsv",sep='\t')
    uniport_ids=list(uniport_ids["Entry"])
    drug_target={}
    for uniport_id in uniport_ids[:]:
        query_url=f"https://go.drugbank.com/unearth/q?searcher=bio_entities&query={uniport_id}&button="
        try:
            response = requests.get(query_url,headers=headers)
            response.raise_for_status()  


            soup = BeautifulSoup(response.text, 'html.parser')

            result=soup.find('p',class_="lead")

            if "No results." in result:
                drug_target[uniport_id]=0
            else:
                drug_target[uniport_id]=1    
        except requests.RequestException as e:
            return f"Error during request: {e}"

def get_uniport_id():
    idx=298
    subgraph_data=pd.read_csv(f'result_save/gnn_2025-03-14-18-49-10/GNNExplainer/node_298/protein_name_{idx}_only.csv')
    subgraph_data_list=subgraph_data["Protein_Name"].tolist()
    gene_data=pd.read_csv("target_info\idmapping_27k.tsv",sep='\t')
    sub_gene_data=gene_data["From"].isin(subgraph_data_list)
    protein_name_all=gene_data.loc[sub_gene_data,:]
    protein_name_all.to_csv(f'result_save/gnn_2025-03-14-18-49-10/GNNExplainer/node_298/protein_name_{idx}_all.csv',sep='\t')
    


if __name__ == "__main__":
    #merge_all_target()
    #convert_to_dataframe()
    get_uniport_id()
    #get_target_from_drugbank()
    #trans_ensembl_to_uniport()