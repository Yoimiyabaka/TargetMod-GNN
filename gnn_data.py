import os
import json
import numpy as np
import copy
import torch
import random
import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from utils import UnionFindSet, get_bfs_sub_graph, get_dfs_sub_graph
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
from sklearn.model_selection import StratifiedKFold


class GNN_DATA:
    def __init__(self, ppi_path, exclude_protein_path=None, max_len=2000, skip_head=True, p1_index=0, p2_index=1,
                 label_index=2, graph_undirection=True, bigger_ppi_path=None):
        self.ppi_list = []
        self.ppi_dict = {}
        self.ppi_label_list = []
        self.protein_dict = {}
        self.protein_name = {}
        self.ppi_path = ppi_path
        self.bigger_ppi_path = bigger_ppi_path
        self.max_len = max_len

        name = 0
        ppi_name = 0
        # maxlen = 0
        self.node_num = 0
        self.edge_num = 0
        if exclude_protein_path != None:
            with open(exclude_protein_path, 'r') as f:
                ex_protein = json.load(f)
                f.close()
            ex_protein = {p: i for i, p in enumerate(ex_protein)}
        else:
            ex_protein = {}

        class_map = {'reaction': 0, 'binding': 1, 'ptmod': 2, 'activation': 3, 'inhibition': 4, 'catalysis': 5,
                     'expression': 6}

        for line in tqdm(open(ppi_path)):
            if skip_head:
                skip_head = False
                continue
            line = line.strip().split('\t')

            if line[p1_index] in ex_protein.keys() or line[p2_index] in ex_protein.keys():
                continue

            # get node and node name
            if line[p1_index] not in self.protein_name.keys():
                self.protein_name[line[p1_index]] = name
                name += 1

            if line[p2_index] not in self.protein_name.keys():
                self.protein_name[line[p2_index]] = name
                name += 1
            if p1_index == 11:
                aaaaaa = 1
            # get edge and its label
            temp_data = ""
            zj1 = line[p1_index]
            zj2 = line[p2_index]
            if line[p1_index] < line[p2_index]:
                temp_data = line[p1_index] + "__" + line[p2_index]
            else:
                temp_data = line[p2_index] + "__" + line[p1_index]

            if temp_data not in self.ppi_dict.keys():
                self.ppi_dict[temp_data] = ppi_name
                temp_label = [0, 0, 0, 0, 0, 0, 0]
                temp_label[class_map[line[label_index]]] = 1
                self.ppi_label_list.append(temp_label)
                ppi_name += 1
            else:
                index = self.ppi_dict[temp_data]
                temp_label = self.ppi_label_list[index]
                temp_label[class_map[line[label_index]]] = 1
                self.ppi_label_list[index] = temp_label

        if bigger_ppi_path != None:
            skip_head = True
            for line in tqdm(open(bigger_ppi_path)):
                if skip_head:
                    skip_head = False
                    continue
                line = line.strip().split('\t')

                if line[p1_index] not in self.protein_name.keys():
                    self.protein_name[line[p1_index]] = name
                    name += 1

                if line[p2_index] not in self.protein_name.keys():
                    self.protein_name[line[p2_index]] = name
                    name += 1

                temp_data = ""
                if line[p1_index] < line[p2_index]:
                    temp_data = line[p1_index] + "__" + line[p2_index]
                else:
                    temp_data = line[p2_index] + "__" + line[p1_index]

                if temp_data not in self.ppi_dict.keys():
                    self.ppi_dict[temp_data] = ppi_name
                    temp_label = [0, 0, 0, 0, 0, 0, 0]
                    temp_label[class_map[line[label_index]]] = 1
                    self.ppi_label_list.append(temp_label)
                    ppi_name += 1
                else:
                    index = self.ppi_dict[temp_data]
                    temp_label = self.ppi_label_list[index]
                    temp_label[class_map[line[label_index]]] = 1
                    self.ppi_label_list[index] = temp_label

        i = 0
        for ppi in tqdm(self.ppi_dict.keys()):
            name = self.ppi_dict[ppi]
            assert name == i
            i += 1
            temp = ppi.strip().split('__')
            self.ppi_list.append(temp)

        ppi_num = len(self.ppi_list)
        self.origin_ppi_list = copy.deepcopy(self.ppi_list)
        assert len(self.ppi_list) == len(self.ppi_label_list)
        for i in tqdm(range(ppi_num)):
            seq1_name = self.ppi_list[i][0]
            seq2_name = self.ppi_list[i][1]
            # print(len(self.protein_name))
            self.ppi_list[i][0] = self.protein_name[seq1_name]
            self.ppi_list[i][1] = self.protein_name[seq2_name]

        if graph_undirection:
            for i in tqdm(range(ppi_num)):
                temp_ppi = self.ppi_list[i][::-1]
                temp_ppi_label = self.ppi_label_list[i]
                # if temp_ppi not in self.ppi_list:
                self.ppi_list.append(temp_ppi)
                self.ppi_label_list.append(temp_ppi_label)

        self.node_num = len(self.protein_name)
        self.edge_num = len(self.ppi_list)

    def get_protein_aac(self, pseq_path):
        # aac: amino acid sequences

        self.pseq_path = pseq_path
        self.pseq_dict = {}
        self.protein_len = []

        for line in tqdm(open(self.pseq_path)):
            line = line.strip().split('\t')
            ls1 = line[0]
            ls2 = line[1]
            if line[0] not in self.pseq_dict.keys():
                self.pseq_dict[line[0]] = line[1]
                self.protein_len.append(len(line[1]))

        print("protein num: {}".format(len(self.pseq_dict)))
        print("protein average length: {}".format(np.average(self.protein_len)))
        print("protein max & min length: {}, {}".format(np.max(self.protein_len), np.min(self.protein_len)))

    def embed_normal(self, seq, dim):
        if len(seq) > self.max_len:
            return seq[:self.max_len]
        elif len(seq) < self.max_len:
            less_len = self.max_len - len(seq)
            return np.concatenate((seq, np.zeros((less_len, dim))))
        return seq

    def vectorize(self, vec_path):
        self.acid2vec = {}
        self.dim = None
        for line in open(vec_path):
            line = line.strip().split('\t')
            temp = np.array([float(x) for x in line[1].split()])
            self.acid2vec[line[0]] = temp
            if self.dim is None:
                self.dim = len(temp)
        print("acid vector dimension: {}".format(self.dim))

        self.pvec_dict = {}

        for p_name in tqdm(self.pseq_dict.keys()):
            temp_seq = self.pseq_dict[p_name]
            temp_vec = []
            for acid in temp_seq:
                temp_vec.append(self.acid2vec[acid])
            temp_vec = np.array(temp_vec)

            temp_vec = self.embed_normal(temp_vec, self.dim)

            self.pvec_dict[p_name] = temp_vec

    def get_feature_origin(self, pseq_path, vec_path):
        self.get_protein_aac(pseq_path)

        self.vectorize(vec_path)

        self.protein_dict = {}
        for name in tqdm(self.protein_name.keys()):
            self.protein_dict[name] = self.pvec_dict[name]

    def get_connected_num(self):
        self.ufs = UnionFindSet(self.node_num)
        ppi_ndary = np.array(self.ppi_list)
        for edge in ppi_ndary:
            start, end = edge[0], edge[1]
            self.ufs.union(start, end)
    
    def generate_random_labels(self, positive_ratio=0.2):
        print(f"Generating random labels for nodes with positive ratio: {positive_ratio}")
        self.y = torch.zeros(self.node_num, dtype=torch.long)
        positive_count = int(self.node_num * positive_ratio)
        positive_indices = torch.randperm(self.node_num)[:positive_count]
        self.y[positive_indices] = 1
        print(f"Generated {positive_count} positive labels out of {self.node_num} nodes.")

    def generate_data_labels(self,file_path):
        
        label_data = pd.read_csv(file_path)
        #print(label_data)
        self.y = torch.zeros(self.node_num, dtype=torch.long)
        label_data["ProteinID"] = "9606." + label_data["ProteinID"].astype(str)
        protein_name_to_index = {name: index for name, index in self.protein_name.items()}
        
        label_data["index"] = label_data["ProteinID"].map(protein_name_to_index)
        #print(list(self.protein_name.keys())[:10])
        valid_labels = label_data.dropna(subset=["index"])
        valid_labels["index"] = valid_labels["index"].astype(int)
        #print(f"Valid labels: {valid_labels.head()}")
        self.y[valid_labels["index"]] = torch.tensor(valid_labels["IsTarget"].values, dtype=torch.long)
        #print(f"Number of nodes with label 1: {self.y.sum().item()}")

    def generate_data(self):
        self.get_connected_num()

        print("Connected domain num: {}".format(self.ufs.count))

        ppi_list = np.array(self.ppi_list)
        ppi_label_list = np.array(self.ppi_label_list)

        self.edge_index = torch.tensor(ppi_list, dtype=torch.long)
        self.edge_attr = torch.tensor(ppi_label_list, dtype=torch.long)
        self.x = []
        i = 0
        for name in self.protein_name:
            assert self.protein_name[name] == i
            i += 1
            self.x.append(self.protein_dict[name])

        self.x = np.array(self.x)
        self.x = torch.tensor(self.x, dtype=torch.float)

        '''
        if not hasattr(self, "y"):
            print("Warning: No labels found. Initializing y with zeros.")
            self.y = torch.zeros(self.node_num, dtype=torch.long)
        '''
        self.data = Data(x=self.x, edge_index=self.edge_index.T, edge_attr_1=self.edge_attr,y=self.y)

    def split_dataset(self,test_size=0.2,index_path=r"train_val_split_data/node_split.json"):
        
        labels = self.data.y.cpu().numpy()  #
        node_indices = torch.arange(self.node_num).numpy() 

        train_indices, valid_indices = train_test_split(
            node_indices,
            test_size=test_size,
            stratify=labels,  
            random_state=42
        )

        self.ppi_split_dict = {
            'train_index': train_indices.tolist(),
            'valid_index': valid_indices.tolist()
        }
        self.data.train_mask = train_indices
        self.data.val_mask = valid_indices

        print(f"Train nodes: {len(self.data.train_mask)}, Valid nodes: {len(self.data.val_mask)}")
        
        with open(index_path, 'w') as f:
            json.dump(self.ppi_split_dict, f, indent=4)
            print(f"Node split indices saved to {index_path}")

    def split_dataset_kfold(self, n_splits=8, index_path=r"train_val_split_data/kfold_split.json"):
        labels = self.data.y.cpu().numpy()
        node_indices = torch.arange(self.node_num)  
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        self.kfold_split_dict = {"folds": []}

        
        for fold, (train_indices, valid_indices) in enumerate(skf.split(node_indices, labels)):
            print(f"Fold {fold + 1}: Train nodes = {len(train_indices)}, Valid nodes = {len(valid_indices)}")
            
            
            self.kfold_split_dict["folds"].append({
                "train_index": train_indices.tolist(),
                "valid_index": valid_indices.tolist()
            })
            
            train_mask = torch.zeros(self.node_num, dtype=torch.bool)
            val_mask = torch.zeros(self.node_num, dtype=torch.bool)

            
            train_mask[train_indices] = True
            val_mask[valid_indices] = True
            
            
            self.data.train_mask = train_mask
            self.data.val_mask = val_mask

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, 'w') as f:
            json.dump(self.kfold_split_dict, f, indent=4)
            print(f"K-Fold split indices saved to {index_path}")

    def get_protein_index_df(self):
        
        protein_items = sorted(
            self.protein_name.items(),
            key=lambda x: x[1]  # 按索引排序
        )
        
        df = pd.DataFrame(protein_items, columns=["Protein_Name", "Index"])
        df["Protein_Name"]=df["Protein_Name"].str.replace("9606.", "", regex=False)
        return df