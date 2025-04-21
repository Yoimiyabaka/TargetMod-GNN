import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.explain import Explainer,GNNExplainer
from gnn_explainer_layer import GNNExplainer_old
from gnn_data import GNN_DATA  
from gnn_models_sag import ppi_model,ppi_model_explain,ppi_model_got_embs
import numpy as np
import pandas as pd
import os
from draw_ppi import visualize_custom_subgraph,visualize_feature_importance,visualize_feature_importance_violin,visualize_custom_subgraph_with_pathway

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
os.environ['TORCH_USE_CUDA_DSA'] = '1' 

def multi2big_x(x_ori):
    x_cat = torch.zeros(1, 7)
    x_num_index = torch.zeros(1553)
    for i in range(1553):
        x_now = torch.tensor(x_ori[i])
        x_num_index[i] = torch.tensor(x_now.size(0))
        x_cat = torch.cat((x_cat, x_now), 0)
    return x_cat[1:, :], x_num_index

def multi2big_batch(x_num_index):
    num_sum = x_num_index.sum()
    num_sum = num_sum.int()
    batch = torch.zeros(num_sum)
    count = 1
    for i in range(1,1553):
        zj1 = x_num_index[:i]
        zj11 = zj1.sum()
        zj11 = zj11.int()
        zj22 = zj11 + x_num_index[i]
        zj22 = zj22.int()
        size1 = x_num_index[i]
        size1 = size1.int()
        tc = count * torch.ones(size1)
        batch[zj11:zj22] = tc
        test = batch[zj11:zj22]
        count = count + 1
    batch = batch.int()
    return batch

def multi2big_edge(edge_ori, num_index):
    edge_cat = torch.zeros(2, 1)
    edge_num_index = torch.zeros(1553)
    for i in range(1553):
        edge_index_p = edge_ori[i]
        edge_index_p = np.asarray(edge_index_p)
        edge_index_p = torch.tensor(edge_index_p.T)
        edge_num_index[i] = torch.tensor(edge_index_p.size(1))
        if i == 0:
            offset = 0
        else:
            zj = torch.tensor(num_index[:i])
            offset = zj.sum()
        edge_cat = torch.cat((edge_cat, edge_index_p + offset), 1)
    return edge_cat[:, 1:], edge_num_index


ppi_path = "./protein_info/protein.actions.SHS27k.STRING.pro2.txt"
pseq_path = "./protein_info/protein.SHS27k.sequences.dictionary.pro3.tsv"
vec_path = "./protein_info/vec5_CTC.txt"
p_feat_matrix = "./protein_info/x_list.pt"
p_adj_matrix = "./protein_info/edge_list_12.npy"

# as same as training
ppi_data = GNN_DATA(ppi_path=ppi_path)
ppi_data.get_feature_origin(pseq_path=pseq_path, vec_path=vec_path)
node_label_path = "target_info\drug_target_ALL.csv"
ppi_data.generate_data_labels(node_label_path)
ppi_data.generate_data()
graph = ppi_data.data
ppi_list = ppi_data.ppi_list
protein_names_map=ppi_data.get_protein_index_df()

edge_type = torch.argmax(ppi_data.data.edge_attr_1, dim=1)

p_x_all = torch.load(p_feat_matrix)
p_edge_all = np.load(p_adj_matrix, allow_pickle=True)
p_x_all, x_num_index = multi2big_x(p_x_all)
p_edge_all, edge_num_index = multi2big_edge(p_edge_all, x_num_index)
batch = multi2big_batch(x_num_index) + 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ppi_model_got_embs().to(device)


checkpoint = torch.load("result_save\gnn_2025-03-14-18-49-10\gnn_model_valid_best.ckpt", map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

with torch.no_grad():
        output,embs_x = model(batch, p_x_all, graph.edge_index, p_edge_all,edge_type)
        pred_labels = (torch.sigmoid(output) > 0.5).int().cpu().data.flatten()
        true_labels = graph.y.cpu().data.flatten()

# 获取新预测的药物靶标数据
new_predict_mask = ((pred_labels==1) & (true_labels==0)).cpu().numpy()
new_predict_indices = np.where(new_predict_mask)[0].tolist()
true_indices=np.where(true_labels)[0].tolist()
pred_indices=np.where(true_labels)[0].tolist()



model_e = ppi_model_explain().to(device)
explainer = Explainer(
    model=model_e,
    algorithm=GNNExplainer(epochs=300),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='binary_classification',
        task_level='node',
        return_type='probs',
    ),
)

save_dir='result_save\gnn_2025-03-14-18-49-10\GNNExplainer'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
importance_edge_with_idx=[]
for idx in new_predict_indices[:]:
    idx=1056
    save_idx_dir=os.path.join(save_dir, f'node_{idx}')
    if not os.path.exists(save_idx_dir):
        os.mkdir(save_idx_dir)
    explanation = explainer(
        x=embs_x,
        edge_index=graph.edge_index,
        index=idx 
    )
    explanation = explanation.cpu()
    
    explanation.edge_index = explanation.edge_index.cpu()
    if explanation.edge_mask is not None:
        explanation.edge_mask = explanation.edge_mask.cpu()
    #print(explanation.edge_index)
    feat_path = os.path.join(save_idx_dir, f'feat_importance_node_{idx}.png')
    explanation.visualize_feature_importance(feat_path, top_k=10)
    #print(explanation.node_mask)

    
    #从 explanation 中获取边和边权重
    edge_index = explanation.edge_index.cpu()
    edge_mask = explanation.edge_mask.cpu() if explanation.edge_mask is not None else None
    most_importance_edge_score=edge_mask.max().item()
    


    #归一化并过滤低权重边
    if edge_mask is not None:
        edge_mask = edge_mask - edge_mask.min()
        edge_mask = edge_mask / (edge_mask.max() + 1e-12)
        mask = edge_mask > 1e-7
        edge_index = edge_index[:, mask]
        edge_mask = edge_mask[mask]
    else:
        edge_mask = torch.ones(edge_index.size(1))
    
    #获取子图蛋白质节点
    global_node_indices = torch.unique(edge_index).tolist()
    if not global_node_indices:
        print(f"Skipping empty subgraph for node {idx}.")
        continue
    sub_protein_indice = protein_names_map['Index'].isin(global_node_indices)
    df = protein_names_map.loc[sub_protein_indice, :]
    df.iloc[:,0]=df.iloc[:,0].str.replace("9606.", "", regex=False) 
    protein_name_path = os.path.join(save_idx_dir, f'protein_name_{idx}.csv')
    df.to_csv(protein_name_path,index=False)
    df1=df.iloc[:,0]#只保存蛋白名称用于DAVAID
    df1.to_csv(os.path.join(save_idx_dir, f'protein_name_{idx}_only.csv'),index=False)

    #可视化特征图
    visualize_feature_importance(
        node_dataframe=df,
        explanation=explanation,
        save_path=os.path.join(save_idx_dir, f'Node_Importance_node_{idx}.png'),
        top_k=10,
        )
    
    module_drug_score=visualize_feature_importance_violin(
        node_dataframe=df,
        true_indices=true_indices,
        explanation=explanation,
        save_path=os.path.join(save_idx_dir, f'Importance_violin_node_{idx}.png'),
        )

    #保存子图最重要边掩码和idx和模块药物重要性分数
    importance_edge_with_idx.append({
        "idx": idx,
        "most_importance_edge": most_importance_edge_score,  # 转换为Python数值
        "module_drug_score":module_drug_score
    })

    #用内置方法画子图
    #subgraph_path = os.path.join(save_idx_dir, f'subgraph_node_{idx}.png')
    visualize_custom_subgraph(
        explanation=explanation,
        protein_index_to_name=protein_names_map.set_index('Index')['Protein_Name'].to_dict(),
        idx=idx,
        true_labels=true_labels,  # 传入真实标签数组
        save_path=os.path.join(save_idx_dir, f'subgraph_node_{idx}.png'),
        true_color='#ff0000',      # 红色表示 True 标签节点
        false_color='#ffffff',     # 白色表示 False 标签节点
        central_node_color='#00ff00',  # 绿色表示中心节点
        edge_colormap='plasma',    # 等离子体颜色映射
        layout='spring',     #  布局算法
        node_size=400,            # 节点尺寸
        font_size=8,              
        dpi=400                    # 更高分辨率
    )
    
    visualize_custom_subgraph_with_pathway(
        explanation=explanation,
        protein_index_to_name=protein_names_map.set_index('Index')['Protein_Name'].to_dict(),
        idx=idx,
        true_labels=true_labels,  # 传入真实标签数组
        save_path=save_idx_dir,
        true_color='#ff0000',      # 红色表示 True 标签节点
        false_color='#ffffff',     # 白色表示 False 标签节点
        central_node_color='#00ff00',  # 绿色表示中心节点
        edge_colormap='plasma',    # 等离子体颜色映射
        layout='spring',     #  布局算法
        node_size=400,            # 节点尺寸
        font_size=8,              
        dpi=400                    # 更高分辨率
    )
    
    #explanation.visualize_graph(subgraph_path)
    print(f"Explanation for node {idx} saved to {save_idx_dir}")
    break

importance_edge_with_idx = pd.DataFrame(importance_edge_with_idx)
importance_edge_with_idx = importance_edge_with_idx.sort_values(by="module_drug_score", ascending=False)
#protein_names_map = ppi_data.get_protein_index_df()
name_mapping = protein_names_map.set_index('Index')['Protein_Name'].to_dict()
importance_edge_with_idx['Protein_Name'] = importance_edge_with_idx['idx'].map(name_mapping)
save_csv_dir=os.path.join(save_dir, f'most_importance_score_with_idx.csv')
importance_edge_with_idx.to_csv(save_csv_dir,index=False)





    


