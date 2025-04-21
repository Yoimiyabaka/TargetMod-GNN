import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

def plot_metrics(epoch_metrics, save_path):
    epochs = range(len(epoch_metrics["train_loss"]))

    plt.figure()
    plt.plot(epochs, epoch_metrics["train_loss"], label="Train Loss")
    plt.plot(epochs, epoch_metrics["valid_loss"], label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(os.path.join(save_path, "loss_curve.png"))

    plt.figure()
    plt.plot(epochs, epoch_metrics["train_f1"], label="Train F1")
    plt.plot(epochs, epoch_metrics["valid_f1"], label="Valid F1")
    plt.plot(epochs, epoch_metrics["train_recall"], label="Train Recall")
    plt.plot(epochs, epoch_metrics["valid_recall"], label="Valid Recall")
    plt.plot(epochs, epoch_metrics["train_precision"], label="Train Precision")
    plt.plot(epochs, epoch_metrics["valid_precision"], label="Valid Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Metrics")
    plt.legend()
    plt.title("F1/Recall/Precision Curve")
    plt.savefig(os.path.join(save_path, "metrics_curve.png"))

    print("Plots saved.")
    plt.close()

import networkx as nx
import torch



def draw_ppi_graph_with_predictions(edge_index, node_labels, pred_labels, save_path="ppi_graph_predictions.png"):
    G = nx.Graph()
    edges = edge_index.cpu().numpy().T
    G.add_edges_from(edges)
    G.remove_nodes_from(list(nx.isolates(G)))

    
    node_colors = []
    for true_label, pred_label in zip(node_labels, pred_labels):

        if true_label == 0 and pred_label == 1:
            node_colors.append("red")  #pr->dr
        elif true_label == 1 and pred_label == 0:
            node_colors.append("blue") #dr->pr  
        elif true_label == 1 and pred_label==1:
            node_colors.append("green")  #dr
        else:
            node_colors.append("gray")  #pr

    plt.figure(figsize=(30, 30))
    pos = nx.spring_layout(G, seed=42,iterations=200,k=2)
    nx.draw(
        G,
        pos,
        node_color=node_colors,
        with_labels=True,
        node_size=300,
        font_size=5
    )
    plt.title("PPI Graph with Predictions")
    plt.savefig(save_path)
    plt.close()



def visualize_custom_subgraph(
    explanation,
    protein_index_to_name: dict,
    true_labels: np.ndarray,
    idx: int,
    save_path: str,
    true_color: str = '#ff0000',
    false_color: str = '#ffffff',
    central_node_color: str = '#00ff00',
    edge_colormap: str = 'viridis',
    layout: str = 'spring',
    node_size: int = 800,
    font_size: int = 8,
    dpi: int = 300,
):
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx

    #从 explanation 中获取边和边权重
    edge_index = explanation.edge_index.cpu()
    edge_mask = explanation.edge_mask.cpu() if explanation.edge_mask is not None else None

    #一化并过滤低权重边
    if edge_mask is not None:
        edge_mask = edge_mask - edge_mask.min()
        edge_mask = edge_mask / (edge_mask.max() + 1e-12)  # 避免除以零
        mask = edge_mask > 1e-7
        edge_index = edge_index[:, mask]
        edge_mask = edge_mask[mask]
    else:
        edge_mask = torch.ones(edge_index.size(1))

    #所有在过滤后边中出现的节点
    global_node_indices = torch.unique(edge_index).tolist()
    if not global_node_indices:
        print(f"Skipping empty subgraph for node {idx}.")
        return

    
    G = nx.DiGraph() if explanation.is_directed() else nx.Graph()
    for node in global_node_indices:
        G.add_node(node)
    for (src, dst), w in zip(edge_index.t().tolist(), edge_mask.tolist()):
        G.add_edge(src, dst, alpha=float(w))

    
    
    node_colors = []
    for node in G.nodes():
        if true_labels[node] == 1:
            color = true_color
        else:
            color = false_color
        node_colors.append(color)
    if idx in G.nodes():
        node_colors[list(G.nodes()).index(idx)] = central_node_color

    #节点标签
    node_labels = {node: protein_index_to_name.get(node, f'Unknown_{node}') for node in G.nodes()}

    #布局设置
    if layout == 'spring':
        pos = nx.spring_layout(G, k=0.5/(len(G.nodes())**0.5 + 1e-3), iterations=100)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G)

    #绘图
    fig, ax = plt.subplots(figsize=(20, 16))
    plt.axis('off')

    #绘制节点
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_size,
        edgecolors='black',
        linewidths=0.5,
        ax=ax
    )

    #绘制边
    edge_alpha = []
    edge_colors = []
    cmap = plt.cm.get_cmap(edge_colormap)

    for (src, dst), w in zip(edge_index.t().tolist(), edge_mask.tolist()):
        if (src, dst) in G.edges():
            G[src][dst]['alpha'] = float(w)
            edge_alpha.append(float(w))  # 透明度
            edge_colors.append(cmap(w))  # RGBA 颜色
    if len(G.edges()) > 0:
        edges = nx.draw_networkx_edges(
            G, pos,
            edgelist=G.edges(),
            width=0.8,
            alpha=edge_alpha,
            edge_color=edge_colors, 
            ax=ax
        )
    else:
        print(f"No edges to draw for node {idx}.")

    #绘制标签
    nx.draw_networkx_labels(
        G, pos,
        labels=node_labels,
        font_size=font_size,
        verticalalignment='bottom',
        ax=ax
    )

    #颜色条
    if edge_mask is not None and len(edge_mask) > 0:
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.get_cmap(edge_colormap),
            norm=plt.Normalize(vmin=0, vmax=1)
        )
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Edge Importance', shrink=0.8)

    #保存图像
    plt.title(f'Explanation Subgraph for Node {idx}\n({node_labels.get(idx, "Unknown")})', fontsize=12)
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close()
    plt.close(fig)

def visualize_feature_importance(
        node_dataframe:pd.DataFrame,
        explanation,
        save_path:str,
        top_k: int = None,
        ):
    import matplotlib.pyplot as plt
    import pandas as pd

    node_mask = explanation.get('node_mask')
    if node_mask is None:
        raise ValueError(f"The attribute 'node_mask' is not available "
                        f"in '{explanation.__class__.__name__}' "
                        f"(got {explanation.available_explanations})")
    if node_mask.dim() != 2 or node_mask.size(1) <= 1:
        raise ValueError(f"Cannot compute feature importance for "
                        f"object-level 'node_mask' "
                        f"(got shape {node_mask.size()})")


    node_indices = torch.tensor(node_dataframe['Index'].tolist())
    protein_labels = node_dataframe['Protein_Name'].tolist()
    selected_node_mask = node_mask[node_indices]
    node_scores = selected_node_mask.sum(dim=1).cpu().numpy()
    df = pd.DataFrame({
        'protein': protein_labels,
        'score': node_scores
    }).set_index('protein')


    # 排序并限制显示数量
    df = df.sort_values('score', ascending=True)  # 改为升序保证从上到下递减
    df = df.round(decimals=3)

    if top_k is not None:
        df_1 = df.tail(top_k)  # 使用tail因为排序是升序
        title = f"Top {top_k} Important Nodes"
    else:
        title = "Node Importance Analysis"
        df_1=df
        

    # 绘制图表
    fig=plt.figure(figsize=(10, 0.6 * len(df_1)))
    ax = df_1.plot(
        kind='barh',
        title=title,
        xlabel='Importance Score',
        legend=False,
        color='#1f77b4'
    )
    
    # 优化样式
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (score, _) in enumerate(zip(df_1['score'], df_1.index)):
        ax.text(score + 0.01, i, f'{score:.3f}', va='center')

    # 保存或显示
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close(fig)
               

def draw_DMI_barplot():
    df = pd.read_csv('result_save/gnn_2025-03-14-18-49-10/GNNExplainer/ami_scores_pro_nodes.csv')

    # 按重要性分数降序排序（如果尚未排序）
    #df = df.sort_values('most_importance_edge', ascending=False)
    
    # 创建画布和坐标轴
    df = df.sort_values('module_drug_score', ascending=True)

    df=df.tail(10)
    convert_id=pd.read_csv('protein_info\entrezgene_id_and_symbol_converted.tsv',sep='\t')
    convert_id['ENTREZID'] = convert_id['ENTREZID'].astype(str)
    df = df.merge(convert_id[['ENSEMBL_ID', 'SYMBOL']], left_on='Protein_Name', right_on='ENSEMBL_ID', how='left')

    plt.figure(figsize=(10, 8))  # 宽度10英寸，高度8英寸

    # 绘制水平条形图
    ax = df.plot(
        kind='barh',
        y='module_drug_score',
        x='SYMBOL',
        color='#2c7bb6',  # 使用十六进制颜色代码
        edgecolor='black',  # 条形边框颜色
        linewidth=0.7,     # 边框粗细
        legend=False
    )

    # 优化图表样式
    plt.title('DMI Score Ranking Top 10', fontsize=14, pad=20)
    plt.xlabel('DMI Score', fontsize=12)
    plt.ylabel('Protein Name', fontsize=12)
    #plt.xlim(0.967, 0.971)  # 根据数据范围调整x轴范围

    # 添加数值标签（显示在条形右侧）
    
    for i, (score, name) in enumerate(zip(df['module_drug_score'], df['SYMBOL'])):
        ax.text(score + 0.0003,  # X坐标：分数值右侧偏移0.0003
                i,               # Y坐标：当前条形位置
                f'{score:.4f}',  # 显示4位小数
                va='center',     # 垂直居中
                ha='left',       # 水平左对齐
                fontsize=10)

    # 美化坐标轴
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')

    # 添加横向网格线
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    # 调整y轴标签间距
    plt.tight_layout()

    # 保存或显示
    plt.savefig('result_save\gnn_2025-03-14-18-49-10\GNNExplainer\DMI.png', dpi=300, bbox_inches='tight')  # 保存为高清图片
    plt.close()

#draw_DMI_barplot()


def visualize_feature_importance_violin(
        node_dataframe,
        true_indices,
        explanation,
        save_path:str,
        ):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import torch

    node_mask = explanation.get('node_mask')
    if node_mask is None:
        raise ValueError(f"The attribute 'node_mask' is not available "
                        f"in '{explanation.__class__.__name__}' "
                        f"(got {explanation.available_explanations})")
    if node_mask.dim() != 2 or node_mask.size(1) <= 1:
        raise ValueError(f"Cannot compute feature importance for "
                        f"object-level 'node_mask' "
                        f"(got shape {node_mask.size()})")
    

    node_indices = node_dataframe['Index'].values
    true_mask = np.isin(node_indices, true_indices)
    #print(true_mask)
    false_mask = ~true_mask
    # 使用子图索引提取掩码对应的行
    subgraph_node_mask = node_mask[node_indices]  # 形状 [334, 64]
    # 计算得分
    true_scores = subgraph_node_mask[true_mask].sum(dim=1).cpu().numpy()
    false_scores = subgraph_node_mask[false_mask].sum(dim=1).cpu().numpy()
    
    # 计算每个节点的总重要性得分（对特征维度求和）
    node_scores = subgraph_node_mask.sum(dim=1)  # 形状 [334]

    # 计算模块药物重要性分数
    true_total = node_scores[true_mask].sum().item()  # 真实节点总分
    total = node_scores.sum().item()                  # 所有节点总分
    module_drug_score = true_total / total if total != 0 else 0.0



    # Create DataFrame for visualization
    df = pd.DataFrame({
        'Score': np.concatenate([true_scores, false_scores]),
        'Group': ['True'] * len(true_scores) + ['False'] * len(false_scores)
    })

    # Plot configuration
    fig=plt.figure(figsize=(10, 4))
    sns.violinplot(
        data=df,
        x='Score',
        y='Group',
        orient='h',
        palette={'True': '#1f77b4', 'False': '#ff7f0e'},
        cut=0  # Extend violin range to data limits
    )

    # Style adjustments
    plt.title('Distribution of Feature Importance Scores', pad=20)
    plt.xlabel('Importance Score', labelpad=10)
    plt.ylabel('')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(True)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    # Save or show plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close(fig)

    return module_drug_score



def draw_most_enrich_plot():
    import pandas as pd
    import matplotlib.pyplot as plt

    # 读取数据
    df = pd.read_csv('result_save\gnn_2025-03-14-18-49-10\GNNExplainer\sorce_with_enrich.csv')  # 适当调整文件路径和分隔符
    df=df.head(100)
    #############KEGG绘制################
    pathway_counts = df['top_KEGG_pathway'].value_counts()
    pathway_counts=pathway_counts.head(10)
    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    pathway_counts.plot(kind='bar', color='#2c7bb6', edgecolor='black', linewidth=0.7)

    # 添加标签
    plt.xlabel('KEGG Pathway', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('KEGG Pathway Occurrences', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')

    # 显示数值标签
    for i, count in enumerate(pathway_counts):
        plt.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=10)

    # 美化图表
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # 保存或显示
    plt.savefig('result_save\gnn_2025-03-14-18-49-10\GNNExplainer\KEGG_pathway_counts.png', dpi=300, bbox_inches='tight')
    
    #############GO_BP绘制################   
    pathway_counts_BP = df['top_BP_pathway'].value_counts()
    pathway_counts_BP=pathway_counts_BP.head(10)
    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    pathway_counts_BP.plot(kind='bar', color='#2c7bb6', edgecolor='black', linewidth=0.7)

    # 添加标签
    plt.xlabel('GO BP Pathway', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('GO BP Pathway Occurrences', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')

    # 显示数值标签
    
    for i, count in enumerate(pathway_counts):
        plt.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=10)
    
    # 美化图表
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # 保存或显示
    plt.savefig('result_save\gnn_2025-03-14-18-49-10\GNNExplainer\BP_pathway_counts.png', dpi=300, bbox_inches='tight')
#draw_most_enrich_plot()


def visualize_custom_subgraph_with_pathway(
    explanation,
    protein_index_to_name: dict,
    true_labels: np.ndarray,
    idx: int,
    save_path: str,
    true_color: str = '#ff0000',
    false_color: str = '#ffffff',
    central_node_color: str = '#00ff00',
    edge_colormap: str = 'viridis',
    layout: str = 'spring',
    node_size: int = 800,
    font_size: int = 8,
    dpi: int = 300,
):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx
    import pandas as pd
    import torch

    # 加载 KEGG 富集结果
    save_png_path=os.path.join(save_path, f'subgraph_with_pathway_node_{idx}.png')
    kegg_path=f"result_save/gnn_2025-03-14-18-49-10/GNNExplainer/node_{idx}/enrich/KEGG_{idx}.tsv"
    if not os.path.exists(kegg_path):
        return
    
    kegg_df = pd.read_csv(kegg_path, sep="\t")

    kegg_df["GeneRatio_float"] = kegg_df["GeneRatio"].apply(lambda x: eval(x))
    top_kegg_df = kegg_df.sort_values(by=["GeneRatio_float", "pvalue"], ascending=[False, True]).head(3)

    # 映射信息加载
    id_mapping_df = pd.read_csv("protein_info/entrezgene_id_and_symbol_converted.tsv", sep="\t")
    id_mapping_df["ENTREZID"] = id_mapping_df["ENTREZID"].astype(str).str.strip()
    entrez_to_ensembl = id_mapping_df.set_index("ENTREZID")["ENSEMBL_ID"].to_dict()
    ensembl_to_symbol = id_mapping_df.set_index("ENSEMBL_ID")["SYMBOL"].to_dict()
    ensembl_to_node_index = {v: k for k, v in protein_index_to_name.items()}

    # 通路节点提取
    pathway_node_map = {}
    pathway_color_map = {}
    color_palette = ['#ffcc00', '#00ccff', '#cc66ff']
    legend_handles = []

    for i, (_, row) in enumerate(top_kegg_df.iterrows()):
        entrez_list = row['geneID'].split('/')
        pid = row['ID']
        pname = row['Description']
        color = color_palette[i % len(color_palette)]
        pathway_color_map[pid] = color
        legend_handles.append(mpatches.Patch(color=color, label=pname))
        node_set = set()
        for entrez in entrez_list:
            ensembl = entrez_to_ensembl.get(str(entrez))
            if ensembl and ensembl in ensembl_to_node_index:
                node_set.add(ensembl_to_node_index[ensembl])
        pathway_node_map[pid] = node_set
    
    # 从解释器获取边和边权重
    edge_index = explanation.edge_index.cpu()
    edge_mask = explanation.edge_mask.cpu() if explanation.edge_mask is not None else None
    if edge_mask is not None:
        edge_mask = edge_mask - edge_mask.min()
        edge_mask = edge_mask / (edge_mask.max() + 1e-12)
        mask = edge_mask > 1e-7
        edge_index = edge_index[:, mask]
        edge_mask = edge_mask[mask]
    else:
        edge_mask = torch.ones(edge_index.size(1))

    global_node_indices = torch.unique(edge_index).tolist()
    if not global_node_indices:
        print(f"Skipping empty subgraph for node {idx}.")
        return

    # 构图
    G = nx.DiGraph() if explanation.is_directed() else nx.Graph()
    for node in global_node_indices:
        G.add_node(node)
    for (src, dst), w in zip(edge_index.t().tolist(), edge_mask.tolist()):
        G.add_edge(src, dst, alpha=float(w))

    # 节点颜色和边框颜色设置
    node_fill_colors = []
    node_border_colors = []
    for node in G.nodes():
        # 填充颜色（通路）
        fill_color = false_color
        for pid, nodes in pathway_node_map.items():
            if node in nodes:
                fill_color = pathway_color_map[pid]
                break
        if node == idx:
            fill_color = central_node_color

        # 边框颜色（真假）
        border_color = true_color if true_labels[node] == 1 else 'black'

        node_fill_colors.append(fill_color)
        node_border_colors.append(border_color)

    # 节点标签（Symbol）
    node_labels = {
        node: ensembl_to_symbol.get(protein_index_to_name.get(node, ""), f"Unknown_{node}")
        for node in G.nodes()
    }

    # 布局
    if layout == 'spring':
        pos = nx.spring_layout(G, k=1/(len(G.nodes())**0.5 + 1e-3), iterations=200)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G)

    # 绘图开始
    fig, ax = plt.subplots(figsize=(20, 16))
    plt.axis('off')

    # 绘制节点
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_fill_colors,
        edgecolors=node_border_colors,  # 边框颜色
        node_size=node_size,
        linewidths=1.5,
        ax=ax
    )

    # 边颜色/宽度
    cmap = plt.cm.get_cmap(edge_colormap)
    edge_colors, edge_widths, edge_alpha = [], [], []
    for (src, dst), w in zip(edge_index.t().tolist(), edge_mask.tolist()):
        width = 0.8
        alpha = w
        color = cmap(w)
        for pid, node_set in pathway_node_map.items():
            if src in node_set and dst in node_set:
                color = pathway_color_map[pid]
                width = 2.0
                alpha = 1.0
                break
        edge_colors.append(color)
        edge_widths.append(width)
        edge_alpha.append(alpha)

    # 绘制边
    if len(G.edges()) > 0:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=G.edges(),
            width=edge_widths,
            alpha=edge_alpha,
            edge_color=edge_colors,
            ax=ax
        )

    # 绘制标签
    nx.draw_networkx_labels(
        G, pos,
        labels=node_labels,
        font_size=font_size,
        verticalalignment='bottom',
        ax=ax
    )

    # 添加边权重图例
    if edge_mask is not None and len(edge_mask) > 0:
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.get_cmap(edge_colormap),
            norm=plt.Normalize(vmin=0, vmax=1)
        )
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Edge Importance', shrink=0.8)

    # 添加图例
    ax.legend(handles=legend_handles, title="Top 3 KEGG Pathways", loc='lower left', fontsize=10, title_fontsize=11)
    nx.write_graphml(G, f"{save_path}\graph.graphml")
    plt.title(f'Subgraph with Pathway for Node {idx}\n({node_labels.get(idx, "Unknown")})', fontsize=12)
    plt.savefig(save_png_path, bbox_inches='tight', dpi=dpi)
    plt.close()



def draw_AMI_barplot():
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv('result_save/gnn_2025-03-14-18-49-10/GNNExplainer/ami_scores_pro_nodes.csv')
    convert_id=pd.read_csv('protein_info\entrezgene_id_and_symbol_converted.tsv',sep='\t')
    convert_id['ENTREZID'] = convert_id['ENTREZID'].astype(str)
    df['ami'] = df['ami'].astype(float)
    #df = df.sort_values('ami', ascending=True)  # 排序让高分在图上方
    #df = df.tail(10)
    df = df.head(10)[::-1]
    df = df.merge(convert_id[['ENSEMBL_ID', 'SYMBOL']], left_on='Protein_Name', right_on='ENSEMBL_ID', how='left')
    

    plt.figure(figsize=(10, 6))

    # 画水平条形图
    ax = df.plot(
        kind='barh',
        y='ami',
        x='SYMBOL',
        color='#2c7bb6',
        edgecolor='black',
        linewidth=0.7,
        legend=False
    )

    # 图表标题 & 标签
    plt.title(f'AMI Score Ranking By DMI Head 10', fontsize=14, pad=20)
    plt.xlabel('AMI Score', fontsize=12)
    plt.ylabel('Protein Name', fontsize=12)

    # 添加数值标签（显示在条形右侧）
    for i, (score, name_) in enumerate(zip(df['ami'], df['SYMBOL'])):
        ax.text(score + 0.001, i, f'{score:.4f}', va='center', ha='left', fontsize=9)

    # 美化边框
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')

    ax.grid(axis='x', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f'result_save/gnn_2025-03-14-18-49-10/GNNExplainer/AMI_not_sorted.png', dpi=300, bbox_inches='tight')
    plt.close()
draw_AMI_barplot()