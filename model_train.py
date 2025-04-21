import os
import time
import math
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
from gnn_data import GNN_DATA
# from gnn_models_sag import GIN_Net2, ppi_model
from gnn_models_sag import ppi_model
from utils import Metrictor_PPI, print_file
from tensorboardX import SummaryWriter
from draw_ppi import plot_metrics,draw_ppi_graph_with_predictions
parser = argparse.ArgumentParser(description='HIGH-PPI_model_training')

parser.add_argument('--ppi_path', default=None, type=str,
                    help="ppi path")
parser.add_argument('--pseq_path', default=None, type=str,
                    help="protein sequence path")
parser.add_argument('--vec_path', default='./protein_info/vec5_CTC.txt', type=str,
                    help='protein sequence vector path')
parser.add_argument('--p_feat_matrix', default=None, type=str,
                    help="protein feature matrix")
parser.add_argument('--p_adj_matrix', default=None, type=str,
                    help="protein adjacency matrix")
parser.add_argument('--split', default=None, type=str,
                    help='split method, random, bfs or dfs')
parser.add_argument('--save_path', default=None, type=str,
                    help="save folder")
parser.add_argument('--epoch_num', default=None, type=int,
                    help='train epoch number')
seed_num = 2
np.random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)

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


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def train(batch, p_x_all, p_edge_all, model, graph, ppi_list, loss_fn, optimizer, device,
          result_file_path, summary_writer, save_path,
          batch_size=32, epochs=1000, scheduler=None,scheduler_cosine=None,edge_type=None):
    global_step = 0
    global_best_valid_f1 = 0.0
    global_best_valid_f1_epoch = 0

    epoch_metrics = {
        "train_loss": [],
        "train_f1": [],
        "train_recall": [],
        "train_precision": [],
        "valid_loss": [],
        "valid_f1": [],
        "valid_recall": [],
        "valid_precision": [],
    }

    for epoch in range(epochs):

        recall_sum = 0.0
        precision_sum = 0.0
        f1_sum = 0.0
        loss_sum = 0.0

        steps = math.ceil(len(graph.train_mask) / batch_size)

        model.train()

        random.shuffle(graph.train_mask)

        for step in range(steps):
            if step == steps - 1:
                    train_node_id = graph.train_mask[step * batch_size:]
            else:
                    train_node_id = graph.train_mask[step * batch_size: step * batch_size + batch_size]

            output = model(batch, p_x_all, p_edge_all, graph.edge_index,edge_type, train_node_id)
            label = graph.y[train_node_id].type(torch.FloatTensor).to(device)
            
            #output = model(batch, p_x_all, p_edge_all, graph.edge_index,train_node_id)
            #print(f"Model output: {output[:20]}")   
             

            loss = loss_fn(output,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m = nn.Sigmoid()
            pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

            metrics = Metrictor_PPI(pre_result.cpu().data, label.cpu().data, m(output).cpu().data,save_path,True)
            metrics.show_result()

            recall_sum += metrics.Recall
            precision_sum += metrics.Precision
            f1_sum += metrics.F1
            loss_sum += loss.item()

            summary_writer.add_scalar('train/loss', loss.item(), global_step)
            summary_writer.add_scalar('train/precision', metrics.Precision, global_step)
            summary_writer.add_scalar('train/recall', metrics.Recall, global_step)
            summary_writer.add_scalar('train/F1', metrics.F1, global_step)

            global_step += 1

        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict()},
                   os.path.join(save_path, 'gnn_model_train.ckpt'))
        if scheduler_cosine!=None:
            scheduler_cosine.step()

        valid_pre_result_list = []
        valid_label_list = []
        true_prob_list = []
        valid_loss_sum = 0.0

        model.eval()

        valid_steps = math.ceil(len(graph.val_mask) / batch_size)

        with torch.no_grad():
            for step in range(valid_steps):
                if step == valid_steps - 1:
                    valid_node_id = graph.val_mask[step * batch_size:]
                else:
                    valid_node_id = graph.val_mask[step * batch_size: step * batch_size + batch_size]

                output = model(batch, p_x_all, p_edge_all, graph.edge_index,edge_type,valid_node_id)
                label = graph.y[valid_node_id].type(torch.FloatTensor).to(device)

                loss = loss_fn(output, label)
                valid_loss_sum += loss.item()

                m = nn.Sigmoid()
                pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

                valid_pre_result_list.append(pre_result.cpu().data)
                valid_label_list.append(label.cpu().data)
                true_prob_list.append(m(output).cpu().data)

        valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
        valid_label_list = torch.cat(valid_label_list, dim=0)
        true_prob_list = torch.cat(true_prob_list, dim=0)

        metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list, true_prob_list,save_path)
        metrics.show_result()

        recall = recall_sum / steps
        precision = precision_sum / steps
        f1 = f1_sum / steps
        loss = loss_sum / steps

        valid_loss = valid_loss_sum / valid_steps

        if scheduler is not None:
            scheduler.step(loss)
            print_file("epoch: {}, now learning rate: {}".format(epoch, scheduler.optimizer.param_groups[0]['lr']),
                       save_file_path=result_file_path)

        if global_best_valid_f1 < metrics.F1:
            global_best_valid_f1 = metrics.F1
            global_best_valid_f1_epoch = epoch
            best_model_state = model.state_dict()

            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict()},
                       os.path.join(save_path, 'gnn_model_valid_best.ckpt'))
        else:
            best_model_state=model.state_dict()
        torch.save(best_model_state, f"{save_path}/gnn_model_with_structure.pth")
        summary_writer.add_scalar('valid/precision', metrics.Precision, global_step)
        summary_writer.add_scalar('valid/recall', metrics.Recall, global_step)
        summary_writer.add_scalar('valid/F1', metrics.F1, global_step)
        summary_writer.add_scalar('valid/loss', valid_loss, global_step)

        print_file(
            f"epoch: {epoch}, Training_avg: label_loss: {loss}, recall: {recall}, precision: {precision}, F1: {f1}, "
            f"Validation_avg: loss: {valid_loss}, recall: {metrics.Recall}, precision: {metrics.Precision}, F1: {metrics.F1}, "
            f"Best valid_f1: {global_best_valid_f1}, in {global_best_valid_f1_epoch} epoch", 
            save_file_path=result_file_path)
        
        epoch_metrics["train_loss"].append(loss)
        epoch_metrics["train_f1"].append(f1)
        epoch_metrics["train_recall"].append(recall)
        epoch_metrics["train_precision"].append(precision)
        epoch_metrics["valid_loss"].append(valid_loss)
        epoch_metrics["valid_f1"].append(metrics.F1)
        epoch_metrics["valid_recall"].append(metrics.Recall)
        epoch_metrics["valid_precision"].append(metrics.Precision)
    
    return epoch_metrics,best_model_state

def main():
    args = parser.parse_args()
    ppi_data = GNN_DATA(ppi_path=args.ppi_path)
    ppi_data.get_feature_origin(pseq_path=args.pseq_path,
                                vec_path=args.vec_path)

    # load node labels
    #ppi_data.generate_random_labels()
    node_label_path = "target_info\drug_target_ALL.csv"
    ppi_data.generate_data_labels(node_label_path)

    ppi_data.generate_data()
    if args.split == 'k_fold': 
        ppi_data.split_dataset_kfold()
    else:
        ppi_data.split_dataset()
    
    graph = ppi_data.data
    ppi_list = ppi_data.ppi_list
    edge_type = torch.argmax(ppi_data.data.edge_attr_1, dim=1)

    graph.y = ppi_data.y
    num_positive_labels = (graph.y == 1).sum().item()
    print(f"Number of nodes with label 1: {num_positive_labels}")
    

    

    p_x_all = torch.load(args.p_feat_matrix)
    p_edge_all = np.load(args.p_adj_matrix, allow_pickle=True)
    p_x_all, x_num_index = multi2big_x(p_x_all)
    p_edge_all, edge_num_index = multi2big_edge(p_edge_all, x_num_index)
    batch = multi2big_batch(x_num_index) + 1

    print(f"train gnn, train_nodes: {len(graph.train_mask)}, valid_nodes: {len(graph.val_mask)}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    graph.to(device)

    model = ppi_model().to(device)
    lr=0.001
    weight_decay=5e-4
    factor=0.5
    patience=10
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    #scheduler_cosine=None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)
    #pos_weight = torch.tensor([0.7])
    loss_fn = nn.BCEWithLogitsLoss().to(device)

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)


    
    time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    if args.split=='k_fold':
        
        save_path = os.path.join(save_path, f"gnn_k_fold_{time_stamp}")
    else:
        save_path = os.path.join(save_path, f"gnn_{time_stamp}")
    os.mkdir(save_path)
    
    result_file_path = os.path.join(save_path, "valid_results.txt")
    #config_path = os.path.join(save_path, "config.txt")
    summary_writer = SummaryWriter(save_path)
    
    batch_size = 32
    kfold_results = []
    if args.split == 'k_fold':
        for fold_idx, fold in enumerate(ppi_data.kfold_split_dict["folds"]):
            print(f"Starting training for Fold {fold_idx + 1}/{args.split}")
            graph.train_mask = torch.tensor(fold["train_index"])
            graph.val_mask = torch.tensor(fold["valid_index"])
            epoch_metrics,best_model_state=train(batch, p_x_all, p_edge_all, model, graph, ppi_list, loss_fn, optimizer, device,
            result_file_path, summary_writer, save_path,
            batch_size=batch_size, epochs=args.epoch_num, scheduler=scheduler,scheduler_cosine=scheduler_cosine,edge_type=edge_type)
            kfold_results.append(epoch_metrics)

    else:
        graph.train_mask = ppi_data.ppi_split_dict['train_index']
        graph.val_mask = ppi_data.ppi_split_dict['valid_index']
        epoch_metrics,best_model_state=train(batch, p_x_all, p_edge_all, model, graph, ppi_list, loss_fn, optimizer, device,
            result_file_path, summary_writer, save_path,
            batch_size=batch_size, epochs=args.epoch_num, scheduler=scheduler,scheduler_cosine=scheduler_cosine,edge_type=edge_type)
    if args.split == 'k_fold':
        f1_scores = [result["f1"] for result in kfold_results]
        recall_scores = [result["recall"] for result in kfold_results]
        precision_scores = [result["precision"] for result in kfold_results]
        loss_scores = [result["loss"] for result in kfold_results]

        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        mean_recall = np.mean(recall_scores)
        mean_precision = np.mean(precision_scores)
        mean_loss = np.mean(loss_scores)

        print(f"K-Fold Validation Results:")
        print(f"F1 Score: Mean = {mean_f1:.4f}, Std = {std_f1:.4f}")
        print(f"Precision: Mean = {mean_precision:.4f}")
        print(f"Recall: Mean = {mean_recall:.4f}")
        print(f"Loss: Mean = {mean_loss:.4f}")

        with open(os.path.join(save_path, "kfold_summary.txt"), "w") as f:
            f.write(f"K-Fold Validation Results:\n")
            f.write(f"F1 Score: Mean = {mean_f1:.4f}, Std = {std_f1:.4f}\n")
            f.write(f"Precision: Mean = {mean_precision:.4f}\n")
            f.write(f"Recall: Mean = {mean_recall:.4f}\n")
            f.write(f"Loss: Mean = {mean_loss:.4f}\n")
    save_path = os.path.join(save_path, "png")
    plot_metrics(epoch_metrics,save_path)

    summary_writer.close()

    model.load_state_dict(best_model_state)
    model.eval()

    with torch.no_grad():
        output = model(batch, p_x_all, p_edge_all, graph.edge_index,edge_type)
        pred_labels = (torch.sigmoid(output) > 0.5).int().cpu().data.flatten()
        true_labels = graph.y.cpu().data.flatten()
    #print(f"Shape of pred_labels: {pred_labels.shape}, Sample: {pred_labels[:10]},class:{type(pred_labels)}")
    #print(f"Shape of true_labels: {true_labels.shape}, Sample: {true_labels[:10]},class:{type(true_labels)}")
    
    draw_ppi_graph_with_predictions(
        edge_index=graph.edge_index,
        node_labels=true_labels,
        pred_labels=pred_labels,
        save_path=os.path.join(save_path, f"ppi_graph.png")
    )
    with open(result_file_path, "a") as file:
        file.write(f"lr:{lr},weight_decay:{weight_decay},factor:{factor},patience;{patience}")


if __name__ == "__main__":
    main()
