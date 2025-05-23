from sklearn.metrics import roc_curve, auc
import os
import numpy as np
import random
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt


def print_file(str_, save_file_path=None):
    print(str_)
    if save_file_path != None:
        f = open(save_file_path, 'a')
        print(str_, file=f)


class Metrictor_PPI:
    def __init__(self, pre_y, truth_y, true_prob, save_path,is_binary=True):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        
        self.save_path=f"{save_path}/png"
        
        self.pre = np.array(pre_y).squeeze()
        self.tru = np.array(truth_y).squeeze()
        self.true_prob = np.array(true_prob).squeeze()

        if is_binary:
            
            length = len(self.pre)
            for i in range(length):
                if self.pre[i] == self.tru[i]:
                    if self.tru[i] == 1:
                        self.TP += 1
                    else:
                        self.TN += 1
                elif self.tru[i] == 1:
                    self.FN += 1
                elif self.pre[i] == 1:
                    self.FP += 1
            self.num = length
        else:
            
            if len(pre_y.shape) == 1:  
                pre_y = np.expand_dims(pre_y, axis=-1)  
            N, C = pre_y.shape  
            for i in range(N):
                for j in range(C):
                    if pre_y[i][j] == truth_y[i][j]:
                        if truth_y[i][j] == 1:
                            self.TP += 1
                        else:
                            self.TN += 1
                    elif truth_y[i][j] == 1:
                        self.FN += 1
                    elif truth_y[i][j] == 0:
                        self.FP += 1
            self.num = N * C

    def show_result(self, is_print=False, file=None):
        
        self.Accuracy = (self.TP + self.TN) / (self.num + 1e-10)
        self.Precision = self.TP / (self.TP + self.FP + 1e-10) 
        self.Recall = self.TP / (self.TP + self.FN + 1e-10)
        self.F1 = 2 * self.Precision * self.Recall / (self.Precision + self.Recall + 1e-10)

        aupr_entry_1 = self.tru
        aupr_entry_2 = self.true_prob

        if aupr_entry_1.ndim == 1:
            aupr_entry_1 = aupr_entry_1[:, np.newaxis]
        if aupr_entry_2.ndim == 1:
            aupr_entry_2 = aupr_entry_2[:, np.newaxis]

        if aupr_entry_1.shape[1] == 1:  
            precision, recall, _ = precision_recall_curve(aupr_entry_1[:, 0], aupr_entry_2[:, 0])
            self.Aupr = auc(recall, precision)
        else: 
            aupr = np.zeros(aupr_entry_1.shape[1])
            for i in range(aupr_entry_1.shape[1]):
                precision, recall, _ = precision_recall_curve(aupr_entry_1[:, i], aupr_entry_2[:, i])
                aupr[i] = auc(recall, precision)
            self.Aupr = aupr

        if is_print:
            print_file(f"Accuracy: {self.Accuracy}", file)
            print_file(f"Precision: {self.Precision}", file)
            print_file(f"Recall: {self.Recall}", file)
            print_file(f"F1-Score: {self.F1}", file)
            print_file(f"AUPR: {self.Aupr}", file)
        
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        fpr, tpr, _ = roc_curve(self.tru, self.true_prob)
        self.AUC = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {self.AUC:.2f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig(os.path.join(self.save_path, "roc_curve.png"))
        plt.close()


class UnionFindSet(object):
    def __init__(self, m):
        # m, n = len(grid), len(grid[0])
        self.roots = [i for i in range(m)]
        self.rank = [0 for i in range(m)]
        self.count = m

        for i in range(m):
            self.roots[i] = i

    def find(self, member):
        tmp = []
        while member != self.roots[member]:
            tmp.append(member)
            member = self.roots[member]
        for root in tmp:
            self.roots[root] = member
        return member

    def union(self, p, q):
        parentP = self.find(p)
        parentQ = self.find(q)
        if parentP != parentQ:
            if self.rank[parentP] > self.rank[parentQ]:
                self.roots[parentQ] = parentP
            elif self.rank[parentP] < self.rank[parentQ]:
                self.roots[parentP] = parentQ
            else:
                self.roots[parentQ] = parentP
                self.rank[parentP] -= 1
            self.count -= 1


def get_bfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    candiate_node = []
    selected_edge_index = []
    selected_node = []

    random_node = random.randint(0, node_num - 1)
    while len(node_to_edge_index[random_node]) > 5:
        random_node = random.randint(0, node_num - 1)
    candiate_node.append(random_node)

    while len(selected_edge_index) < sub_graph_size:
        cur_node = candiate_node.pop(0)
        selected_node.append(cur_node)
        for edge_index in node_to_edge_index[cur_node]:

            if edge_index not in selected_edge_index:
                selected_edge_index.append(edge_index)

                end_node = -1
                if ppi_list[edge_index][0] == cur_node:
                    end_node = ppi_list[edge_index][1]
                else:
                    end_node = ppi_list[edge_index][0]

                if end_node not in selected_node and end_node not in candiate_node:
                    candiate_node.append(end_node)
            else:
                continue
        # print(len(selected_edge_index), len(candiate_node))
    node_list = candiate_node + selected_node
    # print(len(node_list), len(selected_edge_index))
    return selected_edge_index


def get_dfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    stack = []
    selected_edge_index = []
    selected_node = []

    random_node = random.randint(0, node_num - 1)
    while len(node_to_edge_index[random_node]) > 5:
        random_node = random.randint(0, node_num - 1)
    stack.append(random_node)

    while len(selected_edge_index) < sub_graph_size:
        # print(len(selected_edge_index), len(stack), len(selected_node))
        cur_node = stack[-1]
        if cur_node in selected_node:
            flag = True
            for edge_index in node_to_edge_index[cur_node]:
                if flag:
                    end_node = -1
                    if ppi_list[edge_index][0] == cur_node:
                        end_node = ppi_list[edge_index][1]
                    else:
                        end_node = ppi_list[edge_index][0]

                    if end_node in selected_node:
                        continue
                    else:
                        stack.append(end_node)
                        flag = False
                else:
                    break
            if flag:
                stack.pop()
            continue
        else:
            selected_node.append(cur_node)
            for edge_index in node_to_edge_index[cur_node]:
                if edge_index not in selected_edge_index:
                    selected_edge_index.append(edge_index)

    return selected_edge_index
