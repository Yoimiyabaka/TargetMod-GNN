import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
# from pygcn.layers import GraphConvolution
# from dgl.nn import GraphConv, EdgeWeightNorm
from torch_geometric.nn import GINConv, JumpingKnowledge, global_mean_pool, SAGEConv, GCNConv,RGCNConv
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn import global_mean_pool



class GIN(torch.nn.Module):
    def __init__(self,  hidden=128, train_eps=True, class_num=1):
        super(GIN, self).__init__()
        self.train_eps = train_eps
        self.gin_conv1 = GINConv(
            nn.Sequential(
                nn.Linear(64, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                #nn.Linear(hidden, hidden),
                #nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_conv3 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_conv4 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        #self.bn1=nn.BatchNorm1d(hidden)
        self.lin1 = nn.Linear(hidden, hidden)
        self.fc = nn.Linear(hidden, class_num) #clasifier for node
 



    def reset_parameters(self):

        self.gin_conv1.reset_parameters()
        self.gin_conv2.reset_parameters()
        #self.gin_conv3.reset_parameters()
        #self.gin_conv4.reset_parameters()
        self.lin1.reset_parameters()
        self.fc.reset_parameters()
        


    def forward(self, x, edge_index,train_node_id=None,p=0.5):
        x = self.gin_conv1(x, edge_index)
        x = self.gin_conv2(x, edge_index)
        x = self.gin_conv3(x, edge_index)
        #x = self.gin_conv4(x, edge_index)
        x = F.relu(self.lin1(x))
        #x = self.bn1(x)
        #p = 0.5 if self.training else 0.0
        x = F.dropout(x, p=0.5, training=self.training)
        if train_node_id !=None:
            x = x[train_node_id]
        x = self.fc(x).squeeze(-1)  
        return x


class TGCNConv(torch.nn.Module):
    def __init__(self, hidden=256, class_num=1):
        super(TGCNConv, self).__init__()
        self.conv1 = GCNConv(128, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.conv4 = GCNConv(hidden, hidden)

        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.bn4 = nn.BatchNorm1d(hidden)

        self.lin1 = nn.Linear(hidden, hidden)
        self.fc = nn.Linear(hidden, class_num)  

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.lin1.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, x, edge_index, train_node_id=None, p=0.5):
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        #x = self.sage_conv4(x, edge_index)
        #x = F.relu(x)
        x = F.relu(self.lin1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = x[train_node_id]
        x = self.fc(x).squeeze(-1)
        return x

class RGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super(RGCN, self).__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations)
        #self.conv3 = RGCNConv(hidden_channels, out_channels, num_relations)
        #self.bn1 = nn.BatchNorm1d(hidden_channels)
        #self.bn2 = nn.BatchNorm1d(out_channels)
        self.fc = nn.Linear(out_channels, 1) 

    def forward(self, x, edge_index, edge_type,train_node_id=None,p=0.5):
        x = self.conv1(x, edge_index, edge_type)
        #x = self.bn1(x)
        x = F.relu(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index, edge_type)
        #x = self.bn2(x)
        x = F.relu(x)
        
        #x = F.dropout(x, p=0.5, training=self.training)
        #x = self.conv3(x, edge_index, edge_type)
        #x = self.bn2(x)
        #x = F.relu(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        
        x = x[train_node_id]
        return self.fc(x).squeeze(-1)



class GCN(nn.Module):
    def __init__(self,hidden=64):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(7, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.conv4 = GCNConv(hidden, hidden)
  
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.bn4 = nn.BatchNorm1d(hidden)

        self.sag1 = SAGPooling(hidden,0.5)
        self.sag2 = SAGPooling(hidden,0.5)
        self.sag3 = SAGPooling(hidden,0.5)
        self.sag4 = SAGPooling(hidden,0.5)

        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)

        self.dropout = nn.Dropout(0.5)

    
    '''
    def reset_parameters(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4]:
            nn.init.xavier_uniform_(layer.weight)
    '''

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.fc1(x)
        x = F.relu(x) 
        x = self.bn1(x)
        #消融实验：去除图注意力机制
      
        y = self.sag1(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]
        
        #x = global_mean_pool(x, batch) #消融实验
        
        x = self.conv2(x, edge_index)
        x = self.fc2(x)
        x = F.relu(x) 
        x = self.bn2(x)
        y = self.sag2(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]
        
        
        x = self.conv3(x, edge_index)
        x = self.fc3(x)
        x = F.relu(x) 
        x = self.bn3(x)
        y = self.sag3(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        
        x = self.conv4(x, edge_index)
        x = self.fc4(x)
        x = F.relu(x) 
        x = self.bn4(x)
        y = self.sag4(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]
        
        return global_mean_pool(y[0], y[3])#x消融实验


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class ppi_model(nn.Module):
    def __init__(self):
        super(ppi_model,self).__init__()
        self.BGNN = GCN()
        #self.TGNN = GIN()
        self.TGNN = RGCN(in_channels=64, hidden_channels=128, out_channels=64, num_relations=7)

    def forward(self,batch,p_x_all, p_edge_all, edge_index,edge_type=None,train_node_id=None):
        edge_index = edge_index.to(torch.int64).to(device)
        batch = batch.to(torch.int64).to(device)
        x = p_x_all.to(torch.float32).to(device)
        edge = torch.LongTensor(p_edge_all.to(torch.int64)).to(device)
        embs = self.BGNN(x, edge, batch-1)
        #RGCN
        #final = self.TGNN(embs, edge_index,edge_type=edge_type,train_node_id=train_node_id,p=0.5)
        #GIN
        final = self.TGNN(embs, edge_index,train_node_id=train_node_id,p=0.5)
        return final

class ppi_model_explain(nn.Module):
    def __init__(self):
        super(ppi_model_explain,self).__init__()
        self.TGNN = GIN()
        #self.TGNN = RGCN(in_channels=64, hidden_channels=128, out_channels=64, num_relations=7)

    def forward(self,x,edge_index):
        x=x.to(torch.float32).to(device)
        edge_index = edge_index.to(torch.int64).to(device)
        train_node_id=None
        final = self.TGNN(x, edge_index,train_node_id=train_node_id,p=0.5)
        return torch.sigmoid(final)
    
class ppi_model_got_embs(nn.Module):
    def __init__(self):
        super(ppi_model_got_embs,self).__init__()
        self.BGNN = GCN()
        self.TGNN = GIN()
        #self.TGNN = RGCN(in_channels=64, hidden_channels=128, out_channels=64, num_relations=7)

    def forward(self,batch,p_x_all, edge_index, p_edge_all,edge_type=None,train_node_id=None):
        edge_index = edge_index.to(torch.int64).to(device)
        batch = batch.to(torch.int64).to(device)
        x = p_x_all.to(torch.float32).to(device)
        edge = torch.LongTensor(p_edge_all.to(torch.int64)).to(device)
        embs = self.BGNN(x, edge, batch-1)
        #RGCN
        #final = self.TGNN(embs, edge_index,edge_type=edge_type,train_node_id=train_node_id,p=0.5)
        #GIN
        final = self.TGNN(embs, edge_index,train_node_id=train_node_id,p=0.5)
        return final,embs