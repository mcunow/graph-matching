import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.models import GIN,PNA,GAT
from torch_geometric.utils import dense_to_sparse, add_self_loops

from torch_geometric.data import Data,Batch

class NodeNumPrediction(nn.Module):
    def __init__(self,global_dim):
        super(NodeNumPrediction,self).__init__()
        
        self.layers=nn.Sequential(
                    nn.Linear(global_dim,128),
                    nn.ReLU(),
                    nn.Linear(128,128),
                    nn.ReLU(),
                    nn.Linear(128,1)
        )


    def forward(self,global_vec):
        return self.layers(global_vec)
    
class GraphInitializer(nn.Module):
    def __init__(self,global_dim,hidden_dim):
        super(GraphInitializer,self).__init__()
        self.hidden_dim=hidden_dim

        self.layers=nn.Sequential(
            nn.Linear(global_dim+hidden_dim,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,16)
        )
    def forward(self,global_vec,num_nodes):
        return self.layers(global_vec)

class LaplacianInitializer():
    def __init__(self):
        ls=[]
        for i in range(1,10):

            # Compute the first few (e.g., 3) smallest non-zero eigenvectors of the Laplacian
            structural_features=self.compute_eigenvector(i)
            #structural_features=F.pad(structural_features,pad=(0,9-i),value=0)
            ls.append(structural_features)
        self.lap_vec=ls

    def compute_eigenvector(self,num_nodes):
        adj_matrix = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
        # Compute the graph Laplacian: L = D - A
        degree_matrix = np.diag(adj_matrix.sum(axis=1))
        laplacian_matrix = degree_matrix - adj_matrix

        _, eigenvectors = np.linalg.eig(laplacian_matrix)
        eigenvectors = np.real(eigenvectors)
        eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)
        structural_features = torch.tensor(eigenvectors, dtype=torch.float32)
        structural_features=F.pad(structural_features,(0,9-num_nodes),value=0)

        return structural_features
    
    def init_features(self,global_vec,num_node):
        index=num_node-1
        lap_eigvec=self.lap_vec[index]
        global_vector_expanded = global_vec.unsqueeze(0).repeat(num_node, 1)
        combined_input = torch.cat((global_vector_expanded, lap_eigvec), dim=1)
        return combined_input
    
    def init_batch_features(self,global_vecs,num_nodes):
        ls=[]
        for global_vec,num_node in zip(global_vecs,num_nodes):
            temp=self.init_features(global_vec,num_node)
            ls.append(temp)
        return torch.cat(ls,dim=0)
        


def fully_connected(feature_vec,num_nodes,batch):
    masks=torch.zeros((num_nodes.size(0),9),dtype=bool)
    indices = torch.arange(9).expand(num_nodes.size(0), -1)
    masks = indices < num_nodes.unsqueeze(1)
    adj=torch.ones(num_nodes.size(0),9,9)
    indices = torch.arange(9)
    adj[:, indices, indices] = 0
    for i in range(adj.shape[0]):
        adj[i][~masks[i]] = 0  # Update rows that are not masked
        adj[i][:, ~masks[i]] = 0 
        
    edge_index, _ = dense_to_sparse(adj,masks)

    return Batch(x=feature_vec,edge_index=edge_index,batch=batch)

class Decoder(nn.Module):
    def __init__(self,latent_dim):
        super(Decoder,self).__init__()
        
        self.num_net=NodeNumPrediction(latent_dim)
        self.lap_init=LaplacianInitializer()

        """
        self.layers=nn.Sequential(
            nn.Linear(global_dim+9,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,16)
        )
        """
        self.layers=nn.Linear(latent_dim,16) #Check how strong this layer needs to be
        self.layers2=nn.Linear(16+9,16)
        self.feature_layer=nn.Sequential(
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,4)
        )
        self.edge_layer=nn.Sequential(
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,5)
        )

        self.gnn=GIN(in_channels=16,hidden_channels=16,num_layers=4,out_channels=16,dropout=0.,act="leakyrelu",norm="graph")
        

    def forward(self,global_vec,data=None):
        pred_num=self.num_net(global_vec).view(-1)
        global_vec=F.relu(self.layers(global_vec))
    
        if data is not None:
            num_nodes=torch.bincount(data.batch) 
            node_vecs=self.lap_init.init_batch_features(global_vec,num_nodes)
            batch_idx=data.batch
        else:
            num_nodes=torch.clip(pred_num,min=1,max=9).to(int)
            node_vecs=self.lap_init.init_batch_features(global_vec,num_nodes)
            batch_idx=torch.cat([torch.tensor([i] * num_nodes[i]) for i in range(len(num_nodes))])
            
        node_vecs=self.layers2(node_vecs)
        batch=fully_connected(node_vecs,num_nodes,batch=batch_idx)
        batch.edge_index,_=add_self_loops(batch.edge_index)
        node_embeddings=self.gnn(batch.x,batch.edge_index,batch=batch.batch)

        edge_attr=(node_embeddings[batch.edge_index[0,:]]+node_embeddings[batch.edge_index[1,:]])/2
        edge_attr=self.edge_layer(edge_attr)

        node_embeddings=self.feature_layer(node_embeddings)

        return Batch(x=node_embeddings,edge_index=batch.edge_index,edge_attr=edge_attr,batch=batch_idx),pred_num

