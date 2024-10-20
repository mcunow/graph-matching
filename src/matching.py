import torch
import torch.nn as nn
import torch.nn.functional as F
import ot
import numpy as np
from torch_geometric.data import Data,Batch
import copy
from torch_geometric.utils import to_dense_batch, scatter, to_dense_adj,subgraph, to_undirected, add_self_loops,to_torch_coo_tensor
from scipy.optimize import linear_sum_assignment
import itertools

def generate_permutation_matrices(n):
    # Generate all permutations of the numbers [0, 1, ..., n-1]
    permutations = list(itertools.permutations(range(n)))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Stack permutation matrices into a single tensor
    permutation_matrices = torch.stack([
        torch.eye(n)[torch.tensor(perm)] for perm in permutations
    ])
    return permutation_matrices.to(device)

def to_dense(data,with_mask=False):
    x,mask=to_dense_batch(data.x,batch=data.batch)
    adj=to_dense_adj(data.edge_index,edge_attr=data.edge_attr,batch=data.batch)
    if with_mask:
        return x,adj,mask
    else:
        return x,adj

class NaiveMatcher():
    def __init__(self):
        # Precompute permutation matrices for sizes 1 to 9
        self.permutations_matrices = [generate_permutation_matrices(i) for i in range(1, 10)]
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def diffusion_matching(self, in_data, rec_data,test):
        #temp=to_torch_coo_tensor(in_data.edge_index,in_data.edge_attr).values()
        in_x, in_adj, mask = to_dense(in_data, with_mask=True)
        rec_x, rec_adj = to_dense(rec_data)
        B, n, _ = in_x.size()

        batch=[]
        unmatched_batch=[]

        for i in range(B):
            num_nodes = torch.sum(mask[i]).item()    
            in_x_temp=in_x[i][mask[i]]
            in_adj_temp=in_adj[i][mask[i]][:,mask[i]]
            rec_x_temp = rec_x[i][mask[i]]
            rec_adj_temp = rec_adj[i][mask[i]][:, mask[i]]
            zeros = torch.ones(( num_nodes, num_nodes, 1)).to(self.device)
            temp=in_adj_temp.sum(dim=2).unsqueeze(-1)
            zeros = zeros-temp
            in_adj_temp=torch.cat((zeros,in_adj_temp),dim=2)
            indices = torch.arange(num_nodes).to(self.device)
            in_adj_temp[indices, indices] = 0

            triu_mask=torch.triu(torch.ones((num_nodes,num_nodes),dtype=bool),1).to(self.device)

            f_loss=(rec_x_temp-in_x_temp)**2
            adj_loss=(rec_adj_temp[triu_mask]-in_adj_temp[triu_mask])**2
            temp_loss=torch.cat((f_loss.sum(dim=-1),adj_loss.sum(dim=-1)),dim=0)
            temp_loss=temp_loss.mean().unsqueeze(0)
            unmatched_batch.append(temp_loss)
            
            if num_nodes == 1:
                batch.append(temp_loss)
                continue  # Skip to the next batch item
            if test:
                batch.append(temp_loss)
                continue

            # Gather valid permutations for the current number of nodes
            perm_matrices = self.permutations_matrices[num_nodes-1]

            sample_perc=0.1
            if perm_matrices.size(0)>6:
                num_elements = int(sample_perc * perm_matrices.size(0))
                indices = torch.randperm(perm_matrices.size(0),device=self.device)[:num_elements]
                perm_matrices=perm_matrices[indices]

            
            # Apply all permutations at once using vectorized operations
            reordered_x = torch.einsum("pij,jk->pik", perm_matrices, rec_x_temp)
            reordered_adj = torch.einsum("bij,jle,bkl->bike", perm_matrices, rec_adj_temp, perm_matrices)

            #TODO Write test for einsum
            reordered_adj=reordered_adj[:,triu_mask]
            in_adj_temp=in_adj_temp[triu_mask]
            with torch.no_grad():
                f_loss=(reordered_x-in_x_temp)**2
                adj_loss=(reordered_adj-in_adj_temp)**2
                f_loss=f_loss.sum(dim=2)
                adj_loss=adj_loss.sum(dim=2)
                temp_loss=torch.cat((f_loss,adj_loss),dim=1)
                temp_loss=temp_loss.mean(1)
                _,min_idx = torch.min(temp_loss,dim=0)

            f_loss=(reordered_x[min_idx]-in_x_temp)**2
            adj_loss=(reordered_adj[min_idx]-in_adj_temp)**2
            temp_loss=torch.cat((f_loss.sum(1),adj_loss.sum(1)),dim=0)

            batch.append(temp_loss.mean().unsqueeze(0))    

        batch=torch.cat(batch)
        unmatched_batch=torch.cat(unmatched_batch)
        assert(batch.size()==unmatched_batch.size())

            
        return torch.mean(batch),torch.mean(unmatched_batch)