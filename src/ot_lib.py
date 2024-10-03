import torch
import torch.nn as nn
import torch.nn.functional as F
import ot
import numpy as np
from torch_geometric.data import Data,Batch
import copy
from torch_geometric.utils import to_dense_batch, scatter, to_dense_adj,subgraph, to_undirected
import utils


class Sinkhorn_Loss():
    def __init__(self,encoder,sinkhorn_max_it=100,opt_method='sinkhorn_log',
                 sinkhorn_entropy=1):
        self.encoder=encoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sinkhorn_max_it=sinkhorn_max_it
        self.opt_method=opt_method
        self.sinkhorn_entropy=sinkhorn_entropy

    def set_encoder(self,encoder):
        self.encoder=copy.deepcopy(encoder)


    def embedd(self,data):
        embeddings=self.encoder(data)
        embedds,masks=to_dense_batch(embeddings,data.batch)
        return embedds,masks
    
    def soft_matching(self,e1,e2,mask1,mask2=None):
        cost_mat=torch.cdist(e1, e2, p=2)
        ls=[]
        ot_dist_ls=[]
        mask1=mask1.to(self.device)
        if mask2 is None:
            mask2=mask1
        mask2=mask2.to(self.device)
        for i,mat in enumerate(cost_mat):
            num=torch.max(torch.sum(mask1[i]),torch.sum(mask2[i]))
            temp_mat=mat[mask1[i]][:,mask2[i]]
            H1 = torch.full((torch.sum(mask1[i]),), 1, device=self.device) / torch.sum(mask1[i])
            H2 = torch.full((torch.sum(mask2[i]),), 1, device=self.device) / torch.sum(mask2[i])
            ot_plan=ot.bregman.sinkhorn_log(a=H1, b=H2, M=temp_mat,stopThr=1e-5,
                    numItermax=self.sinkhorn_max_it,
                    reg=self.sinkhorn_entropy)
            assert(~torch.isnan(ot_plan).any())
            ot_dist_ls.append(torch.sum(ot_plan*temp_mat))     
            ot_plan=ot_plan*torch.sum(mask1[i])
            ot_plan=torch.round(ot_plan,decimals=2)
            pad_size = e1.size(1) - num if e1.size(1)>e2.size(1) else e2.size(1)-num
            ot_plan=F.pad(ot_plan,pad=(0,pad_size,0,pad_size),value=0)
            ls.append(ot_plan)
        ot_plan=torch.stack(ls).to(torch.float)
        ot_dist=torch.stack(ot_dist_ls)

        return ot_plan,ot_dist


    def sinkhorn_soft(self,in_data,rec_data,loss,hard=False,scat=False,soft=False):
        
        num_nodes=4 
        #Transform logits into probability for matching (not for loss)
        rec_data_sm=Data(F.softmax(rec_data.x,dim=1)[:,1:],rec_data.edge_index,edge_attr=F.softmax(rec_data.edge_attr,dim=1),batch=rec_data.batch)
        in_x,in_adj=self.to_dense(in_data)
        rec_x,rec_adj= self.to_dense(Data(F.softmax(rec_data.x[:,1:],dim=1),rec_data.edge_index,batch=rec_data.batch,
                                    edge_attr=F.softmax(rec_data.edge_attr,dim=1)))
      
        edge_mask=torch.argmax(rec_data_sm.edge_attr,dim=1)!=0
        edge_index=rec_data_sm.edge_index[:,edge_mask]
        attr=rec_data_sm.edge_attr[edge_mask][:,1:]
        rec_data_sm=Data(rec_data_sm.x,edge_index,edge_attr=attr,batch=rec_data_sm.batch)

        #Embedd input and reconstructed graph with auxiliary GNN
        initial_x,masks=self.embedd(in_data)

        #Use probability or one-hot encoding for the embedding used for matching
        rec_data_sm= rec_data_sm if not hard else self.one_hot_encode(rec_data_sm,num_nodes)
        rec_embedd,_=self.embedd(rec_data_sm)

        #Compute and stack OT plan; Wasserstein distance can also be used for the loss
        #Each OT plan is rescaled to [0,1] by constant num_nodes to form permutation matrix. Shape (b,max_node,max_node)
        correspondence_matrix,ot_dists=self.soft_matching(initial_x,rec_embedd,masks)
        correspondence_matrix=self._max_h(correspondence_matrix,masks) if not soft else correspondence_matrix
        
        #Get GT and logits of edges and features
        #Shape (b,max_node,max_node,5) with five edge types (None, Single,Aromatic, Double, Triple)
        rec_x_prob=F.softmax(rec_data.x,dim=1)
        input=rec_x_prob[:,0]
        input=torch.clamp(input,0,1)
        tgt=torch.cat([torch.tensor([0],dtype=torch.float) for _ in range(rec_x_prob.size(0))])
        non_node_loss=F.binary_cross_entropy(input, tgt,reduction="none")
        non_node_loss=non_node_loss.sum()
        

        tau=0.1
        rec_data_sm=Data(F.softmax(rec_data.x,dim=1)[:,1:],rec_data.edge_index,edge_attr=F.softmax(rec_data.edge_attr/tau,dim=1)[:,1:],batch=rec_data.batch)
        if scat:
            reordered_x=torch.einsum("bij,bjk->bik",correspondence_matrix,rec_x)
            masks=masks.reshape((-1,))
            reordered_x=reordered_x.reshape((-1,4))[masks]
            in_x=in_x.reshape((-1,4))[masks]    
            scat1=scatter(in_data.edge_attr,index=in_data.edge_index[0],dim=0,dim_size=in_data.x.size(0))
            #rec_attr=F.gumbel_softmax(rec_data_sm.edge_attr,dim=1,hard=True)
            scat2=scatter(rec_data_sm.edge_attr,index=rec_data_sm.edge_index[0],dim=0,dim_size=rec_data_sm.x.size(0))
            dense_scat1,_=to_dense_batch(scat1,batch=in_data.batch)
            dense_scat2,_=to_dense_batch(scat2,batch=rec_data.batch)

            reordered_scat=torch.einsum("bij,bjk->bik",correspondence_matrix,dense_scat2)
            feat_weight=torch.tensor([1,3,4,10],dtype=torch.float)
            edge_weight=torch.tensor([1,3,3,5],dtype=torch.float)
            
            edge_loss=self.loss(dense_scat1,reordered_scat,loss="MSE",weight=edge_weight)
            if loss=="CE":
                idx=torch.argmax(in_data.x,dim=1)
                reordered_x=torch.log(reordered_x+1e-15)
                feature_loss=F.nll_loss(reordered_x,idx,weight=feat_weight,reduction="none")
                feature_loss=feature_loss.sum()
            else:
                feature_loss=self.loss(in_data.x,reordered_x,loss="MSE",weight=feat_weight)  
        else:
            reordered_x,reordered_adj=self.graph_mapping(correspondence_matrix,rec_x,F.softmax(rec_adj,dim=-1))
            adj1,reordered_adj=self.flat_adjacency(in_adj,reordered_adj,pad_mask=masks)
            feat1,reordered_x=self.flat_features(rec_x,reordered_x,masks)
            feature_loss=self.loss(feat1,reordered_x,loss=loss)
            adj1=self.extend_flatadj(adj1)
            edge_loss=self.loss(adj1,reordered_adj,loss=loss)

        return (non_node_loss+ feature_loss)/rec_data.num_graphs,edge_loss/rec_data.num_graphs,torch.mean(ot_dists)

        
    def scatter_mapping(self,ot_plan,data):
        feat,_,mask=self.to_dense(data,with_mask=True)
        reordered_x=torch.einsum("bij,bjk->bik",ot_plan,feat)
        scat=scatter(data.edge_attr,index=data.edge_index[0],dim=0)
        dense_scat,_=to_dense_batch(scat,batch=data.batch)
        reordered_scat=torch.einsum("bij,bjk->bik",ot_plan,dense_scat)
        return reordered_x,reordered_scat
    
    def graph_matching_loss(self,data1,data2,loss,real,scat_loss):
        ot_plan,ot_dist=self.compare_graphs(data1,data2)
        
        feat1,adj1,mask=self.to_dense(data1,with_mask=True)
        feat2,adj2=self.to_dense(data2)
        if scat_loss:
            scat=scatter(data1.edge_attr,index=data1.edge_index[0],dim=0)
            reordered_x,reordered_scat=self.scatter_mapping(ot_plan,data2)
            if not real:
                reordered_x=F.softmax(reordered_x,dim=1)
            feat_loss=self.loss(feat1,reordered_x,loss=loss)
            edge_loss=self.loss(scat,reordered_scat.squeeze(),loss="MSE")
            return feat_loss,edge_loss,ot_plan,ot_dist

        else:
            reordered_x,reordered_adj=self.graph_mapping(ot_plan,feat2,adj2)
            adj1,reordered_adj=self.flat_adjacency(adj1,reordered_adj,pad_mask=mask)
            feat1,reordered_x=self.flat_features(feat1,reordered_x,mask)
            if not real:
                reordered_x=F.softmax(reordered_x,dim=1)
                reordered_adj=F.softmax(reordered_adj,dim=1)
            
            feat_loss=self.loss(feat1,reordered_x,loss=loss)
            edge_loss=self.loss(adj1,reordered_adj,loss=loss)


        return feat_loss,edge_loss,ot_plan,ot_dist
    
    def soft_topk(self,data_in,data_rec,loss,hard=False,scatter=False,soft=False):
        ls=[]
        num_nodes=9
        for count,i in enumerate(range(0,data_rec.num_nodes,num_nodes)):
            k=torch.sum(data_in.batch==count).to(self.device)      
            topk_ind=torch.topk(F.softmax(data_rec.x[i:i+num_nodes,:],dim=1)[:,0],k=k,sorted=False,largest=False).indices+i
            ls.append(topk_ind)
        idx=torch.cat(ls)
        idx,_=idx.sort()
        edge_index,attr=subgraph(idx,data_rec.edge_index,data_rec.edge_attr,relabel_nodes=True)
        data_rec_reduced=Batch(data_rec.x[idx,:],edge_index,edge_attr=attr,batch=data_in.batch)
        feature_loss,edge_loss,ot_dist=self.sinkhorn_soft(data_in,data_rec_reduced,loss=loss,hard=hard,scat=scatter,soft=soft)
        condition=torch.isin(torch.arange(0,data_rec.x.size(0),device=self.device),idx).to(self.device)
        non_node_idx = torch.where(~condition)[0]
        non_node_loss=0
        if non_node_idx.size(0)>0:
            temp1=data_rec.x[non_node_idx,:]
            tgt=torch.stack([torch.tensor([1,0,0,0,0],dtype=torch.float,device=self.device) for _ in range(non_node_idx.size(0))])
            tgt=torch.clamp(tgt,0,1)
            non_node_loss= self.CE_loss(tgt,temp1) if loss=="CE" else self.MSE_loss(tgt,F.softmax(temp1,dim=1))
            non_node_loss/=data_rec.num_graphs
        return feature_loss+non_node_loss,edge_loss,ot_dist
    


    def nomatching(self,data_in,data_rec,loss):
        num_nodes=9
        idx=[]
        for i in range(0,data_in.num_graphs):
            temp=torch.sum(data_in.batch==i)
            idx.append(torch.arange(i*9,i*9+temp))

        idx=torch.cat(idx)
        idx,_=idx.sort()
        edge_index,attr=subgraph(idx,data_rec.edge_index,data_rec.edge_attr,relabel_nodes=True)
        data_rec_reduced=Batch(data_rec.x[idx,:],edge_index,edge_attr=attr,batch=data_in.batch)

        edge_mask=torch.argmax(data_rec_reduced.edge_attr,dim=1)!=0
        edge_index=data_rec_reduced.edge_index[:,edge_mask]
        attr=data_rec_reduced.edge_attr[edge_mask]
        edges_idx=torch.unique(edge_index[0])
        edge_index,attr=subgraph(edges_idx,edge_index,attr,relabel_nodes=True)
        rec_data_sm=Data(data_rec_reduced.x,edge_index,edge_attr=attr,batch=data_rec_reduced.batch)

        tau=0.1
        rec_data_sm=Data(F.softmax(rec_data_sm.x,dim=1)[:,1:],rec_data_sm.edge_index,edge_attr=F.softmax(rec_data_sm.edge_attr/tau,dim=1)[:,1:],batch=rec_data_sm.batch)
        
        scat1=scatter(data_in.edge_attr,index=data_in.edge_index[0],dim=0,dim_size=data_in.x.size(0))
        scat2=scatter(rec_data_sm.edge_attr,index=rec_data_sm.edge_index[0],dim=0,dim_size=rec_data_sm.x.size(0))
        feat_weight=torch.tensor([1,3,5,20],dtype=torch.float)
        edge_weight=torch.tensor([1,3,3,5],dtype=torch.float)

        edge_loss=self.loss(scat1,scat2,loss="MSE",weight=edge_weight)
        idx=torch.argmax(data_in.x,dim=1)
        feature_loss=F.nll_loss(rec_data_sm.x,idx,weight=feat_weight)


        condition=torch.isin(torch.arange(0,data_rec.x.size(0),device=self.device),idx).to(self.device)
        non_node_idx = torch.where(~condition)[0]
        non_node_loss=0
        if non_node_idx.size(0)>0:
            temp1=data_rec.x[non_node_idx,:]
            tgt=torch.stack([torch.tensor([1,0,0,0,0],dtype=torch.float,device=self.device) for _ in range(non_node_idx.size(0))])
            tgt=torch.clamp(tgt,0,1)
            non_node_loss= self.CE_loss(tgt,temp1) if loss=="CE" else self.MSE_loss(tgt,F.softmax(temp1,dim=1))
            non_node_loss/=data_rec.num_graphs
        return feature_loss+non_node_loss,edge_loss
    
    @staticmethod
    def one_hot_encode(data,num_nodes):
        ei,attr=to_undirected(data.edge_index,data.edge_attr)
        return Data(F.one_hot(torch.argmax(data.x,dim=1),num_classes=num_nodes).to(torch.float), ei,
                                          edge_attr=F.one_hot(torch.argmax(attr,dim=1),num_classes=4).to(torch.float),batch=data.batch)
    
    @staticmethod
    def to_dense(data,with_mask=False):
        x,mask=to_dense_batch(data.x,batch=data.batch)
        adj=to_dense_adj(data.edge_index,edge_attr=data.edge_attr,batch=data.batch)
        if with_mask:
            return x,adj,mask
        else:
            return x,adj
    
    @staticmethod
    def flat_adjacency(adj1,adj2,pad_mask):
        in_ls=[]
        rec_ls=[]
        for in_adj,rec_adj, m in zip(adj1,adj2,pad_mask):
            in_adj=in_adj[m,:][:,m]
            rec_adj=rec_adj[m,:][:,m]
            n,_,_=in_adj.size()
            mask=torch.eye(n).expand(n,n)
            mask=~torch.reshape(mask,(-1,)).to(bool)
            in_adj=torch.reshape(in_adj,(-1,adj1.size(-1)))
            rec_adj=torch.reshape(rec_adj,(-1,adj2.size(-1)))
            in_adj=in_adj[mask]
            rec_adj=rec_adj[mask]
            in_ls.append(in_adj)
            rec_ls.append(rec_adj)

        in_ls=torch.cat((in_ls),dim=0)
        rec_ls=torch.cat((rec_ls),dim=0)
        return in_ls,rec_ls
    
    @staticmethod
    def extend_flatadj(adj):
        adj=torch.cat((torch.full((adj.size(0),1),0),adj),dim=1)
        for idx,row in enumerate(adj):
            if torch.sum(row)==0:
                adj[idx,0]=1
        return adj
    
    @staticmethod
    def flat_features(feat1,feat2,mask):
        return torch.reshape(feat1,(-1,feat1.size(-1)))[mask.view(-1)],torch.reshape(feat2,(-1,feat2.size(-1)))[mask.view(-1)]

    @staticmethod
    def CE_loss(input,reordered_input,weights=None):
        input=torch.argmax(input,dim=1)
        loss=F.cross_entropy(reordered_input,input,weight=weights,reduction="none")
        loss=loss.sum()
        return loss

    @staticmethod
    def MSE_loss(input,reordered_input,weights=None):
        mse_l= F.mse_loss(reordered_input,input,reduction="none")
        mse_l=(mse_l*weights.to(torch.float)) if weights is not None else mse_l
        mse_l=mse_l.sum()
        return mse_l
    
    @staticmethod
    def loss(input,reordered,loss,weight=None):
        if loss=="CE":
            return Sinkhorn_Loss.CE_loss(input,reordered,weight)
        elif loss=="MSE":
            return Sinkhorn_Loss.MSE_loss(input,reordered,weight)
        else:
            raise Exception

    
    @staticmethod
    def graph_mapping(correspondence_matrix,x,adj):
        reordered_adj=torch.einsum("bij,bjle,bkl->bike", correspondence_matrix, adj, correspondence_matrix)
        reordered_x=torch.einsum("bij,bjk->bik",correspondence_matrix,x)
        return reordered_x,reordered_adj

    def compare_graphs(self,data1,data2):
        x1,mask1=self.embedd(data1.detach())
        x2,mask2=self.embedd(data2.detach())
        correspondence_matrix,ot_dists=self.soft_matching(x1,x2,mask1,mask2)
        return correspondence_matrix,ot_dists
    

    @staticmethod
    def kl_loss(mu ,logvar):
        #KL-Divergence is averaged for the whole batch
        kl_loss=torch.exp(logvar) - logvar + torch.pow(mu,2) - 1
        kl_loss = 0.5 * torch.sum(kl_loss,dim=1)
        kl_loss=torch.sum(kl_loss)
        return kl_loss


    def _uniform_dist(self,ind):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        H1=torch.ones(ind)/ind
        H2=torch.ones(ind)/ind
        return H1.to(device),H2.to(device)
    
    
    def _max_h(self,ot_plan,masks):
        ot_plan_clone = ot_plan.clone()  # So we don't modify the original tensor
        b,n,_ = ot_plan_clone.size()
        correspondence_matrix=torch.zeros((b,n,n),dtype=torch.float)
        
        for j,plan in enumerate(ot_plan_clone):
            plan=plan[masks[j],:][:,masks[j]]
            for _ in range(plan.size(0)):
                max_idx = plan.argmax().item()
                row, col = divmod(max_idx, plan.size(0))
                plan[:, col] = -float('inf')
                plan[row, :] = -float('inf')
                correspondence_matrix[j,row,col]=1.

        return correspondence_matrix


