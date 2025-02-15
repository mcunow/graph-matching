import torch
from torch_geometric.utils import to_dense_batch, scatter, to_dense_adj
import gnn
import itertools

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
def permute_graphs_over_n(n):
    permutations = list(itertools.permutations(range(n)))
    ls=[]
    dic={}
    k=n
    for i in range(n):
        for j in range(i,n):
            if i != j:  # Avoid self-loops
                dic[(i,j)]=k
                dic[(j,i)]=k
                k += 1
    for perm in permutations:
        temp=permute_graph(perm,dic)
        ls.append(temp)
    return torch.tensor(ls,device=DEVICE)
def permute_graph(perm,dic):
    ls={}
    for idx,p in enumerate(perm):
        ls[idx]=p

    for i in range(len(perm)):
        for j in range(i,len(perm)):
            if i!=j:
                temp=dic[(i,j)]
                key=((ls[i],ls[j]))
                ls[temp]=dic[key]
    return list(ls.values())

def create_permutations():
    ls=[]
    for i in range (1,10):
        ls.append(permute_graphs_over_n(i))
    return ls


class BruteForceMatcher():
    def __init__(self):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.permutations=create_perms()
        self.num_perms=self.permutations.size(0)
        self.indices=get_data_indices()

    def match(self,batch,rec_batch):
        data=batch.clone()
        rec_data=rec_batch.clone()

        feat_mask=data.x[:,0]==1
        edge_mask=data.x[:,0]==-1

        edge_weights=torch.tensor([.5,5,30,50,100],device=self.device)
        feat_weights=torch.tensor([.1,1,5,5,100],device=self.device)
        feature_loss=(data.x[feat_mask][:,1:]-rec_data.x[feat_mask])**2
        feature_loss*=feat_weights
        #feature_loss=feature_loss.mean()
        edge_loss=(data.x[edge_mask][:,1:]-rec_data.x[edge_mask])**2
        edge_loss*=edge_weights
        loss=torch.cat((feature_loss,edge_loss),dim=0)
        loss=loss.sum(1)
        #edge_loss=edge_loss.mean()
        unmatched_loss=loss.mean()

        
        #Extend data to the structure of graph with 9 nodes if necessary. This enables
        data.x=extend_data(data,self.indices,0)
        temp=torch.cat([torch.tensor([i for _ in range(45)])for i in range(data.num_graphs)])
        rec_data.x=extend_data(rec_data,self.indices,0)
        data.batch=temp
        rec_data.batch=temp
        feat_mask=data.x[:,0]==1
        edge_mask=data.x[:,0]==-1

        in_embedd=data.x[:,1:].view(data.num_graphs,45,5)
        rec_embedd=rec_data.x.view(data.num_graphs,45,5)


        with torch.no_grad():
            pairwise_sum = (in_embedd[:, :, None, :] - rec_embedd[:, None, :, :])**2
            pairwise_sum=pairwise_sum[:,self.permutations[:self.num_perms,:,0],self.permutations[:self.num_perms,:,1]]

            pairwise_sum=pairwise_sum.sum(3)
            pairwise_sum=pairwise_sum.mean(-1)
            values,idx=pairwise_sum.min(1)
            assert not torch.isinf(values).any()

        perms=self.permutations[idx][:,:,1]

        reordered_embedd=rec_embedd[torch.arange(perms.size(0)).unsqueeze(1),perms]

        in_embedd=in_embedd.view(-1,5)
        reordered_embedd=reordered_embedd.view(-1,5)
        feature_loss=(in_embedd[feat_mask]-reordered_embedd[feat_mask])**2
        feature_loss*=feat_weights
        edge_loss=(in_embedd[edge_mask]-reordered_embedd[edge_mask])**2
        edge_loss*=edge_weights
        loss=torch.cat((feature_loss,edge_loss),dim=0)
        loss=loss.sum(1)
        loss=loss.mean()
        return loss,unmatched_loss
    
    def unmatched(self,batch,rec_batch):
        data=batch.clone()
        rec_data=rec_batch.clone()

        feat_mask=data.x[:,0]==1
        edge_mask=data.x[:,0]==-1

        edge_weights=torch.tensor([.5,5,30,50,100],device=self.device)
        feat_weights=torch.tensor([.1,1,5,5,100],device=self.device)
        feature_loss=(data.x[feat_mask][:,1:]-rec_data.x[feat_mask])**2
        feature_loss*=feat_weights
        #feature_loss=feature_loss.mean()
        edge_loss=(data.x[edge_mask][:,1:]-rec_data.x[edge_mask])**2
        edge_loss*=edge_weights
        loss=torch.cat((feature_loss,edge_loss),dim=0)
        loss=loss.sum(1)
        #edge_loss=edge_loss.mean()
        return loss.mean(), torch.tensor([0],device=self.device)
    
    def statistics_loss(self,batch,rec_batch):
        data=batch.clone()
        rec_data=rec_batch.clone()

        feat_mask=data.x[:,0]==1
        edge_mask=data.x[:,0]==-1
        edge_weights=torch.tensor([.0005,.005,.03,.05,.1],device=self.device)
        feat_weights=torch.tensor([.0001,.01,.05,.05,.1],device=self.device)

        x=batch.x[feat_mask][:,1:]
        x_rec=gumbel_softmax_hard(rec_data.x[feat_mask])

        feat=scatter(x,data.batch[feat_mask])
        feat_rec=scatter(x_rec,data.batch[feat_mask])

        attr=batch.x[edge_mask][:,1:].to(torch.float)
        attr_rec=gumbel_softmax_hard(rec_data.x[edge_mask])

        attr=scatter(attr,data.batch[edge_mask])
        attr_rec=scatter(attr_rec,data.batch[edge_mask])

        feat_loss=(feat-feat_rec)**2
        feat_loss*=feat_weights
        attr_loss=(attr-attr_rec)**2
        attr_loss*=edge_weights

        return feat_loss.mean()+attr.mean(),torch.tensor([0],device=self.device)




def gumbel_softmax_hard(logits, tau=0.1):
    # Add Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    y = logits + gumbel_noise
    
    # Compute soft output
    soft_output = torch.softmax(y / tau, dim=-1)
    
    # Convert to hard output (one-hot)
    hard_output = torch.zeros_like(soft_output)
    hard_output.scatter_(-1, soft_output.argmax(dim=-1, keepdim=True), 1.0)
    
    # Straight-through estimator: Replace soft with hard in the forward pass
    return (hard_output - soft_output).detach() + soft_output


def create_perms():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perms=permute_graphs_over_n(9).to(device)

    permutations=torch.stack([torch.stack((torch.arange(45).to(device), perms[i]), dim=1) for i in range(len(perms))])
    return permutations



def get_data_indices():
    t=torch.full((9,45),0,device=DEVICE)
    for num_node in range(1,10):
        last_idx=int((num_node*(num_node-1))/2 +num_node)
        k=num_node
        for i in range(9):
            if i<num_node:
                t[num_node-1,i]=i
            else:
                t[num_node-1,i]=last_idx
        idx=8
        for i in range(9):
            for j in range(i+1,9):
                idx+=1
                if i<num_node and j <num_node:
                    t[num_node-1,idx]=k
                    k+=1
                else:
                    t[num_node-1,idx]=last_idx
    return t

def extend_data(batch,indices,fill_value):
    data_ls=batch.to_data_list()
    ls=[]
    num_dict={45:9,36:8,28:7,21:6,15:5,10:4,6:3,3:2,1:1}
    for data in data_ls:
        x=data.x.clone()
        num_nodes=data.x.size(0)
        num_nodes=num_dict[num_nodes]-1
        idx=indices[num_nodes]
        x=torch.cat((x,torch.full((1,data.x.size(1)),fill_value,device=DEVICE)),dim=0)
        ls.append(x[idx])
    return torch.cat(ls)


class BruteForceSampleMatcher():
    def __init__(self,sample_num):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.permutations=create_perms()
        self.num_perms=self.permutations.size(0)
        self.indices=get_data_indices()
        self.sample_num=sample_num

    def match(self,batch,rec_batch):
        data=batch.clone()
        rec_data=rec_batch.clone()

        feat_mask=data.x[:,0]==1
        edge_mask=data.x[:,0]==-1

        unmatched_loss=self.compute_loss(data.x[:,1:],rec_data.x,feat_mask,edge_mask)
        
        #Extend data to the structure of graph with 9 nodes if necessary. This enables
        data.x=extend_data(data,self.indices,0)
        temp=torch.cat([torch.tensor([i for _ in range(45)])for i in range(data.num_graphs)])
        rec_data.x=extend_data(rec_data,self.indices,0)
        data.batch=temp
        rec_data.batch=temp
        feat_mask=data.x[:,0]==1
        edge_mask=data.x[:,0]==-1

        in_embedd=data.x.view(data.num_graphs,45,6)
        in_embedd=in_embedd[:,:,1:]
        rec_embedd=rec_data.x.view(data.num_graphs,45,5)

        with torch.no_grad():
            pairwise_sum = (in_embedd[:, :, None, :] - rec_embedd[:, None, :, :])**2
            pairwise_sum=pairwise_sum[:,self.permutations[:self.num_perms,:,0],self.permutations[:self.num_perms,:,1]]
            pairwise_sum=pairwise_sum.sum(3)
            pairwise_sum=pairwise_sum.mean(-1)
            values, indices = pairwise_sum.view(pairwise_sum.size(0), -1).topk(self.sample_num, dim=1, largest=False)

        idx=self.sample_idx(values,indices)
        perms=self.permutations[idx][:,:,1]
        reordered_embedd=rec_embedd[torch.arange(perms.size(0)).unsqueeze(1),perms]

        loss=self.compute_loss(in_embedd.view(-1,5),reordered_embedd.view(-1,5),feat_mask,edge_mask)
        return loss,unmatched_loss
    
    def sample_idx(self,values,indices):
        num_valid=torch.isfinite(values).sum(1)
        B=values.size(0)
        sampled_idx=torch.zeros(B,device=self.device,dtype=torch.int)
        for i in range(B):
            idx=torch.randint(0,num_valid[i],(1,),device=self.device)
            sampled_idx[i]=indices[i,idx]
        return sampled_idx
    
    def compute_loss(self,x1,x2,feat_mask,edge_mask):
        edge_weights=torch.tensor([.5,5,30,50,100],device=self.device)
        feat_weights=torch.tensor([.1,1,5,5,100],device=self.device)
        feature_loss=(x1[feat_mask]-x2[feat_mask])**2
        feature_loss*=feat_weights
        edge_loss=(x1[edge_mask]-x2[edge_mask])**2
        edge_loss*=edge_weights
        loss=torch.cat((feature_loss,edge_loss),dim=0).sum(1).mean()
        return loss
    
class GNN_Loss():
    def __init__(self,input_dim,num_encoders):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoders=[]
        self.num_encoders=num_encoders
        for _ in range(num_encoders):
            encoder=gnn.GIN(in_channels=input_dim,hidden_channels=[8,12],out_channels=16,norm="layer",num_layers=3,
                                act="leakyrelu",dropout=0).to(self.device)
            for param in encoder.parameters():
                param.requires_grad = False

            self.encoders.append(encoder)
            
        
    def gnn_loss(self,data,data_rec):
        rand_enc_idx=torch.randint(0,self.num_encoders,size=(10,),device=self.device)
        data_rec2=data_rec.clone()        
        data_rec2.x=torch.cat((data.x[:,0].unsqueeze(1),data_rec.x),dim=1)

        loss=self.encoder_loss(data,data_rec2,rand_enc_idx)

        return 100*loss,torch.tensor([0],device=self.device)

    def encoder_loss(self,data,data_rec,idx):
        ls=[]
        for i in idx:
            x=self.encoders[i](data)
            x_rec=self.encoders[i](data_rec)
            x=scatter(x,data.batch,reduce='mean')
            x_rec=scatter(x_rec,data.batch,reduce='mean')
            loss=(x-x_rec)**2
            loss=torch.sum(loss,dim=1)
            ls.append(loss.mean())
        return torch.stack(ls).sum()




