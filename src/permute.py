import torch
from torch_geometric.data import Batch
import itertools

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def permute_batch(batch,permutations,device):
    num_dict={45:9,36:8,28:7,21:6,15:5,10:4,6:3,3:2,1:1}
    data_ls=batch.to_data_list()
    ls=[]
    for data in data_ls:
        data = data.to(device).clone()
        num_nodes=num_dict[data.x.size(0)]
        perm_num=permutations[num_nodes-1].size(0)
        idx=torch.randint(0,perm_num,(1,),device=device)
        perm=permutations[num_nodes-1][idx].squeeze(0).to(device)
        data.x=data.x[perm]
        ls.append(data.to(device))
    return Batch.from_data_list(ls)


def permute_complete_dataset(dataset,permutations,device):
    #idx=random.sample(range(len(dataset)), 32**2)
    data_ls=[]
    ls=[dataset[i].clone().to(device) for i in range(len(dataset))]
    num_dict={45:9,36:8,28:7,21:6,15:5,10:4,6:3,3:2,1:1}
    for data in ls:
        data = data.to(device)
        num_nodes=num_dict[data.x.size(0)]
        perm_num=permutations[num_nodes-1].size(0)
        idx=torch.randint(0,perm_num,(1,),device=device)
        perm=permutations[num_nodes-1][idx].squeeze(0).to(device)
        data.x=data.x[perm]
        data_ls.append(data.to(device))
    return data_ls

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