import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

def remove_random_edge(data):
    # Select a random edge index
    edge_index=data.edge_index
    random_index = torch.randint(0, edge_index.size(1), (1,)).item()
    # Get the nodes of the selected edge
    src, tgt = edge_index[:, random_index]
    # Create masks to remove both occurrences of the selected edge
    mask1 = (edge_index[0] != src) | (edge_index[1] != tgt)
    mask2 = (edge_index[0] != tgt) | (edge_index[1] != src)
    return Data(data.x,edge_index=edge_index[:, mask1 & mask2],edge_attr=data.edge_attr[ mask1 & mask2],batch=data.batch)

def change_random_edge(data):
    # Select a random edge index
    edge_index=data.edge_index
    random_index = torch.randint(0, edge_index.size(1), (1,)).item()
    # Get the nodes of the selected edge
    src, tgt = edge_index[:, random_index]
    # Create masks to remove both occurrences of the selected edge
    mask1 = (edge_index[0] != src) | (edge_index[1] != tgt)
    mask2 = (edge_index[0] != tgt) | (edge_index[1] != src)

    edge_attr=data.edge_attr.clone()
    attr=data.edge_attr[~(mask1 & mask2)]
    chosen_set=torch.nonzero(attr[0]==0).squeeze()
    random_number = torch.randint(1, chosen_set.size(0), (1,)).item()
    new_attr=torch.tensor([[0,0,0,0],[0,0,0,0]],dtype=torch.float)
    new_attr[:,chosen_set[random_number]]=1
    edge_attr[~(mask1 & mask2)]=new_attr
    return Data(data.x,edge_index=edge_index,edge_attr=edge_attr,batch=data.batch)

def add_random_edge(data):
    # Get the number of existing edges
    
    num_nodes=data.num_nodes
    edge_index=data.edge_index
    # Initialize variables to store the nodes of the new edge
    src, tgt = -1, -1

    # Keep randomly selecting nodes until a valid pair is found
    while True:
        # Randomly select two nodes
        src = torch.randint(0, num_nodes, (1,)).item()
        tgt = torch.randint(0, num_nodes, (1,)).item()

        # Check if the selected nodes are not already connected by an edge
        if not torch.any((edge_index[0] == src) & (edge_index[1] == tgt)) and src!=tgt:
            break

    # Add the new edge to the edge index
    new_edge_index = torch.cat([edge_index, torch.tensor([[src], [tgt]]),torch.tensor([[tgt], [src]])], dim=1)
    ran_idx=torch.randint(1,4,(1,)).item()
    temp=torch.tensor([0,0,0,0])
    temp[ran_idx]=1
    temp=temp.unsqueeze(0)
    new_attr=torch.cat([data.edge_attr,temp,temp], dim=0)

    return Data(data.x,edge_index=new_edge_index,edge_attr=new_attr,batch=data.batch)

def change_random_node(data):
        
    
    random_index = torch.randint(0, data.x.size(0), (1,)).item()
    chosen_set=torch.nonzero(data.x[random_index]==0).squeeze()
    random_number = torch.randint(1, chosen_set.size(0), (1,)).item()
    temp=torch.tensor([0,0,0,0])
    temp[chosen_set[random_number]]=1
    x=data.x.clone()
    x[random_index]=temp

    return Data(x,data.edge_index,data.edge_attr,data.batch)

def augment_graph(data,augment_num=1):
    data_ls=[data]
    augs=[]
    aug_types=["rm_edge","change_edge","change_node","add_edge"]
    for i in range(augment_num):
        aug_idx=torch.randint(0,4,(1,)).item()
        aug_type=aug_types[aug_idx]
        if aug_type=="rm_edge":
            data=remove_random_edge(data)
        elif aug_type=="change_edge":
            data=change_random_edge(data)
        elif aug_type=="change_node":
            data=change_random_node(data)
        elif aug_type=="add_edge":
            data=add_random_edge(data)
        else:
            raise Exception
        augs.append(aug_type)
        data_ls.append(data)
    return data_ls,augs

def drop_node(data):
    idx=torch.arange(0,data.x.size(0))
    mask=torch.ones(data.x.size(0)).to(bool)
    drop_idx=torch.randint(0,data.x.size(0),(1,)).item()
    mask[drop_idx]=False
    edge_index,edge_attr=subgraph(idx[mask],data.edge_index,data.edge_attr,relabel_nodes=True)
    return Data(data.x[mask],edge_index,edge_attr=edge_attr)

