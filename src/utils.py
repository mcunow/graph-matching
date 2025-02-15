
from rdkit import Chem
from torch_geometric.data import Data,Batch
from torch_geometric.utils import dense_to_sparse,to_undirected, to_dense_adj,to_dense_batch, subgraph,to_networkx
import torch
import os.path as osp
import torch
import torch.nn.functional as F
import networkx as nx


 
def txt2smiles(path):
    with open(path,'rb') as f:
        smiles_str=f.readlines()

    #Remove first item and remove last element of each item    
    smiles_ls = [string[:-1] for string in smiles_str[1:]]

    return smiles_ls

def smiles2data_edge_index(smiles_ls):
    data_list=[]
    bond_dict={Chem.BondType.SINGLE:torch.tensor([1,0,0,0]),Chem.BondType.AROMATIC:torch.tensor([0,1,0,0]),
            Chem.BondType.DOUBLE:torch.tensor([0,0,1,0]),Chem.BondType.TRIPLE:torch.tensor([0,0,0,1])}
    for smiles in smiles_ls:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("None")
            break

        Chem.Kekulize(mol)
        # Chem.SanitizeMol(mol)

        # Extract node features
        node_features=one_hot_features(mol)

        # Extract edge index
        edges = []
        edge_attrs = []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.append((start, end))
            edges.append((end, start))  # Because the graph is undirected

            # Bond type as an example edge attribute
            #bond_type = bond.GetBondTypeAsDouble()
            b=bond.GetBondType()
            if b is Chem.BondType.AROMATIC:
                assert(True)
            edge=bond_dict[b]
            edge_attrs.append(edge)
            edge_attrs.append(edge)
            #edge_attrs.extend([bond_type, bond_type])
            # Add bond type for both edge directions
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous() if len (edges)>1 else torch.empty((2,0),dtype=torch.long)
        edge_attr = torch.stack(edge_attrs,dim=0).to(torch.float32) if len(edge_attrs)>1 else torch.empty((0,4),dtype=torch.float)

                
        # Create the Data object
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        data_list.append(data)
    return data_list



def one_hot_features(mol):
    atom_dict={1:[1,0,0,0,0],6:[1,0,0,0],7:[0,1,0,0],8:[0,0,1,0],9:[0,0,0,1]}
    node_features = []
    for atom in mol.GetAtoms():
        atomic_num=atom.GetAtomicNum()
        atom_features = atom_dict[atomic_num]
        node_features.append(atom_features)
    return node_features



def node_number_loss(node_num,pred_node_num):
    loss=torch.square(node_num-pred_node_num)
    return torch.mean(loss)

def get_edge_weights(mol):
    edge_attrs = []
    for bond in mol.GetBonds():
        # Bond type as an example edge attribute
        bond_type = bond.GetBondTypeAsDouble()
        edge_attrs.extend([bond_type, bond_type])  # Add bond type for both edge directions
    return torch.tensor(edge_attrs, dtype=torch.float).view(-1, 1)



def save_data(data_list,path):
     idx=0
     for data in data_list:
        torch.save(data, osp.join(path, f'data_{idx}.pt'))
        idx += 1



def extend_data(data_in):
    ei=torch.ones((9,9)).fill_diagonal_(0).nonzero().t()
    edge_attr=torch.stack([torch.tensor([1,0,0,0,0]) for _ in range(9*8)])
    n=data_in.x.size(0)
    
    x=torch.stack([torch.tensor([0,0,0,0,0],dtype=torch.float) for _ in range(n)])
    x= torch.cat((x,torch.stack([torch.tensor([1,0,0,0,0],dtype=torch.float) for _ in range(9-n)])),dim=0)if n<9 else x
    x[:n,1:]=data_in.x
    mask=ei>=n
    mask=mask[0]|mask[1]
    ei=ei[:,mask]
    attr=edge_attr[mask]
    ei=torch.cat((data_in.edge_index,ei),dim=1).to(int)
    attr=torch.cat((data_in.edge_attr,attr),dim=0)
    batch=torch.tensor([0 for _ in range(9)])
    return Batch(x,ei.contiguous(),edge_attr=attr.to(torch.float32),batch=batch)

def map_attributes(x,edge_index,edge_attr,perm):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x=x[perm]

    # Create a mapping from old node indices to new node indices
    perm=torch.argsort(perm)
    idx_dict = {old: new for new, old in enumerate(perm.tolist())}

    # Adjust the edge indices based on the new node order
    #ei_temp=edge_index-edge_index.min()
    mapped_edge_index = torch.tensor([idx_dict[idx.item()] for idx in edge_index.reshape(-1)]).contiguous().view_as(edge_index)

    # Map the new edge indices of the graph to their corresponding attributes
    edge_to_attr = {(src.item(), tgt.item()): attr for src, tgt, attr in zip(edge_index[0], edge_index[1], edge_attr)}
    new_edge_attrs = [edge_to_attr.get((src.item(), tgt.item())) for src, tgt in zip(mapped_edge_index[0], mapped_edge_index[1])]      
    new_edge_attr = torch.stack(new_edge_attrs) if len(new_edge_attrs)>0 else torch.empty(0,5)

    return x.to(device),mapped_edge_index,new_edge_attr.to(device)

def permute_data(in_data):
    data=in_data.detach().clone()
    perm=torch.randperm(data.x.shape[0])
    # Permute node features
    data.x = data.x[perm]

    # Create a mapping from old node indices to new node indices
    idx_dict = {old: new for new, old in enumerate(perm.tolist())}

    # Use the mapping to update edge indices and get the permutation for edges
    edge_dict = {(u, v): i for i, (u, v) in enumerate(data.edge_index.t().tolist())}
    new_edge_index = []
    edge_perm = []

    for u, v in data.edge_index.t().tolist():
        new_u, new_v = idx_dict[u], idx_dict[v]
        new_edge_index.append([new_u, new_v])

        edge_perm.append(edge_dict[(u, v)])

    data.edge_index = torch.tensor(new_edge_index).t().contiguous().to(torch.long) if len(new_edge_index)>1 else torch.empty((2,0),dtype=torch.long)

    return data,torch.argsort(perm)



def monitor_gradients(model):
  avg_gradient_magnitudes = []

  for name, param in model.named_parameters():
      if param.grad is not None:
          gradient_norm = param.grad.data.norm(2).item()
          avg_gradient_magnitudes.append(gradient_norm)
          print(f"Gradient norm for {name}: {gradient_norm}")

  avg_gradient = sum(avg_gradient_magnitudes) / len(avg_gradient_magnitudes)
  print(f"Average gradient norm: {avg_gradient}")
    
def fully_connected_graph(num_nodes_per_graph,edge_dim=16,fill_value=None):
    
    all_edge_indices = []
    all_edge_attrs = []
    all_batches = []

    start_idx = 0
    for i,num_nodes in enumerate( num_nodes_per_graph):
        # Create a dense adjacency matrix for a fully connected graph
        num_nodes=int(num_nodes.item())
        adj = torch.ones(num_nodes, num_nodes)
        # Remove self-loops
        adj.fill_diagonal_(0)
        edge_index, _ = dense_to_sparse(adj)
        
        # Adjust indices for the batch
        edge_index += start_idx
        all_edge_indices.append(edge_index)

        # Create random edge attributes for the edges of this graph
        if fill_value is None:
            if edge_dim==1:
                edge_attr=torch.zeros((edge_index.size(1), edge_dim))
            else:
                edge_attr=torch.tensor([1,1,1,1,1],dtype=torch.float).repeat(edge_index.size(1),1)
        else:
            edge_attr=fill_value.repeat(edge_index.size(1),1)
        all_edge_attrs.append(edge_attr)
        # Create batch information for the nodes of this graph
        batch = torch.full((num_nodes,), fill_value=i, dtype=torch.long)
        all_batches.append(batch)

        start_idx += num_nodes

    batched_edge_index = torch.cat(all_edge_indices, dim=1)
    batched_edge_attr = torch.cat(all_edge_attrs, dim=0)
    batched_batch = torch.cat(all_batches, dim=0)
    return batched_edge_index,batched_edge_attr,batched_batch

def one_hot_encode(edge_attr):

    # Map the values of edge_attr to their indices in the possible_values array
    ten=torch.tensor([0, 1, 1.5, 2, 3])
    indices = (edge_attr == ten.view(-1, 1, 1)).max(dim=0).indices

    # Perform one-hot encoding
    one_hot = torch.zeros(indices.size(0), ten.size(0), dtype=torch.float32)
    one_hot.scatter_(1, indices.view(-1, 1), 1)

    return one_hot

def fully_connected_encoding(data):
    edge_indices = []
    edge_attrs = []
    for graph_idx in data.batch.unique():
        # Extract nodes for this graph
        nodes = torch.where(data.batch == graph_idx)[0]
        num_nodes = nodes.size(0)
        
        # Create fully connected edges for these nodes
        row = nodes.view(-1, 1).repeat(1, num_nodes).view(-1)
        col = nodes.repeat(num_nodes)
        edge_index = torch.stack([row, col], dim=0)
        
        # Remove self-loops
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        
        # Ensure undirected graph
        edge_index = to_undirected(edge_index)
        
        # Map the edge indices to the attributes from the original graph
        edge_to_attr = {(src.item(), tgt.item()): attr.item() for src, tgt, attr in zip(data.edge_index[0], data.edge_index[1], data.edge_attr.squeeze())}
        attributes = [edge_to_attr.get((src.item(), tgt.item()), -1) for src, tgt in zip(edge_index[0], edge_index[1])]

        edge_indices.append(edge_index)
        edge_attrs.append(torch.tensor(attributes))
    
    # Concatenate all edge indices and attributes for the batch
    edge_index = torch.cat(edge_indices, dim=1)
    edge_attr = torch.cat(edge_attrs,dim=0)
    edge_attr[edge_attr==-1]=0
    
    new_data = Batch(x=data.x, edge_index=edge_index, edge_attr=edge_attr, batch=data.batch)
    
    return new_data


def fully_connected_one_hot(data):
    edge_indices = []
    edge_attrs = []
    unique_vals = torch.tensor([0, 1, 1.5, 2, 3])
    edge_dim=5
    if data.edge_attr.size(1)>1:
        edge_attr=torch.cat((torch.tensor([0]).repeat(data.edge_attr.size(0)).unsqueeze(1),data.edge_attr),dim=1)
        edge_to_attr = {(src.item(), tgt.item()): attr 
                for src, tgt, attr in zip(data.edge_index[0], data.edge_index[1], edge_attr.squeeze())}  
    else:
        edge_attr=data.edge_attr
        edge_to_attr = {(src.item(), tgt.item()): attr.item() 
                        for src, tgt, attr in zip(data.edge_index[0], data.edge_index[1], data.edge_attr.squeeze())}
    
    for graph_idx in data.batch.unique():
        # Extract nodes for this graph
        nodes = torch.where(data.batch == graph_idx)[0]
        num_nodes = nodes.size(0)
        
        # Create fully connected edges for these nodes
        row = nodes.view(-1, 1).repeat(1, num_nodes).view(-1)
        col = nodes.repeat(num_nodes)
        edge_index = torch.stack([row, col], dim=0)
        
        # Remove self-loops
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        
        # Map the edge indices to the attributes from the original graph
        if data.edge_attr.size(1)==1:
            attributes = [edge_to_attr.get((src.item(), tgt.item()), -1) for src, tgt in zip(edge_index[0], edge_index[1])]
            
            # Compute indices for one-hot encoding
            _, indices = (unique_vals == torch.Tensor(attributes).view(-1, 1)).max(dim=1)        
            # Create one-hot encoding
            one_hot = torch.zeros((edge_index.size(1), edge_dim))
            one_hot[torch.arange(edge_index.size(1)), indices] = 1
            edge_attrs.append(one_hot)

        else:
            attributes = [edge_to_attr.get((src.item(), tgt.item()), torch.tensor([1,0,0,0,0])) 
                          for src, tgt in zip(edge_index[0], edge_index[1])]
            edge_attrs.append(torch.stack(attributes, dim=0)) if len(attributes)!=0 else edge_attrs.append(torch.empty(0, edge_dim))
        edge_indices.append(edge_index)
        
    
    # Concatenate all edge indices and attributes for the batch
    edge_index = torch.cat(edge_indices, dim=1)
    edge_attr = torch.cat(edge_attrs, dim=0)
    
    new_data = Batch(x=data.x, edge_index=edge_index, edge_attr=edge_attr, batch=data.batch)
    
    return new_data


def symmetrize_attributes(edge_index, attr):
    edge_dict = {(u, v): i for i, (u, v) in enumerate(edge_index.t().tolist())}
    updated_attr = attr.clone()
    for u, v in edge_index.t().tolist():
        attr_idx1 = edge_dict[(u, v)]
        attr_idx2 = edge_dict[(v, u)]
        
        temp = (attr[attr_idx1] + attr[attr_idx2]) / 2.0
        updated_attr[attr_idx1] = temp
        updated_attr[attr_idx2] = temp

    return updated_attr


def threshold_attributes(edge_index, attr):
    # Get indices where attribute isn't in the 'non-edge' class (assuming it's the first class)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mask = torch.argmax(attr, dim=1) != 0

    # Index to get valid edge indices and attributes
    edge_index = edge_index[:, mask]
    attr = attr[mask]

    # Get the new class of each attribute
    temp = torch.argmax(attr, dim=1)
    

    # Define mapping tensor
    ten = torch.tensor([0,1, 1.5, 2, 3]).to(device)

    # Use indexing for assignment instead of a loop
    attr = ten[temp].view(-1, 1)
    

    return edge_index, attr

def extract_thres_graph(in_data,hard=False,gumbel=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data=in_data.clone().detach()
    m=torch.nn.Softmax(dim=1)
    #attr=m(data.edge_attr)
    edge_index,attr=threshold_attributes(data.edge_index,data.edge_attr)
    #For feature one-hot encoding
    x=m(data.x) if not gumbel else data.x
    if hard:
        max_value, _ = torch.max(x, dim=1)
        x = torch.eq(x, max_value.unsqueeze(1)).float().to(device)

    return Batch(x=x,edge_index=edge_index,edge_attr=attr,batch=data.batch)



def compute_pairwise_means( embeddings):
    n = embeddings.size(0)

    # Create masks for the upper and lower triangular parts without the diagonal
    upper_mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
    lower_mask = torch.tril(torch.ones(n, n), diagonal=-1).bool()
    
    # Get the indices for the rows and columns for both masks
    upper_rows, upper_cols = upper_mask.nonzero().t()
    lower_rows, lower_cols = lower_mask.nonzero().t()

    # Concatenate embeddings for pairs
    upper_pairs = torch.cat([embeddings[upper_rows], embeddings[upper_cols]], dim=1)
    lower_pairs = torch.cat([embeddings[lower_rows], embeddings[lower_cols]], dim=1)     
    # Combine both sets of pairs
    pairwise_concatenated = torch.cat([upper_pairs, lower_pairs], dim=0)

    return pairwise_concatenated

def create_fc_graph_with_nodes_as_edges(sizes,vecs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    attr_length=5

    ei_mod=torch.empty((2,0))
    count=0
    n_ex=0
    embeddings=torch.empty((0,vecs.size(1)+attr_length)).to(device)
    batch=torch.empty((0))
    batch_id=0
    for n,vec in zip(sizes,vecs):
        batch=torch.cat((batch,torch.tensor([batch_id for _ in range(n+int((n*(n-1))/2))])))
        batch_id+=1
        ei=torch.empty((2,0))
        ten=torch.repeat_interleave(torch.cat((vec.unsqueeze(0),torch.tensor([-1,-1,-1,-1,-1]).unsqueeze(0).to(device)),dim=1),n,dim=0)
        embeddings=torch.cat((embeddings,ten.to(device)),dim=0)

        for i in range(count,count+n):
            for j in range(i+1,count+n):
                ei=torch.cat((ei,torch.tensor([[i],[j]]),torch.tensor([[j],[i]])),1)

        length=ei.size(1)
        n_ex+=n
        for i in range(0,length,2):
            ei_mod=torch.cat((ei_mod,torch.tensor([[ei[0][i],n_ex],[n_ex,ei[0][i]]]),
                              torch.tensor([[ei[0][i+1],n_ex],[n_ex,ei[0][i+1]]])),1)
            n_ex+=1
        count=count+n+int((n*(n-1))/2)

        ten=torch.randn((int((n*(n-1))/2),attr_length))
        ten2=-torch.ones((int((n*(n-1))/2),vec.size(0)))
        ten=torch.cat((ten2,ten),dim=1)
        embeddings=torch.cat((embeddings,ten.to(device)),dim=0)
        ei_mod=ei_mod.to(torch.long)
        batch=batch.to(torch.int)
    return Batch(embeddings,ei_mod.contiguous(),batch=batch).to(device)



def reduce_graph(data_rec,node_dim=4):
    """
    Reduces a graph that has nodes as edges, back to its original structure. Does implement the 
    inverse logic to the function create_fc_graph_with_nodes_as_edge(data) that created the graph with nodes as edges. Hence, is not generally applicable to reduce graphs,
    as it depends on the created order of edge_index.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    count=0
    count_edges=0
    count_n=0
    batch_copy=data_rec.clone()
    edge_attr=torch.empty((0,5),dtype=torch.float).to(device)
    ei=torch.empty((2,0),dtype=torch.long).to(device)
    embedds=torch.empty((0,node_dim),dtype=torch.float).to(device)
    num_nodes=torch.bincount(data_rec.batch)
    num_nodes=(-1+torch.sqrt(1+4*(num_nodes*2)))//2
    num_nodes=num_nodes.to(torch.int)
    

    batch_reduced=torch.cat([torch.tensor([ind for _ in range(i)]) for ind,i in enumerate(num_nodes)]).to(device)
    

    for n in num_nodes:
        x = batch_copy.x[count:count+n.item(),:node_dim]
        embedds=torch.cat((embedds,x),dim=0)

   
        src=batch_copy.edge_index[0,count_edges:count_edges+((x.size(0)*(x.size(0)-1)))*2]
        for ind in range(0,src.size(0),4):
            diff=count-count_n
            temp1=src[ind]-diff
            temp2=src[ind+2]-diff
            temp3=src[ind+1]-diff
            ei=torch.cat((ei,torch.tensor([temp1,temp2]).unsqueeze(1).to(device),
                          torch.tensor([temp2,temp1]).unsqueeze(1).to(device)),dim=1)
            temp4=batch_copy.x[temp3+diff,-5:].unsqueeze(0)
            edge_attr=torch.cat((edge_attr,temp4,temp4),dim=0)

        count_edges+=((x.size(0)*(x.size(0)-1)))*2
        count+=n+((n-1)*n)//2
        count_n+=n


    return Batch(embedds,ei.contiguous(),edge_attr=edge_attr,batch=batch_reduced).to(device)

def create_fc_graph_with_nodes_as_edges_from_matrix(sizes,vecs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    attr_length=5

    ei_mod=torch.empty((2,0))
    count=0
    count_n=0
    n_ex=0
    embeddings=torch.empty((0,vecs.size(1)+attr_length)).to(device)
    batch=torch.empty((0))
    batch_id=0
    for n in sizes:
        batch=torch.cat((batch,torch.tensor([batch_id for _ in range(n+int((n*(n-1))/2))])))
        batch_id+=1
        ei=torch.empty((2,0))
        temp=torch.repeat_interleave(torch.tensor([-1,-1,-1,-1,-1]).unsqueeze(0).to(device),n,dim=0)
        temp2=torch.cat((vecs[count_n:count_n+n],temp),dim=1)
        embeddings=torch.cat((embeddings,temp2),dim=0)

        for i in range(count,count+n):
            for j in range(i+1,count+n):
                ei=torch.cat((ei,torch.tensor([[i],[j]]),torch.tensor([[j],[i]])),1)

        length=ei.size(1)
        n_ex+=n
        for i in range(0,length,2):
            ei_mod=torch.cat((ei_mod,torch.tensor([[ei[0][i],n_ex],[n_ex,ei[0][i]]]),
                              torch.tensor([[ei[0][i+1],n_ex],[n_ex,ei[0][i+1]]])),1)
            n_ex+=1
        count+=n+int((n*(n-1))/2)
        count_n+=n

        ten1=-torch.ones((int((n*(n-1))/2),embeddings.size(1)-attr_length))
        ten2=torch.zeros((int((n*(n-1))/2),attr_length))
        
        attr=torch.cat((ten1,ten2),dim=1).to(device)
        embeddings=torch.cat((embeddings,attr),dim=0)
        ei_mod=ei_mod.to(torch.long)
        batch=batch.to(torch.int)
    return Batch(embeddings,ei_mod.contiguous(),batch=batch).to(device)

def filter_hydrogen(dataset):
    ls=[]
    for data in dataset:
        condition=data.x[:,0]==1
        filtered_nodes = torch.where(condition)[0]
        filtered_edge_mask = ~torch.any(torch.isin(data.edge_index, filtered_nodes[:, None]), dim=0)

        src,tgt=data.edge_index[:, filtered_edge_mask]
        val=0
        for node in filtered_nodes:
            src_mask=src>node-val
            tgt_mask=tgt>node-val
            if torch.any(src_mask==True):
                val+=1
            src[src_mask]-=1
            tgt[tgt_mask]-=1
        edge_index=data.edge_index[:, filtered_edge_mask] if torch.numel(src)==0 else torch.stack((src,tgt),dim=0)

        ls.append(Data(x=data.x[~condition,1:5],edge_index=edge_index,
                       edge_attr=data.edge_attr[ filtered_edge_mask]))

    return ls

def plot_data_from_batch(data,i):
    _,m=to_dense_batch(data.x,data.batch)
    d2=to_dense_adj(data.edge_index,edge_attr= data.edge_attr,batch=data.batch)
    d2=d2[i]
    d2=d2[m[i]]
    d2=d2[:,m[i]]
    d2=torch.argmax(d2,dim=2).numpy()
    G2=nx.from_numpy_array(d2)
    nx.draw(G2,with_labels=True)

def dense_batch_to_sparse(nodes,edges,add_self_loops=False):
    n=nodes.size(1)
    ei_adj=torch.ones((n,n))
    if not add_self_loops:
        ei_adj.fill_diagonal_(0)
    edge_index=ei_adj.nonzero().t()
    data_ls=[]
    for x,edge_adj, in zip (nodes,edges):
        edge_attr=edge_adj[edge_index[0],edge_index[1]]
        
        data=Data(x,edge_index,edge_attr=edge_attr)
        data_ls.append(data)
        
    return Batch.from_data_list(data_ls)

def batch_to_data_ls(data):
    m=torch.arange(0,data.x.size(0))
    data_ls=[]
    max=torch.max(data.batch)
    for i in range(max+1):
        ei,attr=subgraph(m[data.batch==i],data.edge_index,edge_attr=data.edge_attr,relabel_nodes=True)
        d=Data(data.x[data.batch==i],ei,edge_attr=attr)
        data_ls.append(d)
    return data_ls

def fully_connected_data(data):
    edge_dim=5
    edge_attr=torch.cat((torch.tensor([0]).repeat(data.edge_attr.size(0)).unsqueeze(1),data.edge_attr),dim=1)
    edge_to_attr = {(src.item(), tgt.item()): attr 
            for src, tgt, attr in zip(data.edge_index[0], data.edge_index[1], edge_attr.squeeze())}  
   
    # Extract nodes for this graph
    num_nodes = data.x.size(0)
    
    # Create fully connected edges for these nodes
    edge_index = torch.ones((num_nodes,num_nodes)).fill_diagonal_(0).nonzero().t()

    attributes = [edge_to_attr.get((src.item(), tgt.item()), torch.tensor([1,0,0,0,0])) 
                    for src, tgt in zip(edge_index[0], edge_index[1])]

    # Concatenate all edge indices and attributes for the batch
    edge_attr = torch.stack(attributes, dim=0) if len(attributes)!=0 else torch.empty(0, edge_dim)
    
    new_data = Data(x=data.x, edge_index=edge_index, edge_attr=edge_attr)
    
    return new_data

def extract_subgraph(data,idx):
    filtered_edge_mask = torch.isin(data.edge_index, idx)
    filtered_edge_mask=filtered_edge_mask[0]&filtered_edge_mask[1]

    src,tgt=data.edge_index[:, filtered_edge_mask]
    val=0
    condition=torch.isin(torch.arange(0,data.x.size(0)),idx)
    filtered_nodes = torch.where(~condition)[0]

    for node in filtered_nodes:
        src_mask=src>node-val
        tgt_mask=tgt>node-val
        if torch.any(src_mask==True):
            val+=1
        src[src_mask]-=1
        tgt[tgt_mask]-=1
    edge_index=data.edge_index[:, filtered_edge_mask] if torch.numel(src)==0 else torch.stack((src,tgt),dim=0)
 
    x=data.x if data.x.size(0)==1 else data.x[idx]

    return Data(x=x,edge_index=edge_index,
            edge_attr=data.edge_attr[ filtered_edge_mask],batch=torch.tensor([0 for _ in range(idx.size(0))]))


def reduce_fc_graph(data):
    edge_mask=torch.argmax(data.edge_attr,dim=1)!=0
    return Data(data.x,data.edge_index[:,edge_mask],data.edge_attr[edge_mask])

def rm_non_edges(data,reduce=False):
    #from torch_geometric.utils import subgraph
    edge_mask=torch.argmax(data.edge_attr,dim=1)!=0
    edge_index=data.edge_index[:,edge_mask]
    attr=data.edge_attr[edge_mask]
    #edges_idx=torch.unique(edge_index[0],sorted=True)

    #edge_index,attr=subgraph(edges_idx,edge_index,attr,relabel_nodes=True)
    if reduce:
        attr=attr[:,1:]
    #embeddings=data.x[edges_idx] if edge_index.numel()>0 else data.x
    return Batch(data.x,edge_index=edge_index,edge_attr=attr,batch=data.batch)

def keep_largest_component(data):
    edge_index = data.edge_index
    x = data.x
    edge_attr = data.edge_attr
    num_nodes = data.num_nodes
    
    # Convert edge_index to NetworkX graph
    edge_index = edge_index.cpu().numpy()
    G = nx.Graph()
    G.add_edges_from(edge_index.T)

    # Find connected components
    connected_components = list(nx.connected_components(G))

    # If only one connected component, return original data
    if len(connected_components) == 1:
        return data

    # Keep the largest connected component
    largest_component = max(connected_components, key=len)

    # Create a mask to filter out nodes not in the largest component
    mask = torch.tensor([node in largest_component for node in range(num_nodes)], dtype=torch.bool)

    # Filter out edges based on the mask
    filtered_edge_index = edge_index[:, [i for i in range(edge_index.shape[1]) if mask[edge_index[0, i]] and mask[edge_index[1, i]]]]

    # Filter out node features
    filtered_x = x[mask]

    # Filter out edge features
    if edge_attr is not None:
        filtered_edge_attr = edge_attr[[i for i in range(edge_attr.shape[0]) if mask[edge_index[0, i]] and mask[edge_index[1, i]]]]
    else:
        filtered_edge_attr = None

    # Update data object
    filtered_edge_index = torch.tensor(filtered_edge_index)
    batch=torch.tensor([0 for _ in range(filtered_x.size(0))])

    return Data(filtered_x,filtered_edge_index,edge_attr=filtered_edge_attr,batch=batch)


def extract_largest_smiles_string(smiles):
    molecule_ls = smiles.split('.')
    longest_index = 0
    longest_length = len(molecule_ls[0])

    for i in range(1, len(molecule_ls)):
        if len(molecule_ls[i]) > longest_length:
            longest_index = i
            longest_length = len(molecule_ls[i])

    return molecule_ls[longest_index]

def add_virtual_nodes(batch):
    vn=VirtualNode()
    data_ls=batch.to_data_list()
    ls=[]
    for data in data_ls:
        ls.append(vn(data))
    return Batch.from_data_list(ls)
def batch_to_list(data):
    ls=[]
    count=0
    for i in range(torch.max(data.batch)+1):
        try:
            mask=data.batch==i
            idx=torch.arange(count,count+torch.sum(mask))
            count+=torch.sum(mask)
            ei,attr=subgraph(idx,data.edge_index,data.edge_attr,relabel_nodes=True)
            temp=Data(data.x[idx],ei,attr,torch.tensor([0 for _ in range(torch.sum(mask))]))
            temp=oh_encode_reconstructed_data(temp)
            ls.append(temp)
        except:
            ls.append(None)
    return ls

def oh_encode_reconstructed_data(data):
    nodes_idx=torch.nonzero(data.x[:,0]<0.5).view(-1)
    edge_index,attr=subgraph(nodes_idx,data.edge_index,data.edge_attr,relabel_nodes=True)
    x=data.x[nodes_idx,1:]
    #Remove non-type edges and remove isolated nodes
    edge_index,attr=to_undirected(edge_index,attr,reduce="mean")

    edge_mask=torch.argmax(attr,dim=1)!=0
    edge_index=edge_index[:,edge_mask]
    attr=attr[edge_mask][:,1:]
    edges_idx=torch.unique(edge_index[0])
    edge_index,attr=subgraph(edges_idx,edge_index,attr,relabel_nodes=True)
    
    return Data(F.one_hot(torch.argmax(x,dim=1),num_classes=4).to(torch.float),edge_index,
                        edge_attr=F.one_hot(torch.argmax(attr,dim=1),num_classes=4),batch=data.batch)


def create_dataset(dataset_smiles):
    data_ls=[]
    for smiles in dataset_smiles:
        mol = Chem.MolFromSmiles(smiles)
        atom_dict={1:[1,1,0,0,0,0],6:[1,0,1,0,0,0],7:[1,0,0,1,0,0],8:[1,0,0,0,1,0],9:[1,0,0,0,0,1]}
        node_features = []
        for atom in mol.GetAtoms():
            atomic_num=atom.GetAtomicNum()
            atom_features = atom_dict[atomic_num]
            node_features.append(atom_features)
        bond_dict={Chem.BondType.SINGLE:[-1,0,1,0,0,0],Chem.BondType.AROMATIC:[-1,0,0,1,0,0],
            Chem.BondType.DOUBLE:[-1,0,0,0,1,0],Chem.BondType.TRIPLE:[-1,0,0,0,0,1]}
        edges = []

        dic={}
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type=bond.GetBondType()
            dic[(start,end)]=bond_dict[bond_type]
            dic[(end,start)]=bond_dict[bond_type]

        k=len(node_features)
        num_atoms=len(node_features)
        for i in range(num_atoms):
            for j in range(i,num_atoms):
                if i!=j:
                    edges.append([i, k])
                    edges.append([k, i])
                    edges.append([k, j])
                    edges.append([j, k])
                    k+=1
                else:
                    edges.append([i,j])
        for i in range(num_atoms):
            for j in range(i+1,num_atoms):
                    x=dic.get((i,j),[-1,1,0,0,0,0])
                    node_features.append(x)

        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous() if len (edges)>1 else torch.empty((2,0),dtype=torch.long)

        node_features=torch.tensor(node_features)
        data=Data(x=node_features,edge_index=edge_index)
        data_ls.append(data)
    return data_ls
def compute_gradients(model):
    """
    Computes the gradients for the encoder and decoder separately and calculates their norms.

    Args:
        model: The VAE model containing encoder and decoder.

    Returns:
        encoder_norm: Norm of the encoder gradients.
        decoder_norm: Norm of the decoder gradients.
    """

    
    encoder_norm = 0.0
    decoder_norm = 0.0

    aggregation_norm=0.0
    latent_norm=0.0

    # Loop through model parameters
    for name, param in model.named_parameters():
        if param.grad is not None:  # Ensure the gradients exist
            if 'encoder' in name:

                encoder_norm += param.grad.data.norm(2).item() ** 2  # L2 norm
            elif 'decoder' in name:
                decoder_norm += param.grad.data.norm(2).item() ** 2  # L2 norm
            elif 'aggregation' in name:
                aggregation_norm+=param.grad.data.norm(2).item() ** 2  # L2 norm
            elif 'linear_mu' in name or 'linear_sigma':
                latent_norm+=param.grad.data.norm(2).item() ** 2  # L2 norm



    # Take the square root of the sums to get the final norms
    encoder_norm = encoder_norm ** 0.5
    decoder_norm = decoder_norm ** 0.5

    return encoder_norm, decoder_norm,aggregation_norm,latent_norm
