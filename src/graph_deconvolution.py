import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.utils import softmax,to_dense_batch
from torch_scatter import scatter_max,scatter_add,scatter_mean,scatter


def message_aggregation(messages, attn_probs, edge_index,concat=False):
    """
    messages: Pre-computed messages of shape [num_edges, message_dim]
    attn_probs: Attention probabilities of shape [num_edges, heads]
    edge_index: Indices of source and target nodes for each edge, shape [2, num_edges]
    """
    
    num_heads = attn_probs.size(1)
    
    # Extend message dimension for heads
    messages_extended = messages.unsqueeze(1).expand(-1, num_heads, -1)  # [num_edges, heads, message_dim]
    
    # Weighting the messages
    weighted_messages = attn_probs * messages_extended
    
    # Aggregating the messages for each target node using scatter_add
    aggregated_messages_list = []
    for i in range(num_heads):
        aggregated_message = scatter_add(weighted_messages[:, i], edge_index[1], dim=0)
        aggregated_messages_list.append(aggregated_message)
    
    aggregated_messages = torch.stack(aggregated_messages_list, dim=1)  # [num_nodes, heads, message_dim]
    if concat:
        aggregated_messages=aggregated_messages.view(aggregated_messages.size(0),-1)
    else:
        aggregated_messages=torch.mean(aggregated_messages,dim=1)
    
    return aggregated_messages


class MessagePassingLayer(nn.Module):
    def __init__(self, in_dim,out_dim, edge_embedding_dim):
        super(MessagePassingLayer, self).__init__()

        # Define the weight matrices
        self.W_e = nn.Linear(edge_embedding_dim, out_dim, bias=False) 
        self.W_hu = nn.Linear(in_dim, out_dim, bias=False) 
        self.W_hw = nn.Linear(in_dim, out_dim, bias=False)

        nn.init.kaiming_uniform_(self.W_e.weight, a=0.01, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.W_hu.weight, a=0.01, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.W_hw.weight, a=0.01, nonlinearity='leaky_relu')

    def forward(self, h,edge_index,e):
        """
        h: Node embeddings of shape [num_nodes, node_embedding_dim]
        e: Edge embeddings of shape [num_edges, edge_embedding_dim]
        edge_index: Indices of source and target nodes for each edge, shape [2, num_edges]
        """
        
        # Get source and target node indices for each edge
        src, tgt = edge_index
        # Compute the messages
        messages =self.W_e(e) + self.W_hu(h[src]) + self.W_hw(h[tgt])
        return messages

class AttentionMechanism(nn.Module):
    def __init__(self, message_dim,heads):
        super(AttentionMechanism, self).__init__()

        # Define the weight matrix for attention mechanism
        self.W = nn.Linear(message_dim, message_dim*heads, bias=False)
        self.heads=heads
        self.message_dim=message_dim

    def forward(self, messages, edge_index):
        """
        messages: Pre-computed messages of shape [num_edges, message_dim]
        edge_index: Indices of source and target nodes for each edge, shape [2, num_edges]
        """
        
        # Compute attention logits for each head
        attn_logits = self.W(messages)  # [num_edges, heads]
        attn_logits=attn_logits.view(attn_logits.size(0),self.heads,-1)
        
        attn_probs_list = []
        for i in range(self.heads):
            single_head_logits = attn_logits[:, i]

            # Calculate max for each source node for numerical stability
            max_per_source_node = scatter(single_head_logits, edge_index[0],dim=0,reduce="max")
            attn_logits_maxed = single_head_logits - max_per_source_node[edge_index[0]]

            # Compute attention weights using the given formula
            attn_weights = torch.exp(attn_logits_maxed)
            attn_sum = scatter(attn_weights, edge_index[0],dim=0,reduce="sum")
            attn_probs = attn_weights / attn_sum[edge_index[0]]

            attn_probs_list.append(attn_probs)

        # Stack the attention probabilities from all heads
        attn_probs_stacked = torch.stack(attn_probs_list, dim=1)  # [num_edges, heads]
        
        return attn_probs_stacked

    def custom_scatter_max(self, src, index, dim_size=None):
        if dim_size is None:
            dim_size = index.max().item() + 1
        
        out, _ = scatter_max(src, index,dim=0, dim_size=dim_size)
        
        return out

    def custom_scatter_sum(self, src, index, dim_size=None):
        if dim_size is None:
            dim_size = index.max().item() + 1
        out = src.new_full((dim_size, ), 0).scatter_add_(0, index, src)
        return out
    
class GATAttention(nn.Module):
    def __init__(self, features,edge_dim, num_heads, dropout=0.0, negative_slope=0.2):
        super(GATAttention, self).__init__()

        self.features = features
 
        self.num_heads = num_heads
        self.dropout = dropout
        self.edge_dim=edge_dim
        self.negative_slope = negative_slope
        self.act=nn.LeakyReLU(negative_slope)

        # Learnable parameters (linear transformation weights)
        self.W_s = Parameter(torch.Tensor(features, num_heads * features))
        self.W_t = Parameter(torch.Tensor(features, num_heads * features))

        self.att = Parameter(torch.empty(1, num_heads, features))
        self.edge_embedding = nn.Linear(edge_dim,  num_heads*features)

        # Initialization
        nn.init.kaiming_uniform_(self.W_s.data, a=0.01, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.W_t.data, a=0.01, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.att.data, a=0.01, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.edge_embedding.weight, a=0.01, nonlinearity='leaky_relu')
        

    def forward(self, x, edge_index, edge_attr):
        x_i = x[edge_index[0]]
        x_i=x_i@self.W_s
        x_i=x_i.view(x_i.size(0),self.num_heads,self.features)  
        x_j = x[edge_index[1]]
        x_j=x_j@self.W_t
        x_j=x_j.view(x_j.size(0),self.num_heads,self.features)
        e_ij=self.edge_embedding(edge_attr)
        e_ij=e_ij.view(e_ij.size(0),self.num_heads,self.features)
        x=self.act(x_i+x_j+e_ij)
        alpha=(x * self.att)
        alpha = alpha.sum(dim=-1)       
        index=edge_index[1]
        alpha = softmax(alpha, index)
        return alpha


    
class GNN_Layer(nn.Module):
    def __init__(self,node_dim,out_dim,edge_dim,heads=4,concat=False):
        super(GNN_Layer,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.concat=concat
        self.message_passing=MessagePassingLayer(in_dim=node_dim,out_dim=out_dim,edge_embedding_dim=edge_dim)
        self.attention_mechanism=AttentionMechanism(out_dim,heads=heads)
        if concat:
            self.gru_cell=nn.GRUCell(node_dim,out_dim*heads)
        else:
            self.gru_cell=nn.GRUCell(node_dim,out_dim)

        self.act=nn.LeakyReLU()
        

    def forward(self,x,edge_index,edge_attr,batch):
        messages=self.act(self.message_passing(x,edge_index,edge_attr)).to(self.device)
        attention=self.attention_mechanism(messages,edge_index)
        agg_message=message_aggregation(messages,attention,edge_index,concat=self.concat)
        x,x_masks=to_dense_batch(x,batch)
        batched_agg_message,_=to_dense_batch(agg_message,batch)
        output = []
        for i in range(x.size(0)):
            hx=self.gru_cell(x[i][x_masks[i]],batched_agg_message[i][x_masks[i]])
            output.append(hx)

        embeddings=torch.cat(output,dim=0)
        return embeddings,messages
    
class GNN_Layer2(nn.Module):
    def __init__(self,node_dim,edge_dim,heads=4):
        super(GNN_Layer2,self).__init__()
        self.message_passing=MessagePassingLayer(node_embedding_dim=node_dim,edge_embedding_dim=edge_dim)
        self.attention_mechanism=GATAttention(node_dim,edge_dim,num_heads=heads)

        self.weight=nn.Linear(node_dim,node_dim,bias=False)
        self.lin_mes=nn.Linear(node_dim,edge_dim,bias=False)
        self.act=nn.LeakyReLU()

        nn.init.kaiming_uniform_(self.weight.weight, a=0.01, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.lin_mes.weight, a=0.01, nonlinearity='leaky_relu')
        

    def forward(self,x,edge_index,edge_attr):
        messages=self.act(self.message_passing(x,edge_index,edge_attr))
        attributes=edge_attr+messages
        attributes=self.lin_mes(attributes)
        #Should I use the updated attributes?
        attention=self.attention_mechanism(x,edge_index,attributes)
        agg_message=message_aggregation(messages,attention,edge_index,concat=False)
        agg_message=self.lin_mes(agg_message)
        embeddings=x+self.act(self.weight(x)+agg_message)
        
        return embeddings,attributes
    
    
class GNN_Layer3(nn.Module):
    def __init__(self,in_dim,edge_dim,out_dim,edge_out_dim=None):
        super(GNN_Layer3,self).__init__()
        if edge_out_dim is  None:
            edge_out_dim=edge_dim

        self.message_passing=MessagePassingLayer(in_dim=in_dim,out_dim=out_dim,edge_embedding_dim=edge_dim)
        #self.weight1=nn.Linear(edge_dim,out_dim)
        self.weight2=nn.Linear(in_dim,out_dim,bias=False)
        self.lin_emb=nn.Linear(in_dim,out_dim,bias=False)
        self.lin_attr=nn.Linear(out_dim,edge_out_dim,bias=False)
        self.act=nn.LeakyReLU()

        #nn.init.kaiming_uniform_(self.weight1.weight, a=0.01, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.weight2.weight, a=0.01, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.lin_emb.weight, a=0.01, nonlinearity='leaky_relu')
        

    def forward(self,x,edge_index,edge_attr):
        messages=self.act(self.message_passing(x,edge_index,edge_attr))
        #attributes=edge_attr+messages
        attributes=messages
        #attributes=self.weight1(attributes)
        #Should I use the updated attributes?
        #agg_message_sum = scatter_add(attributes, edge_index[1], dim=0)
        agg_message=scatter_mean(attributes,edge_index[1],dim=0)
        #agg_message=torch.stack([agg_message_mean,agg_message_sum],dim=1).squeeze(-1)
        #agg_message=torch.mean(agg_message,dim=1)
        #Transform to right dimensionality
        attributes=self.lin_attr(attributes)
        embeddings=self.lin_emb(x)+self.act(self.weight2(x)+agg_message)
     
        return embeddings,attributes
    

class GNN_Layer4(nn.Module):
    def __init__(self,node_dim,out_dim,edge_dim,heads=4,concat=False):
        super(GNN_Layer4,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.concat=concat
        self.message_passing=MessagePassingLayer(in_dim=node_dim,out_dim=out_dim,edge_embedding_dim=edge_dim)
        self.attention_mechanism=AttentionMechanism(out_dim,heads=heads)
        self.act=nn.LeakyReLU()
        self.lin=nn.Linear(out_dim,out_dim)
        
        self.lin2=nn.Linear(node_dim,out_dim)if node_dim!=out_dim else nn.Identity()
        self.lin3=nn.Linear(edge_dim,out_dim)if edge_dim!=out_dim else nn.Identity()

    def forward(self,x,edge_index,edge_attr,batch):
        messages=self.act(self.message_passing(x,edge_index,edge_attr)).to(self.device)
        attention=self.attention_mechanism(messages,edge_index)
        agg_message=message_aggregation(messages,attention,edge_index,concat=self.concat)
        embeddings=self.lin2(x)+self.act(self.lin(agg_message))

        attr=(self.lin3(edge_attr)+messages)/2
        return embeddings,attr
    