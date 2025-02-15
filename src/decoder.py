import torch
import torch.nn as nn
from torch_geometric.data import Data,Batch

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
class TransformerGraphInit(nn.Module):
    def __init__(self, latent_dim, embed_dim, n_heads, n_layers,norm):
        super(TransformerGraphInit, self).__init__()
        
        # Linear layer to transform one-hot node identifiers to embeddings
        self.type_embedd = nn.Linear(1, 8)
        
        # Latent vector transformation
        self.latent_linear = nn.Linear(latent_dim, embed_dim-8-18)
    
    
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,batch_first=True,dropout=0.01)

    
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,norm=norm)
        
        # Output MLP (optional)
        self.output_mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256, 5))
    
    def forward(self, node_features, latent_vector, src_key_padding_mask):
        # Transform one-hot node identifiers to embeddings
        #node_features = self.node_embed(node_ids_onehot)  # Shape: [batch_size, max_nodes, embed_dim]

        # Integrate latent vector by adding it to node features
        type_embedds=self.type_embedd(node_features[:,:,0].unsqueeze(-1))
        node_features=torch.cat((type_embedds,node_features[:,:,1:]),dim=2)


        latent_embed = self.latent_linear(latent_vector).unsqueeze(1).expand(-1, node_features.size(1), -1)  #Shape: [batch_size, 1, embed_dim]

        combined_features = torch.cat((node_features, latent_embed), dim=-1)
        # Pass through Transformer (transpose to shape [max_nodes, batch_size, embed_dim])

        transformer_out = self.transformer(combined_features, src_key_padding_mask=src_key_padding_mask)
        
        
        # Optional MLP for refined output
        graph_init = self.output_mlp(transformer_out)  # Shape: [max_nodes, batch_size, embed_dim]
        
        return graph_init,transformer_out
    
class TransformerDecoder(nn.Module):
    def __init__(self,latent_dim,embed_dim,attention_heads,layers,norm=None):
        super(TransformerDecoder,self).__init__()
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.num_net=NodeNumPrediction(latent_dim).to(self.device)

        self.skeleton=self.precompute_skeletons()

        self.embed_dim=embed_dim
        if norm=="layer":
            norm=nn.LayerNorm(self.embed_dim)
        self.transformer=TransformerGraphInit(latent_dim=latent_dim,embed_dim=self.embed_dim,n_heads=attention_heads,n_layers=layers,norm=norm)


    def forward(self,global_vec,data=None):
        pred_num=self.num_net(global_vec).view(-1)


        if data is not None:
            num_nodes=torch.bincount(data.batch)

            num_nodes=0.5*(torch.sqrt(8*num_nodes+1)-1)
            num_nodes=num_nodes.to(int)
        else:
            num_nodes=torch.round(pred_num).to(int)
            if num_nodes<1 or num_nodes>9:
                return None,None


        batch=self.init_batch(num_nodes)
        src,src_mask=self.transformer_encoding(batch)
        node_embeddings,_=self.transformer(src,global_vec,src_mask)
        node_embeddings=node_embeddings[~src_mask]
        batch.x=node_embeddings

        return batch,pred_num
    
    def init_graph(self,num_nodes,x):
        skeleton=self.skeleton[num_nodes.item()]

        edge_index=skeleton['edge_index']
        identifier=skeleton['intermediaries']
        num=skeleton['num_nodes']
        idx=skeleton['feature_idx']

        if num>1:
            edge_features=torch.cat((x[idx[0]],x[idx[1]]),dim=1)
            node_features=torch.cat((x[:num_nodes],x[:num_nodes]),dim=1)
            x=torch.cat((node_features,edge_features),dim=0)
        else:
            x=torch.cat((x[0].unsqueeze(0),x[0].unsqueeze(0)),dim=1)

        x=torch.cat((identifier.unsqueeze(1),x),dim=1)
        return Data(x=x,edge_index=edge_index)
    
    def init_batch(self,num_nodes):
        ls=[]
        x=self.batch_orthogonal_random_matrices(num_nodes.size(0),9)

        for idx,num in enumerate(num_nodes):
            data=self.init_graph(num,x[idx])
            ls.append(data)
        return Batch.from_data_list(ls)
    
    def transformer_encoding(self,data):
        src_mask=torch.ones((data.batch_size,45),device=self.device,dtype=torch.bool)
        src=torch.ones((data.batch_size,45,19),device=self.device,dtype=torch.float)
        
        for i in range(data.batch_size):
            mask=data.batch==i
            x=data.x[mask]
            src_mask[i,:x.size(0)]=False
            src[i,:,:][~src_mask[i]]=x
        return src,src_mask
    
    def batch_orthogonal_random_matrices(self,batch_size, n):
        """
        Generate orthogonal random noise for positional encoding.
        """
        # Step 1: Generate a batch of random Gaussian matrices
        G = torch.randn(batch_size, n, n, device=self.device)
        # Step 2: Perform batched QR decomposition
        Q, _ = torch.linalg.qr(G)

        return Q
    
    def precompute_skeletons(self,max_n=9):
        """
        This function provides a graph skeleton for different sizes, that is featurized by the decoder output.
        """
        skeletons = {}
        for n in range(1, max_n + 1):
            intermediaries = []
            
            for i in range(1,n+1):
                x=torch.zeros(1,device=DEVICE,dtype=torch.float32)
                x[0]=1.
                intermediaries.append(x)

            edge_index = []
            feature_idx=[]
            k = n
            for i in range(n):
                for j in range(i + 1, n):
                    feature_idx.extend([[i,j]])
                    edge_index.extend([[i, k], [k, i], [k, j], [j, k]])
                    x=torch.zeros(1,device=DEVICE,dtype=torch.float32)
                    x[0]=-1.
                    k += 1
                    intermediaries.append(x)
                edge_index.append([i, i])  # Add self-loop
            
            # Convert to tensor
            edge_index = torch.tensor(edge_index, dtype=torch.long,device=DEVICE).t()
            intermediaries = torch.cat(intermediaries)
            feature_idx=torch.tensor(feature_idx, dtype=torch.long,device=DEVICE).t()
            skeletons[n] = {'edge_index': edge_index, 'intermediaries': intermediaries,'feature_idx':feature_idx,
                            'num_nodes':int(n+(n**2-n)/2)}
        return skeletons
    