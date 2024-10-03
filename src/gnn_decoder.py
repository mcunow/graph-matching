from typing import Optional, Tuple
import  graph_deconvolution
import utils
import gnn
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GATConv
from torch_geometric.nn import PairNorm,LayerNorm
from torch_geometric.data import Data,Batch
from torch_geometric.nn.dense import DenseGATConv
from torch_geometric.utils import add_self_loops,remove_self_loops,subgraph,to_undirected
from torch_geometric.nn.resolver import (
    activation_resolver)

class Sinkhorn_Decoder6(nn.Module):
    def __init__(self,latent_dim,node_dim,edge_dim,num_layers,act="leakyrelu", dropout=0.0):
        super(Sinkhorn_Decoder6, self).__init__()
        self.act = activation_resolver(act, **( {}))
        self.hidden_channels=[64,32,16]
        self.edge_dim=edge_dim
        self.num_layers=num_layers
        #self.latent_transform=nn.Sequential(nn.Linear(latent_dim,hidden_dim,bias=False))

        self.max_nodes=9
        self.mlp_init= Generator(z_dim=latent_dim,output_dim=self.hidden_channels[-2],vertexes=self.max_nodes,conv_dims=[128, 256, 512],dropout_rate=dropout)

        self.Convlayers=nn.ModuleList()
        self.Convlayers.append(graph_deconvolution.GNN_Layer(node_dim=self.hidden_channels[-2],out_dim=self.hidden_channels[-2],
                                                                  edge_dim=self.hidden_channels[-1],heads=4,concat=False))
        for _ in range(num_layers-1):
            self.Convlayers.append(graph_deconvolution.GNN_Layer(node_dim=self.hidden_channels[-2],out_dim=self.hidden_channels[-2],
                                                                  edge_dim=self.hidden_channels[-2],heads=4,concat=False))
        
        self.feature_mlp=nn.Sequential(nn.Linear(self.hidden_channels[-2],64,bias=False),
                            self.act,
                            nn.Linear(64,128,bias=False),
                            self.act,
                            nn.Linear(128,node_dim,bias=False)
        )
        self.edge_mlp=nn.Sequential(nn.Linear(self.hidden_channels[-2],64,bias=False),
                    self.act,
                    nn.Linear(64,128,bias=False),
                    self.act,
                    nn.Linear(128,edge_dim,bias=False)
        )
        self.node_num_layer=nn.Sequential(nn.Linear(latent_dim,self.hidden_channels[0]),
                        self.act,
                        nn.Linear(self.hidden_channels[0],self.hidden_channels[1]),
                        self.act,
                        nn.Linear(self.hidden_channels[1],1)
        )
    def forward(self,latent_vec,data):
        number_nodes= self.node_num_layer(latent_vec).view(-1)
        nn_int=torch.bincount(data.batch) if self.training else number_nodes.to(torch.int)
        nn_int=nn_int[nn_int>0]
        
        #Latent transformation
        #projected_vec=F.sigmoid(self.latent_transform(latent_vec))

        embedding_dense=self.mlp_init(latent_vec).view(-1,self.hidden_channels[-2])

        mask=torch.zeros(nn_int.size(0)*self.max_nodes,dtype=bool)
        ind=0
        for i in nn_int:
            mask[torch.arange(ind,ind+i,dtype=torch.int)]=True
            ind+=self.max_nodes
        embeddings=embedding_dense[mask]

        edge_index,attr,batch=utils.fully_connected_graph(nn_int,edge_dim=self.edge_dim,
                                                          fill_value=torch.tensor([0 for _ in range(self.hidden_channels[-1])],dtype=torch.float))
        #edge_index,attr=add_self_loops(edge_index,attr,fill_value=torch.tensor([0,1,1,1,1]))
        
        #Message passing
        for i in range(self.num_layers-1):
            #Update node embeddings
            embeddings,attr=self.Convlayers[i](embeddings,edge_index,edge_attr=attr,batch=batch)
            #embeddings=F.relu(embeddings)
        embeddings,attr=self.Convlayers[i](embeddings,edge_index,edge_attr=attr,batch=batch)
        #Remove self loops
        # edge_index=edge_index[:,:-sum_num_nodes]
        # attr=attr[:-sum_num_nodes]
    
        embeddings=self.feature_mlp(embeddings)
        attr=self.edge_mlp(attr)

        return Batch(embeddings,edge_index,edge_attr=attr,batch=batch),number_nodes

class Sinkhorn_Decoder5(nn.Module):
    def __init__(self,latent_dim,node_dim,edge_dim=5,act="leakyrelu", dropout=0.0):
        super(Sinkhorn_Decoder5, self).__init__()
        self.act = activation_resolver(act, **( {}))
        self.hidden_channels=[128,64,32,16]
        #self.latent_transform=nn.Sequential(nn.Linear(latent_dim,hidden_dim,bias=False))

        self.max_nodes=9
        self.mlp_init= Generator(z_dim=latent_dim,output_dim=self.hidden_channels[-2],vertexes=self.max_nodes,conv_dims=[128, 256, 512],dropout_rate=dropout)


        self.gnn=gnn.GIN(in_channels=self.hidden_channels[-2]+edge_dim,hidden_channels=[32,32,64,64,128,128,128],
                         out_channels=self.hidden_channels[0],act="leakyrelu",num_layers=8,norm="batchnorm",add_self_loops=True)
        
        self.final_mlp=nn.Sequential(nn.Linear(self.hidden_channels[0],256,bias=False),
                            self.act,
                            nn.Linear(256,128,bias=False),
                            self.act,
                            nn.Linear(128,node_dim+edge_dim,bias=False) 
        )
        self.node_num_layer=nn.Sequential(nn.Linear(latent_dim,self.hidden_channels[0]),
                        self.act,
                        nn.Linear(self.hidden_channels[0],self.hidden_channels[1]),
                        self.act,
                        nn.Linear(self.hidden_channels[1],1)
        )
    def forward(self,latent_vec,data):
        number_nodes= self.node_num_layer(latent_vec).view(-1)
        nn_int=torch.bincount(data.batch) if self.training else number_nodes.to(torch.int)

        nn_int[nn_int>=self.max_nodes]=self.max_nodes
        nn_int[nn_int==0]=1
        
        #Latent transformation
        #projected_vec=F.sigmoid(self.latent_transform(latent_vec))

        embedding_dense=self.mlp_init(latent_vec).view(-1,self.hidden_channels[-2])

        mask=torch.zeros(nn_int.size(0)*self.max_nodes,dtype=bool)
        ind=0
        for i in nn_int:
            mask[torch.arange(ind,ind+i,dtype=torch.int)]=True
            ind+=self.max_nodes

        embeddings=embedding_dense[mask]
    
        data=utils.create_fc_graph_with_nodes_as_edges_from_matrix(nn_int,embeddings)
        embeddings=self.gnn(data)
        embeddings=self.final_mlp(embeddings)
        
        data= utils.reduce_graph(Batch(embeddings,data.edge_index,batch=data.batch))
        if not self.training:
            edge_mask=torch.argmax(data.edge_attr,dim=1)!=0
            nodes_idx=torch.unique(data.edge_index[0,edge_mask])
            data_ex=utils.extract_subgraph(Data(data.x,data.edge_index,data.edge_attr),nodes_idx)
            assert(torch.equal(nodes_idx,torch.unique(data.edge_index[1,edge_mask])))
            return data_ex,number_nodes
            
    

        return data,number_nodes
    



class Sinkhorn_Decoder4(nn.Module):
    def __init__(self,latent_dim,node_dim,hidden_dim=64,edge_dim=5,act="leakyrelu", dropout=0.0):
        super(Sinkhorn_Decoder4, self).__init__()
        self.act = activation_resolver(act, **( {}))
        self.hidden_channels=[64,32,16]
        self.latent_transform=nn.Sequential(nn.Linear(latent_dim,hidden_dim,bias=False))
        self.rnn_hidden_dim=self.hidden_channels[1]
        self.gru_cell = GraphReadInGRU(hidden_dim,self.rnn_hidden_dim,output_dim=self.hidden_channels[1],max_nodes=9)

        self.gnn=gnn.GIN(in_channels=self.hidden_channels[1]+edge_dim,hidden_channels=[32,32,64,64],
                         out_channels=self.hidden_channels[-2],act="leakyrelu",num_layers=4,norm="batchnorm")
        
        self.final_mlp=nn.Sequential(nn.Linear(self.hidden_channels[-2],self.hidden_channels[-2],bias=False),
                            self.act,
                            nn.Linear(self.hidden_channels[-2],self.hidden_channels[-2],bias=False),
                            self.act,
                            nn.Linear(self.hidden_channels[-2],node_dim+edge_dim,bias=False)
        )
        self.node_num_layer=nn.Sequential(nn.Linear(latent_dim,self.hidden_channels[0]),
                        self.act,
                        nn.Linear(self.hidden_channels[0],self.hidden_channels[1]),
                        self.act,
                        nn.Linear(self.hidden_channels[1],1)
        )
    def forward(self,latent_vec,data):
        number_nodes= self.node_num_layer(latent_vec).view(-1)
        nn_int=torch.bincount(data.batch) if self.training else number_nodes.to(torch.int)
        
        #Latent transformation
        projected_vec=F.sigmoid(self.latent_transform(latent_vec))

        embeddings=self.gru_cell(projected_vec,nn_int)

        data=utils.create_fc_graph_with_nodes_as_edges_from_matrix(nn_int,embeddings)
        embeddings=self.gnn(data)
        embeddings=self.final_mlp(embeddings)

        return Batch(embeddings,data.edge_index,batch=data.batch),number_nodes

class Sinkhorn_Decoder3(nn.Module):
    def __init__(self,latent_dim,node_dim,hidden_dim=64,edge_dim=5,act="leakyrelu", dropout=0.0):
        super(Sinkhorn_Decoder3, self).__init__()
        self.act = activation_resolver(act, **( {}))
        self.hidden_channels=[64,32,16]
        self.latent_transform=nn.Sequential(nn.Linear(latent_dim,hidden_dim,bias=False))
        self.rnn_hidden_dim=self.hidden_channels[1]
        self.lstm_cell = GraphReadInLSTM(hidden_dim,self.rnn_hidden_dim)

        self.gnn=gnn.GIN(in_channels=self.hidden_channels[1]+edge_dim,hidden_channels=[32,32,64,64],
                         out_channels=self.hidden_channels[-2],act="leakyrelu",num_layers=4,norm="batchnorm")
        
        self.final_mlp=nn.Sequential(nn.Linear(self.hidden_channels[-2],self.hidden_channels[-2],bias=False),
                            self.act,
                            nn.Linear(self.hidden_channels[-2],self.hidden_channels[-2],bias=False),
                            self.act,
                            nn.Linear(self.hidden_channels[-2],node_dim+edge_dim,bias=False)
        )
        self.node_num_layer=nn.Sequential(nn.Linear(latent_dim,self.hidden_channels[0]),
                        self.act,
                        nn.Linear(self.hidden_channels[0],self.hidden_channels[1]),
                        self.act,
                        nn.Linear(self.hidden_channels[1],1)
        )
    def forward(self,latent_vec,data):
        number_nodes= self.node_num_layer(latent_vec).view(-1)
        #For inference mode
        nn_int=torch.bincount(data.batch) if self.training else number_nodes.to(torch.int)
        
        #Latent transformation
        projected_vec=F.sigmoid(self.latent_transform(latent_vec))
        
        ls=[]
        for vec,n in zip(projected_vec,nn_int):
            ls.append(self.lstm_cell(vec.unsqueeze(0),n))
        embeddings=torch.cat(ls,dim=0)

        data=utils.create_fc_graph_with_nodes_as_edges_from_matrix(nn_int,embeddings)
        embeddings=self.gnn(data)
        embeddings=self.final_mlp(embeddings)

        return Batch(embeddings,data.edge_index,batch=data.batch),number_nodes


class Sinkhorn_Decoder2(nn.Module):
    def __init__(self,latent_dim,node_dim,edge_dim=5,act="leakyrelu", dropout=0.0):
        super(Sinkhorn_Decoder2, self).__init__()
        self.act = activation_resolver(act, **( {}))
        self.hidden_channels=[64,32,16]
        self.latent_transform=nn.Sequential(nn.Linear(latent_dim,self.hidden_channels[1],bias=False),
                                    self.act,
                                    nn.Linear(self.hidden_channels[1],self.hidden_channels[-1],bias=False)
        )
        self.final_mlp=nn.Sequential(nn.Linear(self.hidden_channels[-1],self.hidden_channels[-2],bias=False),
                            self.act,
                            nn.Linear(self.hidden_channels[-2],self.hidden_channels[-2],bias=False),
                            self.act,
                            nn.Linear(self.hidden_channels[-2],node_dim+edge_dim,bias=False)
        )
        self.node_num_layer=nn.Sequential(nn.Linear(latent_dim,self.hidden_channels[0]),
                                self.act,
                                #nn.BatchNorm1d(self.hidden_channels[0]),
                                nn.Linear(self.hidden_channels[0],self.hidden_channels[1]),
                                self.act,
                                #nn.BatchNorm1d(self.hidden_channels[1]),
                                nn.Linear(self.hidden_channels[1],1)
        )
        
        self.gnn=gnn.GIN(in_channels=self.hidden_channels[-1]+edge_dim,hidden_channels=self.hidden_channels[-2],
                         out_channels=self.hidden_channels[-1],act="leakyrelu",num_layers=5,norm="batchnorm")


    def forward(self,latent_vec,data):
        #Node number
        number_nodes= self.node_num_layer(latent_vec).view(-1)
        nn_int=torch.bincount(data.batch)
        
        #Latent transformation
        embeddings=self.latent_transform(latent_vec)

        data=utils.create_fc_graph_with_nodes_as_edges(nn_int,embeddings)
        embeddings=self.gnn(data)
        embeddings=self.final_mlp(embeddings)


        return Batch(embeddings,data.edge_index,batch=data.batch),number_nodes

class Node_Decoder(nn.Module):
    def __init__(self,latent_dim,num_layers,out_edge,node_dim,act="leakyrelu", dropout=0.0):
        super(Node_Decoder, self).__init__()
        self.act = activation_resolver(act, **( {}))
        self.hidden_channels=[64,32,16]
        self.num_layers=num_layers
        self.edge_dim=out_edge


        self.latent_transform=nn.Sequential(nn.Linear(latent_dim,self.hidden_channels[1],bias=False),
                                    self.act,
                                    nn.Linear(self.hidden_channels[1],self.hidden_channels[2],bias=False)
        )
        self.Convlayers=nn.ModuleList()
        self.Convlayers.append(graph_deconvolution.GNN_Layer3(in_dim=self.hidden_channels[-1],out_dim=self.hidden_channels[-1],
                                                                  edge_dim=out_edge,edge_out_dim=self.hidden_channels[-1]//2))
        for _ in range(num_layers-1):
            self.Convlayers.append(graph_deconvolution.GNN_Layer3(in_dim=self.hidden_channels[-1],out_dim=self.hidden_channels[-1],
                                                                  edge_dim=self.hidden_channels[-1]//2))
        
        self.edge_type_layer=nn.Sequential(
            nn.Linear(self.hidden_channels [-1]//2,self.hidden_channels[-1]),
            self.act,
            nn.Linear(self.hidden_channels[-1],out_edge)
        )
        self.node_feat_layer=nn.Sequential(nn.Linear(self.hidden_channels[-1],self.hidden_channels[-2]),
                        self.act,
                        nn.Linear(self.hidden_channels[-2],node_dim)
                        )
        #Init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=0.01, nonlinearity='leaky_relu')

    def forward(self,latent_vec,data):
        #Latent transformation
        embeddings=self.latent_transform(latent_vec)

        nn_int=torch.bincount(data.batch)
        sum_num_nodes=torch.sum(nn_int)
        edge_index,attr,batch=utils.fully_connected_graph(nn_int,edge_dim=self.edge_dim)

        edge_index,attr=add_self_loops(edge_index,attr,fill_value=torch.tensor([0,1,1,1,1]))
        

        #Message passing
        for i in range(self.num_layers-1):
            #Update node embeddings
            embeddings,attr=self.Convlayers[i](embeddings,edge_index,edge_attr=attr)
            embeddings=F.relu(embeddings)
        embeddings,attr=self.Convlayers[i](embeddings,edge_index,edge_attr=attr)
        #Remove self loops
        edge_index=edge_index[:,:-sum_num_nodes]
        attr=attr[:-sum_num_nodes]

        #attr=utils.compute_pairwise_means_batchwise(embeddings,batch)
        attr=self.edge_type_layer(attr).squeeze(-1)
        #attr=utils.symmetrize_attributes(edge_index,attr)
        

        embeddings=self.node_feat_layer(embeddings)

        return Batch(embeddings,edge_index,attr,batch=batch),None


class Sinkhorn_Decoder(nn.Module):
    def __init__(self,latent_dim,act="leakyrelu", dropout=0.0):
        super(Sinkhorn_Decoder, self).__init__()
        self.act = activation_resolver(act, **( {}))
        self.hidden_channels=[64,32,16]
        self.latent_transform=nn.Sequential(nn.Linear(latent_dim,self.hidden_channels[1],bias=False),
                                    self.act,
                                    nn.Linear(self.hidden_channels[1],self.hidden_channels[2],bias=False)
        )
        self.node_num_layer=nn.Sequential(nn.Linear(latent_dim,self.hidden_channels[0]),
                                self.act,
                                #nn.BatchNorm1d(self.hidden_channels[0]),
                                nn.Linear(self.hidden_channels[0],self.hidden_channels[1]),
                                self.act,
                                #nn.BatchNorm1d(self.hidden_channels[1]),
                                nn.Linear(self.hidden_channels[1],1)
                    )
        self.gnn1=nn.ModuleList()
        self.gnn1.append(graph_deconvolution.GNN_Layer3(in_dim=4+self.hidden_channels[-1],out_dim=self.hidden_channels[-1],edge_dim=5))
        self.gnn1.append(graph_deconvolution.GNN_Layer3(in_dim=self.hidden_channels[-1],out_dim=4,edge_dim=5))
        self.gnn2=nn.ModuleList()
        self.gnn2.append(graph_deconvolution.GNN_Layer3(in_dim=4+self.hidden_channels[-1],out_dim=self.hidden_channels[-1],edge_dim=5))
        self.gnn2.append(graph_deconvolution.GNN_Layer3(in_dim=self.hidden_channels[-1],out_dim=4,edge_dim=5))
        self.gnn3=nn.ModuleList()
        self.gnn3.append(graph_deconvolution.GNN_Layer3(in_dim=4+self.hidden_channels[-1],out_dim=self.hidden_channels[-1],edge_dim=5))
        self.gnn3.append(graph_deconvolution.GNN_Layer3(in_dim=self.hidden_channels[-1],out_dim=4,edge_dim=5))
        self.gnn4=nn.ModuleList()
        self.gnn4.append(graph_deconvolution.GNN_Layer3(in_dim=4+self.hidden_channels[-1],out_dim=self.hidden_channels[-1],edge_dim=5))
        self.gnn4.append(graph_deconvolution.GNN_Layer3(in_dim=self.hidden_channels[-1],out_dim=4,edge_dim=5))
        self.norm=PairNorm()

    def forward(self,latent_vec,data):
        #Node number
        number_nodes= self.node_num_layer(latent_vec)
        number_nodes=number_nodes.view(-1)
        nn_int=torch.bincount(data.batch)
        sum_num_nodes=torch.sum(nn_int)
        
        #Latent transformation
        embeddings=self.latent_transform(latent_vec)
        embeddings=torch.repeat_interleave(embeddings, nn_int, dim=0)

        noise=torch.rand((embeddings.size(0),4))
        embedds=torch.cat([embeddings,noise],dim=1)

        edge_index,attr,batch=utils.fully_connected_graph(nn_int,edge_dim=5)

        edge_index,attr=add_self_loops(edge_index,attr,fill_value=torch.tensor([0,1,0,0,0]))
        edge_index_wo_sl=edge_index[:,:-sum_num_nodes]
        data_ls=[]
        

        for i in range(2):
            #Update node embeddings
            embedds,attr=self.gnn1[i](embedds,edge_index,edge_attr=attr)
            embedds=F.relu(self.norm(embedds,batch))
        fc_data2=Batch(embedds,edge_index_wo_sl,attr[:-sum_num_nodes],batch=batch)
        data_ls.append(fc_data2)
        

        embedds=torch.cat([embedds,embeddings],dim=1)
        #attr2=attr1
        for i in range(2):
            #Update node embeddings
            embedds,attr=self.gnn2[i](embedds,edge_index,edge_attr=attr)
            embedds=F.relu(self.norm(embedds,batch))
        fc_data2=Batch(embedds,edge_index_wo_sl,attr[:-sum_num_nodes],batch=batch)
        data_ls.append(fc_data2)


        embedds=torch.cat([embedds,embeddings],dim=1)
        #attr3=attr2
        for i in range(2):
            #Update node embeddings
            embedds,attr=self.gnn3[i](embedds,edge_index,edge_attr=attr)
            embedds=F.relu(self.norm(embedds,batch))
        fc_data2=Batch(embedds,edge_index_wo_sl,attr[:-sum_num_nodes],batch=batch)
        data_ls.append(fc_data2)


        embedds=torch.cat([embedds,embeddings],dim=1)
        #Do not use loop to use norm only one time 
        embedds,attr=self.gnn4[0](embedds,edge_index,edge_attr=attr)
        embedds=F.relu(self.norm(embedds,batch))
        embedds,attr=self.gnn4[1](embedds,edge_index,edge_attr=attr)

        fc_data2=Batch(embedds,edge_index_wo_sl,attr[:-sum_num_nodes],batch=batch)
        data_ls.append(fc_data2)

        return data_ls,number_nodes


class MPNNDecoder(nn.Module):
    def __init__(self,latent_dim,out_feature,out_edge,num_layers,act="leakyrelu", dropout=0.0):
        super(MPNNDecoder, self).__init__()
        self.dropout = dropout
        self.act = activation_resolver(act, **( {}))
        self.relu=nn.ReLU()
        self.num_layers=num_layers
        self.node_dim=out_feature

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
        self.hidden_channels=[64,32,16]
        self.node_num_layer=nn.Sequential(nn.Linear(latent_dim,self.hidden_channels[0]),
                                        self.act,
                                        #nn.BatchNorm1d(self.hidden_channels[0]),
                                        nn.Linear(self.hidden_channels[0],self.hidden_channels[1]),
                                        self.act,
                                        #nn.BatchNorm1d(self.hidden_channels[1]),
                                        nn.Linear(self.hidden_channels[1],1)
                            )
        self.latent_transform=nn.Sequential(nn.Linear(latent_dim,self.hidden_channels[0],bias=False),
                                    self.act,
                                    #nn.BatchNorm1d(self.hidden_channels[0]),
                                    nn.Linear(self.hidden_channels[0],self.hidden_channels[1],bias=False),
                                    self.act,
                                    #nn.BatchNorm1d(self.hidden_channels[1]),
                                    nn.Linear(self.hidden_channels[1],self.hidden_channels[2],bias=False)
        )
        self.node_feat_layer=nn.Sequential(nn.Linear(self.hidden_channels[-1],self.hidden_channels[-2]),
                              self.act,
                              nn.Linear(32,32),
                              self.act,
                              nn.Linear(32,out_feature)
                              )
        self.edge_type_layer=nn.Sequential(
            nn.Linear(self.hidden_channels [-1]//2,self.hidden_channels[-1]),
            self.act,
            nn.Linear(self.hidden_channels[-1],self.hidden_channels[-2]),
            self.act,
            nn.Linear(self.hidden_channels[-2],out_edge)
        )
        self.Convlayers=nn.ModuleList()


        
        self.Convlayers.append(graph_deconvolution.GNN_Layer3(in_dim=self.hidden_channels[-1]+out_feature,out_dim=self.hidden_channels[-1],
                                                                  edge_dim=out_edge,edge_out_dim=self.hidden_channels[-1]//2))
        for _ in range(num_layers-1):
            self.Convlayers.append(graph_deconvolution.GNN_Layer3(in_dim=self.hidden_channels[-1],out_dim=self.hidden_channels[-1],
                                                                  edge_dim=self.hidden_channels[-1]//2))
        #Init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=0.01, nonlinearity='leaky_relu')

    def forward(self,latent_vec,data):
        number_nodes= self.node_num_layer(latent_vec)
        number_nodes=number_nodes.view(-1)
        nn_int=torch.bincount(data.batch)
        sum_num_nodes=torch.sum(nn_int)
        
        #Latent transformation
        embeddings=self.latent_transform(latent_vec)
        embeddings=torch.repeat_interleave(embeddings, nn_int, dim=0)

        noise=torch.rand((embeddings.size(0),self.node_dim))
        embeddings=torch.cat([embeddings,noise],dim=1)

        edge_index,attr,batch=utils.fully_connected_graph(nn_int,edge_dim=5)

        edge_index,attr=add_self_loops(edge_index,attr,fill_value=1)

        #Message passing
        for i in range(self.num_layers-1):
            #Update node embeddings
            embeddings,attr=self.Convlayers[i](embeddings,edge_index,edge_attr=attr)
            embeddings=F.leaky_relu(embeddings)
            attr=F.leaky_relu(attr)
        embeddings,attr=self.Convlayers[i](embeddings,edge_index,edge_attr=attr)

        #Remove self loops
        attr=attr[:-sum_num_nodes]
        edge_index=edge_index[:,:-sum_num_nodes]

        #Graph readout
        node_features=self.node_feat_layer(embeddings)
        #attr=utils.symmetrize_attributes(edge_index,attr)
        attr=self.edge_type_layer(attr).squeeze(-1)
       
        

        return Batch(x=node_features,edge_index=edge_index,edge_attr=attr,batch=batch),number_nodes
    
    def _add_gaussian_noise(self,embeddings, mean=0, std=0.5):
        noise=torch.randn_like(embeddings)
        noise = noise* std + mean
        return embeddings + noise


class MLPDecoder(nn.Module):
    def __init__(self,latent_dim,node_dim=2,act="relu" ,hidden_dim=16, dropout=0.1,warm_up=False):
        super(MLPDecoder, self).__init__()
        
        self.dropout = dropout
        self.act = activation_resolver(act, **( {}))
        self.warm_up=warm_up
        self.node_dim=node_dim
        

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
        self.hidden_channels=[64,32,16]
        self.node_num_layer=nn.Sequential(nn.Linear(latent_dim,self.hidden_channels[0]),
                                        self.act,
                                        #nn.BatchNorm1d(self.hidden_channels[0]),
                                        nn.Linear(self.hidden_channels[0],self.hidden_channels[1]),
                                        self.act,
                                        #nn.BatchNorm1d(self.hidden_channels[1]),
                                        nn.Linear(self.hidden_channels[1],1)
                            )
        self.latent_transform=nn.Sequential(nn.Linear(latent_dim,self.hidden_channels[0],bias=False),
                                    self.act,
                                    #nn.BatchNorm1d(self.hidden_channels[0]),
                                    nn.Linear(self.hidden_channels[0],self.hidden_channels[1],bias=False),
                                    self.act,
                                    #nn.BatchNorm1d(self.hidden_channels[1]),
                                    nn.Linear(self.hidden_channels[1],self.hidden_channels[2],bias=False)
        )
        self.node_feat_layer=nn.Sequential(nn.Linear(hidden_dim,32),
                              self.act,
                              nn.Linear(32,32),
                              self.act,
                              nn.Linear(32,node_dim)
                              )     
        self.edge_type_layer=nn.Sequential(
            nn.Linear(2*hidden_dim,64),
            self.act,
            nn.Linear(64,64),
            self.act,
            nn.Linear(64,5)
        )
        self.gnn=gnn.GIN(in_channels=self.hidden_channels[2]+node_dim,hidden_channels=hidden_dim,out_channels=hidden_dim,num_layers=4,norm='layer',act="leakyrelu")
    def forward(self,latent_vec,data):
        #Graph Readin
        #Node number
        number_nodes= self.node_num_layer(latent_vec)
        number_nodes=number_nodes.view(-1)
        nn_int=torch.bincount(data.batch)
        sum_num_nodes=torch.sum(nn_int)

        #Latent transformation
        embeddings=self.latent_transform(latent_vec)
        embeddings=torch.repeat_interleave(embeddings, nn_int, dim=0)
        #Adding noise
        noise=torch.rand((embeddings.size(0),self.node_dim))
        embeddings=torch.cat([embeddings,noise],dim=1)
        

        edge_index,_,batch=utils.fully_connected_graph(nn_int,edge_dim=1)
        edge_index,_=add_self_loops(edge_index)

        output=self.gnn(Batch(embeddings,edge_index,batch=batch))
        #Remove self loops
        edge_index=edge_index[:,:-sum_num_nodes]
 
        #Feature and node prediction
        node_features=self.node_feat_layer(output)
        edges=utils.compute_pairwise_means_batchwise(output,batch)
        attr=utils.symmetrize_attributes(edge_index,edges)
        attr=self.edge_type_layer(attr).squeeze(-1)
        #attr=F.softmax(attr,dim=-1)
        #Warm up
        if self.warm_up:
            warm_up_first=attr[:,0].unsqueeze(1)
            sum_last_four = attr[:, 1:].sum(dim=1, keepdim=True)
            attr = torch.cat([warm_up_first, sum_last_four], dim=1)
        
        return Batch(x=node_features,edge_index=edge_index,edge_attr=attr,batch=batch),number_nodes
    


class GraphReadInLSTM(nn.Module):
    def __init__(self,input_dim,rnn_hidden_dim):
        
        super(GraphReadInLSTM, self).__init__()

        # LSTM cell for processing each node's initial state
        self.lstm_cell = nn.LSTMCell(input_size=input_dim, hidden_size=rnn_hidden_dim)

    def forward(self, projected_representation,num_nodes):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Initialize hidden and cell states for each node using an LSTM cell
        node_states = []
        hidden_state, cell_state = torch.zeros(1, self.lstm_cell.hidden_size).to(device), torch.zeros(1, self.lstm_cell.hidden_size).to(device)
        for _ in range(num_nodes):
            hidden_state, cell_state = self.lstm_cell(projected_representation, (hidden_state, cell_state))
            node_states.append(hidden_state)

        # Initialize edges with zero vectors
        node_states=torch.cat(node_states,dim=0)

        return node_states
    
class GraphReadInGRU(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,max_nodes):
        
        super(GraphReadInGRU, self).__init__()

        # LSTM cell for processing each node's initial state
        self.gru_cell = nn.GRUCell(input_size=input_dim, hidden_size=hidden_dim)
        self.lin=nn.Linear(hidden_dim,output_dim)
        self.max_nodes=max_nodes
        self.init_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, latent_vec,node_sum):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Initialize hidden and cell states for each node using an LSTM cell
        node_states = []    
        hidden_state=torch.zeros(latent_vec.size(0), self.gru_cell.hidden_size).to(device)
        hidden_state = self.init_mlp(latent_vec).to(device)
        for _ in range(self.max_nodes):
            hidden_state= self.gru_cell(latent_vec, hidden_state)
            features=self.lin(hidden_state)
            node_states.append(features)
            

        # Initialize edges with zero vectors
        node_states=torch.cat(node_states,dim=0)

        mask=torch.zeros(node_states.size(0),dtype=bool)
        ind=0
        for i in node_sum:
            mask[torch.arange(ind,ind+i,dtype=torch.int)]=True
            ind+=self.max_nodes

        return node_states[mask]
    
class MultiDenseLayer(nn.Module):
    def __init__(self, aux_unit, linear_units, activation=None, dropout_rate=0.):
        super(MultiDenseLayer, self).__init__()
        layers = []
        for c0, c1 in zip([aux_unit] + linear_units[:-1], linear_units):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Dropout(dropout_rate))
            if activation is not None:
                layers.append(activation)
        self.linear_layer = nn.Sequential(*layers)

    def forward(self, inputs):
        h = self.linear_layer(inputs)
        return h
    
class Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dims, z_dim,output_dim, vertexes, dropout_rate):
        super(Generator, self).__init__()
        self.multi_dense_layer = MultiDenseLayer(z_dim, conv_dims, torch.nn.Tanh())

        self.vertexes = vertexes
        self.num=output_dim
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * self.num)
        self.dropoout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.multi_dense_layer(x)

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropoout(nodes_logits.view(-1, self.vertexes, self.num))

        return nodes_logits
    
class Generator2(nn.Module):
    """Generator network."""

    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout_rate):
        super(Generator2, self).__init__()
        self.multi_dense_layer = MultiDenseLayer(z_dim, conv_dims, torch.nn.Tanh())

        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes

        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.multi_dense_layer(x)
        edges_logits = self.edges_layer(output).view(-1, self.edges, self.vertexes, self.vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2
        edges_logits = self.dropout(edges_logits.permute(0, 2, 3, 1))

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropout(nodes_logits.view(-1, self.vertexes, self.nodes))

        return nodes_logits,edges_logits
    

class Generator_Decoder(nn.Module):
    def __init__(self,latent_dim,node_dim,edge_dim,num_layers,act="leakyrelu", dropout=0.0):
        super(Generator_Decoder, self).__init__()
        self.act = activation_resolver(act, **( {}))
        self.hidden_channels=[64,32,16]
        self.edge_dim=edge_dim
        self.num_layers=num_layers
        #self.latent_transform=nn.Sequential(nn.Linear(latent_dim,hidden_dim,bias=False))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_nodes=9
        hidden_node_dim=13
        self.mlp_init= Generator2(z_dim=latent_dim,edges=edge_dim,nodes=hidden_node_dim,vertexes=self.max_nodes,conv_dims=[128, 256, 512],dropout_rate=dropout)

        self.Convlayers=nn.ModuleList()
        self.Convlayers.append(graph_deconvolution.GNN_Layer4(node_dim=hidden_node_dim,out_dim=self.hidden_channels[-2],
                                                                  edge_dim=edge_dim,heads=4,concat=False))
        for _ in range(num_layers-1):
            self.Convlayers.append(graph_deconvolution.GNN_Layer4(node_dim=self.hidden_channels[-2],out_dim=self.hidden_channels[-2],
                                                                  edge_dim=self.hidden_channels[-2],heads=4,concat=False))
        
        self.feature_mlp=nn.Sequential(nn.Linear(self.hidden_channels[-2],node_dim,bias=False))
        self.edge_mlp=nn.Sequential(nn.Linear(self.hidden_channels[-2],edge_dim,bias=False))


        self.norms_feat=nn.ModuleList()
        for _ in range(num_layers-1):
            self.norms_feat.append(LayerNorm(self.hidden_channels[-2]))
        

        self.norms_attr=nn.ModuleList()
        for _ in range(num_layers-1):
            self.norms_attr.append(LayerNorm(self.hidden_channels[-2]))
        

    def forward(self,latent_vec):

        nodes,edges=self.mlp_init(latent_vec)
        data=utils.dense_batch_to_sparse(nodes,edges,add_self_loops=True).to(self.device)
        #Message passing
        embeddings=data.x
        attr=data.edge_attr
        for i in range(self.num_layers-1):
            #Update node embeddings
            embeddings,attr=self.Convlayers[i](embeddings,data.edge_index,edge_attr=attr,batch=data.batch)
            embeddings=self.norms_feat[i](embeddings,data.batch)
            #attr=self.norms_attr[i](attr,data.batch)
            embeddings=F.leaky_relu(embeddings)
            attr=F.leaky_relu(attr)
            #embeddings=F.relu(embeddings)
        embeddings,attr=self.Convlayers[i](embeddings,data.edge_index,edge_attr=attr,batch=data.batch)
        embeddings=self.feature_mlp(embeddings)
        attr=self.edge_mlp(attr)
        edge_index,attr=remove_self_loops(data.edge_index,attr)

        if not self.training:
            embeddings=F.softmax(embeddings,dim=1)

            # nodes_idx=torch.argmax(embeddings,dim=1)!=0
            # nodes_idx=torch.nonzero(nodes_idx).view(-1)

            nodes_idx=torch.nonzero(embeddings[:,0]<0.5).view(-1)
            #Remove non-type nodes
            edge_index,attr=subgraph(nodes_idx,edge_index,attr,relabel_nodes=True)
            embeddings=embeddings[nodes_idx,1:]
            #Remove non-type edges and remove isolated nodes
            edge_index,attr=to_undirected(edge_index,attr,reduce="mean")

            #Remove non-type edges
            edge_mask=torch.argmax(attr,dim=1)!=0
            edge_index=edge_index[:,edge_mask]
            attr=attr[edge_mask]

            edges_idx=torch.unique(edge_index[0])
            edge_index,attr=subgraph(edges_idx,edge_index,attr,relabel_nodes=True)
            embeddings=embeddings[edges_idx]
            
            data=Data(embeddings,edge_index=edge_index,edge_attr=attr[:,1:],batch=torch.full((embeddings.size(0),), 0))


            return data


        return Batch(embeddings,edge_index,edge_attr=attr,batch=data.batch)


