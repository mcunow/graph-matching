from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.nn import Set2Set, AttentionalAggregation
from torch_geometric.nn.inits import reset
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data


EPS = 1e-15
MAX_LOGSTD = 10





class GAE(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.

    Args:
        encoder (torch.nn.Module): The encoder module.
        decoder (torch.nn.Module, optional): The decoder module. If set to
            :obj:`None`, will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder: Module, decoder: Optional[Module] = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        #GAE.reset_parameters(self)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.decoder)

    def forward(self, *args, **kwargs) -> Tensor:  # pragma: no cover
        r"""Alias for :meth:`encode`."""
        return self.encode(*args, **kwargs)

    def encode(self, *args, **kwargs) -> Tensor:
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs) -> Tensor:
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)




    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to train against.
            neg_edge_index (torch.Tensor, optional): The negative edges to
                train against. If not given, uses negative sampling to
                calculate negative edges. (default: :obj:`None`)
        """
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(self, z: Tensor, pos_edge_index: Tensor,
             neg_edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to evaluate
                against.
            neg_edge_index (torch.Tensor): The negative edges to evaluate
                against.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)
    
    


class VGAE(GAE):
    r"""The Variational Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper.

    Args:
        encoder (torch.nn.Module): The encoder module to compute :math:`\mu`
            and :math:`\log\sigma^2`.
        decoder (torch.nn.Module, optional): The decoder module. If set to
            :obj:`None`, will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self,encoder: Module, decoder: Module,latent_dims,eval,embedding="graph",sinkhorn_enc=None):
        super().__init__(encoder, decoder)
        self.embedding=embedding
        self.latent_dims=latent_dims
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        #self.aux_gnn=auxiliary_gnn
        #self.s_loss=ot_lib.Sinkhorn_Loss(self.aux_gnn,sinkhorn_entropy=sinkhorn_entropy,sinkhorn_max_it=sinkhorn_max_it,dist="uniform",gumbel=gumbel,gnn_train=gnn_train)
        if embedding=='graph':
            self.linear_mu = torch.nn.Linear(self.encoder.out_channels*2, latent_dims).to(self.device)
            self.linear_sigma = torch.nn.Linear(self.encoder.out_channels*2, latent_dims).to(self.device)
            #self.aggregation=Set2Set(self.encoder.out_channels,processing_steps=6).to(self.device)
            
            mlp_gate=torch.nn.Sequential(nn.Linear(self.encoder.out_channels,128),
                                    nn.LeakyReLU(),
                                    nn.Linear(128,128),
                                    nn.LeakyReLU(),
                                    nn.Linear(128,1)).to(self.device)
            mlp_end=nn.Sequential(nn.Linear(self.encoder.out_channels,128),
                                    nn.LeakyReLU(),
                                    nn.Linear(128,128),
                                    nn.LeakyReLU(),
                                    nn.Linear(128,self.encoder.out_channels)
            ).to(self.device)
            self.linear_mu = torch.nn.Linear(self.encoder.out_channels, latent_dims).to(self.device)
            self.linear_sigma = torch.nn.Linear(self.encoder.out_channels, latent_dims).to(self.device)
            self.aggregation=AttentionalAggregation(mlp_gate,mlp_end).to(self.device)
            
        elif embedding=='node':
            self.linear_mu=torch.nn.Linear(self.encoder.out_channels,latent_dims)
            self.linear_sigma=torch.nn.Linear(self.encoder.out_channels,latent_dims)


        self.eval_mode=eval
        """
        if sinkhorn_enc is None:
            self.sinkhorn_enc=gnn.GINE(in_channels=encoder.in_channels,hidden_channels=encoder.hidden_channels,
                                       out_channels=encoder.out_channels,norm=encoder.norm,num_layers=encoder.num_layers,
                                        act="leakyrelu",edge_dim=4,dropout=0.,add_self_loops=True,affine=True)
            self.sinkhorn_enc.load_state_dict(self.encoder.state_dict())
            if self.eval_mode:
                self.sinkhorn_enc.eval()
            for param in self.sinkhorn_enc.parameters():
                param.requires_grad = False
        
        self.sinkhorn_solver=ot_lib.Sinkhorn_Loss(sinkhorn_enc,sinkhorn_entropy=0.1,
                                                    sinkhorn_max_it=100,opt_method="sinkhorn_stabilized")
        
        """
            
    def set_encoder(self):
        self.sinkhorn_enc.load_state_dict(self.encoder.state_dict())
        if self.eval_mode:
            self.sinkhorn_enc.eval()
        self.sinkhorn_solver.set_encoder(self.sinkhorn_enc)

    @staticmethod
    def kl_loss(mu ,logstd):
        #KL-Divergence is averaged for the whole batch
        """
        logvar_temp=torch.exp(logvar)
        kl_loss=logvar_temp - logvar + torch.pow(mu,2) - 1
        kl_loss = 0.5 * torch.sum(kl_loss,dim=1)
        kl_loss=torch.mean(kl_loss)
        """
        kl_loss=1 + 2 * logstd - mu**2 - logstd.exp()**2
        kl_loss=torch.sum(kl_loss, dim=1)
        kl_loss=-0.5 * torch.mean(kl_loss)
        return kl_loss

        
    
    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd) 
        else:
            return mu

    def encode(self, data, **kwargs) -> Tensor:
        """"""
        node_embedds = self.encoder(data, **kwargs)
        #Aggregate to global representation
        if self.embedding=='graph':
            x=self.aggregation(node_embedds,data.batch)
        #Compute mu and sigma
        mu=self.linear_mu(x)
        logvar=self.linear_sigma(x)
        #Reparametrize
        logvar = logvar.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(mu ,logvar)
        return z,mu,logvar,node_embedds
    

    def sample(self,num_samples:int):
        data_ls=[]  
        for _ in range(num_samples):
            noise=torch.randn((1,self.latent_dims),device=self.device)
            
            data,_=self.decoder(noise)
            if data is not None:
                data=self.reduce_graph(data)
                data=self.remove_edges(data)
            data_ls.append(data)

        return data_ls
    
    def remove_edges(self,data):
        if data.edge_attr.size(0)!=0:
            mask=data.edge_attr[:,0]<0.5
            #mask=torch.argmax(data.edge_attr,dim=1)
            #mask=mask!=0
            data.edge_attr=data.edge_attr[mask]
            data.edge_index=data.edge_index[:,mask]
        return Data(F.one_hot(torch.argmax(data.x,dim=1),num_classes=4).to(torch.float),data.edge_index,
                        edge_attr=F.one_hot(torch.argmax(data.edge_attr[:,1:],dim=1),num_classes=4),batch=data.batch)
    
    def reduce_graph(self,data):
        num_nodes=int(0.5*((8*data.x.size(0)+1)**(1/2)-1))
        edge_index = []
        edge_attr=[]
        k=num_nodes
        for i in range(num_nodes):
            for j in range(i,num_nodes):
                if i!=j:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    edge_attr.append(data.x[k,:])
                    edge_attr.append(data.x[k,:])
                    k+=1
        edge_index = torch.tensor(edge_index, dtype=torch.long,device=self.device).t().contiguous()
        edge_attr=torch.stack(edge_attr) if len(edge_attr)>0 else torch.Tensor(size=(0,5))

        return Data(x=data.x[:num_nodes,1:],edge_index=edge_index,edge_attr=edge_attr)


    def forward(self,*args, **kwargs):
        z,mu,sigma,x=self.encode(*args, **kwargs)
        data,pred_num=self.decoder(z,*args)
        return data,mu,sigma,x,pred_num

