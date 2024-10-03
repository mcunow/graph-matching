from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.nn import Set2Set
from torch_geometric.nn.inits import reset
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
from torch_geometric.data import Data
import utils
import ot_lib
import copy
import gnn


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
        #self.aux_gnn=auxiliary_gnn
        #self.s_loss=ot_lib.Sinkhorn_Loss(self.aux_gnn,sinkhorn_entropy=sinkhorn_entropy,sinkhorn_max_it=sinkhorn_max_it,dist="uniform",gumbel=gumbel,gnn_train=gnn_train)
        if embedding=='graph':
            self.linear_mu = torch.nn.Linear(2*self.encoder.out_channels, latent_dims)
            self.linear_sigma = torch.nn.Linear(2*self.encoder.out_channels, latent_dims)
            self.aggregation=Set2Set(self.encoder.out_channels,processing_steps=6)
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
    def kl_loss(mu ,logvar):
        #KL-Divergence is averaged for the whole batch
        kl_loss=torch.exp(logvar) - logvar + torch.pow(mu,2) - 1
        kl_loss = 0.5 * torch.sum(kl_loss,dim=1)
        kl_loss=torch.sum(kl_loss)
        return kl_loss

        
    
    def reparametrize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if self.training:
            return mu + torch.randn_like(logvar) * torch.exp(0.5*logvar)     
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
            noise=torch.randn((1,self.latent_dims))
            data=self.decoder(noise)
            try:
                data=Data(F.one_hot(torch.argmax(data.x,dim=1),num_classes=4).to(torch.float),data.edge_index,
                        edge_attr=F.one_hot(torch.argmax(data.edge_attr,dim=1),num_classes=4),batch=data.batch)
                data_ls.append(data)
            except:
                data_ls.append(None)
        return data_ls


    def forward(self,*args, **kwargs):
        z,mu,sigma,x=self.encode(*args, **kwargs)
        data,pred_num=self.decoder(z,*args)
        return data,mu,sigma,x,pred_num

