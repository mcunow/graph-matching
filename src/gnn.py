import copy
import inspect
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear, ModuleList
from tqdm import tqdm
from torch_geometric.nn.models import MLP
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops 
from torch_geometric.loader import  NeighborLoader
from torch_geometric.nn.conv import (
    GATConv,
    GATv2Conv,
    PNAConv,
    GINEConv,
    GINConv,
    GCNConv,
    MessagePassing,
    RGCNConv,
    GPSConv

)

from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj, OptTensor, SparseTensor



class BasicGNN(torch.nn.Module):
    r"""An abstract class for implementing basic GNN models.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.conv.MessagePassing` layers.
    """
 

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        edge_dim=None,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.dropout = torch.nn.Dropout(p=dropout)
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.jk_mode = jk
        self.act_first = act_first
        self.norm = norm if isinstance(norm, str) else None
        self.norm_kwargs = norm_kwargs

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels
        if  isinstance(self.hidden_channels,int):
            self.hidden_channels=[hidden_channels for i in range(self.num_layers)]


        self.convs = ModuleList()
        self.norms = ModuleList()
        if self.norm is None:
            norm_layer = torch.nn.Identity()
        if num_layers > 1:
            self.convs.append(
                self.init_conv(in_channels, self.hidden_channels[0],edge_dim=edge_dim,plain_last=False, **kwargs))
            in_channels = self.hidden_channels[0]
            if self.norm is None:
                norm_layer = torch.nn.Identity()
                self.norms.append(norm_layer)
            else:
                self.norms.append(normalization_resolver(
                    norm,self.hidden_channels[0],**(norm_kwargs or {}),))
        for i in range(1,num_layers - 1):
            self.convs.append(
                self.init_conv(in_channels, self.hidden_channels[i],edge_dim=edge_dim,plain_last=False, **kwargs))
            if self.norm is None:
                norm_layer = torch.nn.Identity()
                self.norms.append(norm_layer)
            else:
                self.norms.append(normalization_resolver(
                    norm,self.hidden_channels[i],**(norm_kwargs or {}),))
            if isinstance(in_channels, (tuple, list)):
                in_channels = (hidden_channels, hidden_channels)
            else:
                in_channels = self.hidden_channels[i]
        if out_channels is not None and jk is None:
            self._is_conv_to_out = True
            self.convs.append(
                self.init_conv(in_channels, out_channels,edge_dim=edge_dim,plain_last=True, **kwargs))
        else:
            self.convs.append(
                self.init_conv(in_channels, hidden_channels,edge_dim=edge_dim,plain_last=True, **kwargs))
        self.norms.append(torch.nn.Identity())




        self.supports_norm_batch = False

        if jk is not None and jk != 'last':
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        if jk is not None:
            if jk == 'cat':
                in_channels = num_layers * hidden_channels
            else:
                in_channels = hidden_channels
            self.lin = Linear(in_channels, self.out_channels)

        # We define `trim_to_layer` functionality as a module such that we can
        # still use `to_hetero` on-top.
        

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        raise NotImplementedError

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()
        if hasattr(self, 'jk'):
            self.jk.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()


    def forward( 
        self,
        data,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
    ):
        r"""
        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights (if
                supported by the underlying GNN layer). (default: :obj:`None`)
            edge_attr (torch.Tensor, optional): The edge features (if supported
                by the underlying GNN layer). (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
        """
        x=data.x
        
        edge_index=data.edge_index
        if data.edge_attr is None:
            edge_attr=data.edge_weight
            edge_weight=data.edge_weight
        else:
            edge_attr=data.edge_attr
            edge_weight=data.edge_attr
        if data.batch is not None:
            batch=data.batch
        edge_index,edge_attr=add_self_loops(edge_index,edge_attr)
            
            
        xs: List[Tensor] = []
        assert len(self.convs) == len(self.norms)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):

            # Tracing the module is not allowed with *args and **kwargs :(
            # As such, we rely on a static solution to pass optional edge
            # weights and edge attributes to the module.
            if self.supports_edge_weight and self.supports_edge_attr:
                x = conv(x, edge_index, edge_weight=edge_weight,
                         edge_attr=edge_attr)
            elif self.supports_edge_weight:
                x = conv(x, edge_index, edge_weight)
            elif self.supports_edge_attr:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)



            if i < self.num_layers - 1 or self.jk_mode is not None:
                if self.act is not None and self.act_first:
                    x = self.act(x)
                if self.norm in ["instance","pairnorm","layer","graph"]:
                    x = norm(x, batch)
                else:
                    x = norm(x)
                if self.act is not None and not self.act_first:
                    x = self.act(x)
                x = self.dropout(x)
                if hasattr(self, 'jk'):
                    xs.append(x)

        return x

    @torch.no_grad()
    def inference_per_layer(
        self,
        layer: int,
        x: Tensor,
        edge_index: Adj,
        batch_size: int,
    ) -> Tensor:

        x = self.convs[layer](x, edge_index)[:batch_size]

        if layer == self.num_layers - 1 and self.jk_mode is None:
            return x

        if self.act is not None and self.act_first:
            x = self.act(x)
        if self.norms is not None:
            x = self.norms[layer](x)
        if self.act is not None and not self.act_first:
            x = self.act(x)
        if layer == self.num_layers - 1 and hasattr(self, 'lin'):
            x = self.lin(x)

        return x

    @torch.no_grad()
    def inference(
        self,
        loader: NeighborLoader,
        device: Optional[Union[str, torch.device]] = None,
        embedding_device: Union[str, torch.device] = 'cpu',
        progress_bar: bool = False,
        cache: bool = False,
    ) -> Tensor:
        r"""Performs layer-wise inference on large-graphs using a
        :class:`~torch_geometric.loader.NeighborLoader`, where
        :class:`~torch_geometric.loader.NeighborLoader` should sample the
        full neighborhood for only one layer.
        This is an efficient way to compute the output embeddings for all
        nodes in the graph.
        Only applicable in case :obj:`jk=None` or `jk='last'`.

        Args:
            loader (torch_geometric.loader.NeighborLoader): A neighbor loader
                object that generates full 1-hop subgraphs, *i.e.*,
                :obj:`loader.num_neighbors = [-1]`.
            device (torch.device, optional): The device to run the GNN on.
                (default: :obj:`None`)
            embedding_device (torch.device, optional): The device to store
                intermediate embeddings on. If intermediate embeddings fit on
                GPU, this option helps to avoid unnecessary device transfers.
                (default: :obj:`"cpu"`)
            progress_bar (bool, optional): If set to :obj:`True`, will print a
                progress bar during computation. (default: :obj:`False`)
            cache (bool, optional): If set to :obj:`True`, caches intermediate
                sampler outputs for usage in later epochs.
                This will avoid repeated sampling to accelerate inference.
                (default: :obj:`False`)
        """
        assert self.jk_mode is None or self.jk_mode == 'last'
        assert isinstance(loader, NeighborLoader)
        assert len(loader.dataset) == loader.data.num_nodes
        assert len(loader.node_sampler.num_neighbors) == 1
        assert not self.training
        # assert not loader.shuffle  # TODO (matthias) does not work :(
        if progress_bar:
            pbar = tqdm(total=len(self.convs) * len(loader))
            pbar.set_description('Inference')

        x_all = loader.data.x.to(embedding_device)

        if cache:

            # Only cache necessary attributes:
            def transform(data: Data) -> Data:
                kwargs = dict(n_id=data.n_id, batch_size=data.batch_size)
                if hasattr(data, 'adj_t'):
                    kwargs['adj_t'] = data.adj_t
                else:
                    kwargs['edge_index'] = data.edge_index

                return Data.from_dict(kwargs)

            loader = CachedLoader(loader, device=device, transform=transform)

        for i in range(self.num_layers):
            xs: List[Tensor] = []
            for batch in loader:
                x = x_all[batch.n_id].to(device)
                batch_size = batch.batch_size
                if hasattr(batch, 'adj_t'):
                    edge_index = batch.adj_t.to(device)
                else:
                    edge_index = batch.edge_index.to(device)

                x = self.inference_per_layer(i, x, edge_index, batch_size)
                xs.append(x.to(embedding_device))

                if progress_bar:
                    pbar.update(1)

            x_all = torch.cat(xs, dim=0)

        if progress_bar:
            pbar.close()

        return x_all
    
class GAT(BasicGNN):
    r"""The Graph Neural Network from `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ or `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ papers, using the
    :class:`~torch_geometric.nn.GATConv` or
    :class:`~torch_geometric.nn.GATv2Conv` operator for message passing,
    respectively.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        v2 (bool, optional): If set to :obj:`True`, will make use of
            :class:`~torch_geometric.nn.conv.GATv2Conv` rather than
            :class:`~torch_geometric.nn.conv.GATConv`. (default: :obj:`False`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
   

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:

        v2 = kwargs.pop('v2', False)
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)

        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):
        if getattr(self, '_is_conv_to_out', False):
            concat = False

        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'GATConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        Conv = GATConv if not v2 else GATv2Conv
        return Conv(in_channels, out_channels, heads=heads, concat=concat,
                    dropout=self.dropout.p, **kwargs)

class GINE(BasicGNN):
    supports_edge_weight=False
    supports_edge_attr=True
    has_mlp=True
    
    
    def init_conv(self, in_channels: int, out_channels: int,plain_last:bool,edge_dim=5,
                  **kwargs) -> MessagePassing:
        
        
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            plain_last=plain_last,
            act=self.act,   
            norm=self.norm
        )


        return GINEConv(mlp,eps=0.1,train_eps=True,edge_dim=edge_dim,**kwargs)

class GIN(BasicGNN):
    supports_edge_weight = False
    supports_edge_attr = False  
    has_mlp=True


    def init_conv(self, in_channels: int, out_channels: int,plain_last:bool,edge_dim=None,
                  **kwargs) -> MessagePassing:

        mlp = MLP(
            [in_channels, out_channels, out_channels],
            plain_last=plain_last,
            act=self.act,   
            norm=self.norm
        )
        return GINConv(mlp,eps=0.1,train_eps=True, **kwargs)
    

class GCN(BasicGNN):
    supports_edge_weight = True
    supports_edge_attr = False  
    has_mlp=False
    def init_conv(self, in_channels: int, out_channels: int,
            **kwargs) -> MessagePassing:

        return GCNConv(in_channels,out_channels)
    
class PNA(BasicGNN):

    supports_edge_weight= False
    supports_edge_attr= True
    has_mlp=False

    
    def init_conv(self, in_channels: int, out_channels: int,edge_dim,
                  **kwargs) -> MessagePassing:
        return PNAConv(in_channels, out_channels,aggregators=['mean', 'std', 'min', 'max'],
                        scalers=['identity', 'amplification', 'attenuation'],
                        edge_dim=1,towers=4,train_norm=True,deg=torch.tensor([0, 1, 2, 3]), **kwargs)
    

class RGCN(BasicGNN):
    supports_edge_weight= True
    supports_edge_attr= False
    has_mlp=False
    def init_conv(self, in_channels: int, out_channels: int,edge_dim,
        **kwargs) -> MessagePassing:

        return RGCNConv(in_channels,out_channels,num_relations=edge_dim,aggr="mean")

class GPS(BasicGNN):
    supports_edge_weight= False
    supports_edge_attr= False
    def init_conv(self, in_channels: int, out_channels: int,edge_dim,
        **kwargs) -> MessagePassing:
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            plain_last=False,
            act=self.act,   
            norm=self.norm
        )
        gin=GINConv(mlp,eps=0.1,train_eps=True)
        if in_channels%4!=0:
            heads=2
        else:
            heads=4
        return GPSConv(in_channels,conv=gin,heads=heads)
    

class PartialGNN(nn.Module):
    def __init__(self, original_model,num_layers,norm=True):
        super(PartialGNN, self).__init__()
        conv_ls=[]
        norm_ls=[]
        for i in range(num_layers):
            conv_ls.append(original_model.convs[i])
            if not norm:
                norm_ls.append(nn.Identity())
            else:
                norm_ls.append(original_model.norms[i])
        self.convs=nn.ModuleList(conv_ls)
        self.norms=nn.ModuleList(norm_ls)

        #self.convs = nn.ModuleList([original_model.convs[0], original_model.convs[1]])
        #self.norms = nn.ModuleList([original_model.norms[0], original_model.norms[1]])
        self.dropout = original_model.dropout
        self.act = original_model.act

    def forward(self, data):
        x=data.x
        edge_index=data.edge_index
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = self.act(x)
            #x = self.dropout(x)
        return x