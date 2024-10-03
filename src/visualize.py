import networkx as nx
from rdkit import Chem
import torch
from torch_geometric.utils import to_networkx,sort_edge_index
import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 22


def draw_graph(data, ax=None, pos=None, shiftx=0, seed=0):
    node_size = 80

    G = to_networkx(data, to_undirected=True)

    if pos is None:
        pos = nx.spring_layout(G, scale=1., seed=seed)

    if shiftx != 0:
        for k, v in pos.items():
            v[0] = v[0] + shiftx

    edge_index,attr=sort_edge_index(data.edge_index,data.edge_attr,sort_by_row=True)
    if data.edge_attr.size(-1)==5:
        edges_line_style_dict = {1: 'solid', 2: 'dashed', 3: 'dotted', 4: 'dashdot'}
    else:
        edges_line_style_dict = {0: 'solid', 1: 'dashed', 2: 'dotted', 3: 'dashdot'}
        
    edges_style_part = [edges_line_style_dict[i.item()] for i in torch.argmax(attr, dim=1)]

    for edge, edge_style in zip(G.edges, edges_style_part):
        src,tgt=edge
        mask=(edge_index[0,:]==src) & (edge_index[1,:]==tgt)
        temp_attr=attr[mask]
        edge_style=edges_line_style_dict[torch.argmax(temp_attr,dim=1).item()]
        nx.draw_networkx_edges(G, pos, edgelist=[edge], style=edge_style, ax=ax)

    color_map = {0: "grey", 1: "blue", 2: "red", 3: "green"}
    nodes_color_part = [color_map[i.item()] for i in torch.argmax(data.x, dim=1)]

    nodes_ls = torch.argmax(data.x, dim=1)
    node_dict = {0: "C", 1: "N", 2: "O", 3: "F"}
    for node, node_color in enumerate(nodes_color_part):
        temp = node_dict[nodes_ls[node].item()]
        nx.draw_networkx_nodes(G, pos, nodelist=[node],
                               node_size=node_size, alpha=1, label=temp,
                               node_color=node_color, ax=ax)

    labels = {node: node_dict[nodes_ls[node].item()] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

    return pos


def graph_matching(data1, data2, T,
                           ax1=None, pos1=None, pos2=None, shiftx=4, switchx=False,
                           seed_G1=0, seed_G2=0):

    pos1 = draw_graph(data1, ax=ax1, pos=pos1, shiftx=0, seed=seed_G1)
    pos2 = draw_graph(data2, ax=ax1, pos=pos2, shiftx=shiftx, seed=seed_G2)
    T=T.detach().numpy()
    max_indices = np.argmax(T, axis=1)
    if ax1 is not None:
        for k1, v1 in pos1.items():
            max_Tk1 = np.max(T[k1, :])
            k2 = max_indices[k1]  # The column with the maximum value for this row
            v2 = pos2[k2]
            ax1.plot([v1[0], v2[0]], [v1[1], v2[1]], '-', lw=0.7, alpha=1, color="C3")

            for k2, v2 in pos2.items():
                if T[k1, k2] > 0:
                    ax1.plot([pos1[k1][0], pos2[k2][0]],
                            [pos1[k1][1], pos2[k2][1]],
                            '-', lw=0.7, alpha=min(T[k1, k2] / max_Tk1 + 0.1, 1.), color="C3")
    else:

        for k1, v1 in pos1.items():
            max_Tk1 = np.max(T[k1, :])   
            k2 = max_indices[k1]  # The column with the maximum value for this row
            v2 = pos2[k2]
            # plt.plot([v1[0], v2[0]],
            # [v1[1], v2[1]],
            # '-', lw=0.7, alpha=1,
            # color="C3")
            
            for k2, v2 in pos2.items():
                    if (T[k1, k2] > 0):
                
                        plt.plot([pos1[k1][0], pos2[k2][0]],
                                [pos1[k1][1], pos2[k2][1]],
                                '-', lw=0.7, alpha=min(T[k1, k2] / 1, 1.),
                                color="C3")