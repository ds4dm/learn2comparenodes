#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 20:44:58 2021

@author: abdel
from https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation.ipynb with some modifications
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric

class GNNPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_size = emb_size = 16
        self.k = 16
        cons_nfeats = 1
        edge_nfeats = 1
        var_nfeats = 12

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()
        
        self.pool = torch_geometric.nn.global_sort_pool
        
        self.final_mlp = torch.nn.Linear(self.k*emb_size,1)    



    def forward(self, batch):
        constraint_features, edge_indices, edge_features, variable_features = batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features
        
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        
        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        
        # 1 half convolutions
        #DO many convolutions and have constraint features concatenated
        
        #constraint_features = self.conv_v_to_c(variable_features, edge_indices, edge_features, constraint_features)
        #TODO pad if kmaxpool_k out of range
       
        constraint_pooled = self.pool(constraint_features, batch.batch, self.k)

        # DO 1d convs 
        output = constraint_pooled.view(batch.ptr.size(0)-1,self.emb_size*self.k)
        
        output = self.final_mlp(output)

        return torch.sigmoid(output)
    

class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    pass
    
 