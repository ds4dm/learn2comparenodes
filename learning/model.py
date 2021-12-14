#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 20:44:58 2021

@author: abdel
from https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation.ipynb with some modifications
"""

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GeneralConv

class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, idx):
        data = torch.load(self.sample_files[idx])
        return data


class GNNPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.emb_size = emb_size = 16 #uniform node feature embedding dim
        self.k = 32 #kmax pooling
        self.n_convs = 16 #number of convolutions to perform parralelly
        drop_rate = 0.3
        hidden_dims = [8,8,8,1]
        # static data
        cons_nfeats = 1 
        edge_nfeats = 1
        var_nfeats = 6
        

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



        #double check
 
        self.convs = torch.nn.ModuleList( [ GeneralConv((emb_size, emb_size), hidden_dim , in_edge_channels=edge_nfeats) for hidden_dim in hidden_dims ])
        
        self.pool = torch_geometric.nn.global_sort_pool
        
        self.final_mlp = torch.nn.Sequential( 
                                    torch.nn.Linear(self.k*sum(hidden_dims), 256),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(drop_rate),
                                    torch.nn.Linear(256, 1, bias=False)
                                    )


    def forward(self, batch, inv=False, epsilon=0.01):
        
        
        #create constraint masks. COnstraint associated with varialbes for which at least one of their bound have changed
        
        #variables for which at least one of their bound have changed
        vars_to_keep = torch.where(
                                torch.max(
                                    torch.abs(batch.variable_features_s[:,[0,1]] - batch.variable_features_t[:,[0,1]]), dim=1 )[0] 
                                > epsilon )[0]

        #graph1 edges
        cond = [ batch.edge_index_s[0] == var  for var in vars_to_keep ] 
        edge_to_keep_s = torch.where(sum(cond))[0]
        cons_to_keep_s = batch.edge_index_s[1, edge_to_keep_s]
        
        #graph1 edges
        cond = [ batch.edge_index_t[0] == var  for var in vars_to_keep ] 
        edge_to_keep_t = torch.where(sum(cond))[0] 
        cons_to_keep_t = batch.edge_index_s[1, edge_to_keep_t]
        
    

        graph0 = (batch.constraint_features_s, 
                   batch.edge_index_s, 
                  batch.edge_attr_s, 
                  batch.variable_features_s, 
                  cons_to_keep_s,
                  batch.constraint_features_s_batch)
        
    
        graph1 = (batch.constraint_features_t,
                   batch.edge_index_t, 
                  batch.edge_attr_t,
                  batch.variable_features_t,
                  cons_to_keep_t,
                  batch.constraint_features_t_batch)
        
        
        
        if inv:
            graph0, graph1 = graph1, graph0
            
        
        score0 = self.forward_graphs(*graph0)
        score1 = self.forward_graphs(*graph1)
        
        return torch.sigmoid(-score0+score1).squeeze(1)
         
        
        
       
    def forward_graphs(self, constraint_features, edge_indices, edge_features, 
                       variable_features, constraint_batch, variable_batch):

        
        #Assume edge indice var to cons, constraint_mask of shape [Nconvs]       
        
        
        variable_features = self.var_embedding(variable_features)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        
        
        edge_indices_reversed = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        
        
        #Var to cons
        constraint_conveds = [ F.relu(conv((variable_features, constraint_features), 
                                  edge_indices,
                                  edge_feature=edge_features,
                                  size=(variable_features.size(0), constraint_features.size(0)))) for idx, conv in enumerate(self.convs) ]
        
        #cons to var 
        variable_conveds = [ F.relu(conv((constraint_features, variable_features), 
                                  edge_indices_reversed,
                                  edge_feature=edge_features,
                                  size=(constraint_features.size(0), variable_features.size(0)))) for idx, conv in enumerate(self.convs) ]
        
        
        
        
        constraint_conved = torch.cat(constraint_conveds, dim=1)  #N, sum(hiddendims)
        variable_conved = torch.cat(variable_conveds, dim=1)
        
        
        pooled_constraint = self.pool(constraint_conved, constraint_batch, self.k) #B,k*sum(hidden_dims)
        pooled_variable = self.pool(variable_conved, variable_batch, self.k) #B,k*sum(hidden_dims)
        
        return self.final_mlp(torch.cat((pooled_constraint, pooled_variable), dim=0))
    
        
    


 
