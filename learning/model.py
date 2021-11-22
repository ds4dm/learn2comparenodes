#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 20:44:58 2021

@author: abdel
from https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation.ipynb with some modifications
"""

import torch
import torch_geometric

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
        
        #HYPERPARAMETERS
        self.emb_size = emb_size = 16 #uniform node feature embedding dim
        self.k = 16 #kmax pooling
        self.n_convs = 4 #number of convolutions to perform parralelly
        drop_rate = 0.35
        
        # static data
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

        self.convs = torch.nn.ModuleList( [ BipartiteGraphConvolution(emb_size) for i in range(self.n_convs) ])
        
        self.pool = torch_geometric.nn.global_sort_pool
        
        self.final_mlp = torch.nn.Sequential( 
                                    torch.nn.Linear(self.k*emb_size*self.n_convs, self.k*emb_size),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(drop_rate),
                                    torch.nn.Linear(self.k*emb_size, 1)
                                    )



    def forward(self, batch):
        
        constraint_features, edge_indices, edge_features, variable_features = batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        
        # 1 half convolutions (is sufficient)
        
        constraint_conveds = [ conv(variable_features, edge_indices, edge_features, constraint_features) for conv in self.convs ]
       
        constraint_pooleds = [ self.pool(constraint_conved, batch.batch, self.k) for constraint_conved in constraint_conveds ]
        
        constraint_pooleds_cat = torch.cat(constraint_pooleds, dim=1)
        
        # DO 1d convs 
        output = constraint_pooleds_cat
        
        output = self.final_mlp(output).squeeze(-1)

        return torch.sigmoid(output)
    

class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    
    def __init__(self, emb_size):
        super().__init__('add')
        
        dropout_rate = 0.4
        
        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size)
        )
           
        self.post_conv_module = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size)
        )

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2*emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]), 
                                node_features=(left_features, right_features), edge_features=edge_features)
        
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(self.feature_module_left(node_features_i) 
                                           + self.feature_module_edge(edge_features) 
                                           + self.feature_module_right(node_features_j))
        return output
    
 
