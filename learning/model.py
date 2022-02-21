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
from torch_geometric.nn import GraphConv
from data_type import BipartiteGraphPairData


class GNNPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.emb_size = emb_size = 32 #uniform node feature embedding dim
        self.emb_size_bounds = 4
        
        hidden_dim1 = 8
        hidden_dim2 = 4
        hidden_dim3 = 4
        
        # static data
        cons_nfeats = 1 
        edge_nfeats = 1
        bound_nfeats= 2
        var_nfeats = 6
        

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
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
        )
        
        #Bound EMBEDDING
        self.bound_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(bound_nfeats),
            torch.nn.Linear(bound_nfeats, self.emb_size_bounds),
            torch.nn.ReLU()
        )



        #double check
 
        self.convs = []
        
        self.conv1 = GraphConv((emb_size, emb_size), hidden_dim1 )
        self.conv2 = GraphConv((hidden_dim1, hidden_dim1), hidden_dim2 )
        self.conv3 = GraphConv((hidden_dim2, hidden_dim2), hidden_dim3 )
        
        self.convs = [ self.conv1, self.conv2, self.conv3 ]
        
        # out_size = hidden_dim3 if len(self.convs)==3 else emb_size
        
        # self.final_mlp = torch.nn.Sequential( 
        #                     torch.nn.Linear(2*out_size+2, 1, bias=False),
        #                     torch.nn.Sigmoid()
        #                     )
           

    
    def forward(self, batch, inv=False, epsilon=0.01):
        
        
        #create constraint masks. COnstraint associated with varialbes for which at least one of their bound have changed
        
        #variables for which at least one of their bound have changed

        #graph1 edges
        
        try :
       
            graph0 = (batch.constraint_features_s, 
                      batch.edge_index_s, 
                      batch.edge_attr_s, 
                      batch.variable_features_s, 
                      batch.bounds_s,
                      batch.constraint_features_s_batch,
                      batch.variable_features_s_batch)
            
        
            graph1 = (batch.constraint_features_t,
                      batch.edge_index_t, 
                      batch.edge_attr_t,
                      batch.variable_features_t,
                      batch.bounds_t,
                      batch.constraint_features_t_batch,
                      batch.variable_features_t_batch)
                
        except AttributeError:
            graph0 = (batch.constraint_features_s, 
                      batch.edge_index_s, 
                      batch.edge_attr_s, 
                      batch.variable_features_s,
                      batch.bounds_s)
            
        
            graph1 = (batch.constraint_features_t,
                      batch.edge_index_t, 
                      batch.edge_attr_t,
                      batch.variable_features_t,
                      batch.bounds_t)
        
        if inv:
            graph0, graph1 = graph1, graph0
        
        score0 = torch.linalg.norm(self.forward_graph(*graph0), dim=1) #concatenation of averages variable/constraint features after conv 
        score1 = torch.linalg.norm(self.forward_graph(*graph1), dim=1)
        
        return torch.sigmoid(-score0 + score1)
        
        
       
    def forward_graph(self, constraint_features, edge_indices, edge_features, 
                       variable_features, bbounds, constraint_batch=None, variable_batch=None):

        
        #Assume edge indice var to cons, constraint_mask of shape [Nconvs]       
        
        
        variable_features = self.var_embedding(variable_features)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        bbounds = self.bound_embedding(bbounds)
 
        
        edge_indices_reversed = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        
        
        
        for conv in self.convs:
            
            #Var to cons
            constraint_features_next = F.relu(conv((variable_features, constraint_features), 
                                              edge_indices,
                                              edge_weight=edge_features,
                                              size=(variable_features.size(0), constraint_features.size(0))))
            
            #cons to var 
            variable_features = F.relu(conv((constraint_features, variable_features), 
                                      edge_indices_reversed,
                                      edge_weight=edge_features,
                                      size=(constraint_features.size(0), variable_features.size(0))))
            
            constraint_features = constraint_features_next
            
            
            
            
            
            
        
        if constraint_batch is not None:
        
            constraint_avg = torch_geometric.nn.pool.avg_pool_x(constraint_batch, 
                                                                   constraint_features,
                                                                   constraint_batch)[0]
            variable_avg = torch_geometric.nn.pool.avg_pool_x(variable_batch, 
                                                                 variable_features,
                                                                 variable_features)[0]
        else:
            constraint_avg = torch.mean(constraint_features, axis=0, keepdim=True)
            variable_avg = torch.mean(variable_features, axis=0, keepdim=True)

        return torch.cat((variable_avg, constraint_avg, bbounds), dim=1)
    

    



 
