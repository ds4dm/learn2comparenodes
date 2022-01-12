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
        
        self.emb_size = emb_size = 32 #uniform node feature embedding dim
        
        hidden_dims = [8,8,1]
        
        final_mlp_hidden_dim = 128
        
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
 
        self.convs = torch.nn.ModuleList( GeneralConv((emb_size, emb_size), 
                                                        hidden_dim , 
                                                        in_edge_channels=edge_nfeats) 
                                           for hidden_dim in hidden_dims )
        
        self.final_mlp = torch.nn.Sequential( 
                                    torch.nn.LayerNorm(2*sum(hidden_dims)),
                                    torch.nn.Linear(2*sum(hidden_dims), final_mlp_hidden_dim),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(final_mlp_hidden_dim, 1, bias=False),
                                    torch.nn.Sigmoid()
                                    )
     

    
    def forward(self, batch, inv=False, epsilon=0.01):
        
        
        #create constraint masks. COnstraint associated with varialbes for which at least one of their bound have changed
        
        #variables for which at least one of their bound have changed

        #graph1 edges
        
        try :
       
            graph0 = (batch.constraint_features_s, 
                      batch.edge_index_s, 
                      batch.edge_attr_s, 
                      batch.variable_features_s, 
                      batch.constraint_features_s_batch,
                      batch.variable_features_s_batch)
            
        
            graph1 = (batch.constraint_features_t,
                      batch.edge_index_t, 
                      batch.edge_attr_t,
                      batch.variable_features_t,
                      batch.constraint_features_t_batch,
                      batch.variable_features_t_batch)
                
        except AttributeError:
            graph0 = (batch.constraint_features_s, 
                      batch.edge_index_s, 
                      batch.edge_attr_s, 
                      batch.variable_features_s)
            
        
            graph1 = (batch.constraint_features_t,
                      batch.edge_index_t, 
                      batch.edge_attr_t,
                      batch.variable_features_t)
        
        if inv:
            graph0, graph1 = graph1, graph0
        
        score0 = self.forward_graphs(*self.normalize_graph(*graph0)) #concatenation of averages variable/constraint features after conv 
        score1 = self.forward_graphs(*self.normalize_graph(*graph1))

        return self.final_mlp(-score0 + score1).squeeze(1)
        
        
       
    def forward_graphs(self, constraint_features, edge_indices, edge_features, 
                       variable_features, constraint_batch=None, variable_batch=None):

        
        #Assume edge indice var to cons, constraint_mask of shape [Nconvs]       
        
        
        variable_features = self.var_embedding(variable_features)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        
        
        edge_indices_reversed = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        
        
        #Var to cons
        constraint_conveds = [ F.relu(conv((variable_features, constraint_features), 
                                  edge_indices,
                                  edge_feature=edge_features,
                                  size=(variable_features.size(0), constraint_features.size(0))))
                              for idx, conv in enumerate(self.convs) ]
        
        #cons to var 
        variable_conveds = [ F.relu(conv((constraint_features, variable_features), 
                                  edge_indices_reversed,
                                  edge_feature=edge_features,
                                  size=(constraint_features.size(0), variable_features.size(0)))) 
                            for idx, conv in enumerate(self.convs) ]
        
        
        
        
        constraint_conved = torch.cat(constraint_conveds, dim=1)  #N, sum(hiddendims)
        variable_conved = torch.cat(variable_conveds, dim=1)
        
        if constraint_batch is not None:
        
            constraint_conved = torch_geometric.nn.pool.avg_pool_x(constraint_batch, 
                                                                   constraint_conved,
                                                                   constraint_batch)[0]
            variable_conved = torch_geometric.nn.pool.avg_pool_x(variable_batch, 
                                                                 variable_conved,
                                                                 variable_batch)[0]
        else:
            constraint_conved = torch.mean(constraint_conved, axis=0, keepdim=True)
            variable_conved = torch.mean(variable_conved, axis=0, keepdim=True)
            

        return torch.cat(( variable_conved, constraint_conved ), dim=1)
    
    
    def normalize_graph(self, constraint_features, 
                        edge_index,
                        edge_attr,
                        variable_features,
                        constraint_features_t_batch=None,
                        variable_features_t_batch=None):
        

        #Normalize variable bounds to value between 0,1
        vars_to_normalize = torch.where( torch.max(torch.abs(variable_features[:, :2]), axis=1)[0] > 1)[0]

        coeffs = torch.max(torch.abs(variable_features[vars_to_normalize, :2]) , axis=1)[0]
        
        for v, cf in zip(vars_to_normalize, coeffs):
            
            #normaize feature bound
            variable_features[ v, :2] /= cf
            
            #update obj coeff and associated edges
            variable_features[ v, 2 ] *= cf 
            
            associated_edges = torch.where(edge_index[0] == v)[0]
            edge_attr[associated_edges] *= cf
            
        
        
        #Normalize constraints 
        for c in range(constraint_features.shape[0]):
            
            associated_edges =  torch.where(edge_index[1] == c)[0]
            normalizer = max(torch.max(torch.abs(edge_attr[associated_edges]), axis=0)[0], 
                             torch.abs(constraint_features[c]))
            
            #normalize associated edges
            edge_attr[associated_edges] /= normalizer
            
            #normalize right hand side
            constraint_features[c] /= normalizer
        
        #normalize objective
        normalizer = torch.max(torch.abs(variable_features[:,2]), axis=0)[0]
        variable_features[:,2] /= normalizer


        return (constraint_features, edge_index, edge_attr, variable_features,
                constraint_features_t_batch, variable_features_t_batch)
    
        
    


 
