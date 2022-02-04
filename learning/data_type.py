#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:08:53 2022

@author: aglabassi
"""

import torch
import torch_geometric

class BipartiteGraphPairData(torch_geometric.data.Data):
    """
    This class encode a pair of node bipartite graphs observation, s is graph0, t is graph1 
    """
    def __init__(self, constraint_features_s=None, edge_indices_s=None, edge_features_s=None, variable_features_s=None, bounds_s=None, depth_s=None, 
                 constraint_features_t=None, edge_indices_t=None, edge_features_t=None, variable_features_t=None,  bounds_t=None, depth_t=None,
                 y=None): 
        
        super().__init__()
        
        self.variable_features_s, self.constraint_features_s, self.edge_index_s, self.edge_attr_s, self.bounds_s, self.depth_s =  (
            variable_features_s, constraint_features_s, edge_indices_s, edge_features_s, bounds_s, depth_s)
        
        self.variable_features_t, self.constraint_features_t, self.edge_index_t, self.edge_attr_t, self.bounds_t, self.depth_t  = (
            variable_features_t, constraint_features_t, edge_indices_t, edge_features_t, bounds_t, depth_t)
        
        self.y = y
        

   
    def __inc__(self, key, value, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs 
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == 'edge_index_s':
            return torch.tensor([[self.variable_features_s.size(0)], [self.constraint_features_s.size(0)]])
        elif key == 'edge_index_t':
            return torch.tensor([[self.variable_features_t.size(0)], [self.constraint_features_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


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
