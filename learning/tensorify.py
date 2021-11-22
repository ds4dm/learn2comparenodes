#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 10:07:05 2021

@author: abdel, inspired by https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation.ipynb
"""

def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("recorders", "../behaviour_generation/recorders.py" )

import os
import torch
import torch_geometric
import numpy as np
from pathlib import Path
from recorders import CompBehaviourSaver
osp = os.path
load_behaviour_from_pickle = CompBehaviourSaver.load_behaviour_from_pickle


#Maxime
class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a pair of node bipartite graphs observation 
    """
    def __init__(self, variable_features=None, constraint_features=None, edge_indices=None, edge_features=None, y=None, num_nodes=None):
        super().__init__()
        self.variable_features = variable_features
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.y = y
        self.num_nodes = num_nodes
    
   
    def __inc__(self, key, value, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs 
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == 'edge_index':
            return torch.tensor([[self.variable_features.size(0)], [self.constraint_features.size(0)]])
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


def process_raw_data(raw_files, processed_dir):
    i = 0
    for raw_file in raw_files:
        for g_data in load_behaviour_from_pickle(raw_file):
            
            variable_features = torch.FloatTensor(g_data[0])
            constraint_features = torch.FloatTensor(g_data[1])
            edge_indices = torch.LongTensor(g_data[2])
            edge_features = torch.FloatTensor(g_data[3])
            y = g_data[4]
            num_nodes = variable_features.size(0) + constraint_features.size(0)
            
            data = BipartiteNodeData(variable_features, constraint_features, edge_indices,
                                     edge_features, y, num_nodes=num_nodes)
    
            torch.save(data, osp.join(processed_dir, 'data_{}.pt'.format(i)))
            i += 1

    

raw_files = [str(path) for path in Path('../behaviour_generation/data/GISP').glob('*.pickle')]
processed_dir = "./processed"
try:
    os.makedirs(processed_dir)
except FileExistsError:
    ""
#process_raw_data(raw_files, processed_dir)
processed_data_files = np.array(list(Path(processed_dir).glob("*.pt")))









