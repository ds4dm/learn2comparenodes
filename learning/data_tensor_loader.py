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
from sklearn.model_selection import train_test_split
osp = os.path
load_behaviour_from_pickle = CompBehaviourSaver.load_behaviour_from_pickle


LEARNING_RATE = 0.001
NB_EPOCHS = 50
PATIENCE = 10
EARLY_STOPPING = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Maxime
class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a pair of node bipartite graphs observation 
    """
    def __init__(self, variable_features, constraint_features, edge_indices, edge_features, comp_res ):
        super().__init__()
        self.variable_features = torch.FloatTensor(variable_features)
        self.constraint_features = torch.FloatTensor(constraint_features)
        self.edge_index = torch.LongTensor(edge_indices.astype(np.int64))
        self.edge_attr = torch.FloatTensor(edge_features)
        self.comp_res = [comp_res]
        
   
    def __inc__(self, key, value):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs 
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == 'edge_index':
            return torch.tensor([[self.variable_features.size(0)], [self.constraint_features.size(0)]])
        else:
            return super().__inc__(key, value)
        
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
    


def process(raw_files, processed_dir):
    i = 0
    for raw_file in raw_files:
        for g_data in load_behaviour_from_pickle(raw_file):
            data = BipartiteNodeData(*g_data)
            data.num_nodes =  g_data[0].shape[0] + g_data[1].shape[0]
            torch.save(data, osp.join(processed_dir, 'data_{}.pt'.format(i)))
            i += 1
    return i

    

raw_files = [str(path) for path in Path('../behaviour_generation/data/GISP').glob('*.pickle')]
processed_dir = "./processed"
try:
    os.makedirs(processed_dir)
except FileExistsError:
    ""
process(raw_files, processed_dir)
processed_files = np.array(list(Path(processed_dir).glob("*.pt")))

train_files, valid_files = train_test_split(processed_files, train_size=0.8)

train_loader = torch_geometric.data.DataLoader(GraphDataset(train_files),batch_size=32, shuffle=True)
valid_loader = torch_geometric.data.DataLoader(GraphDataset(valid_files), batch_size=128, shuffle=False)