#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 10:07:05 2021

@author: abdel, inspired by https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation.ipynb
"""
import torch
import torch_geometric
import numpy as np
from pathlib import Path
import gzip
from ..behaviour_generation.recorders.CompBehaviourSaver import load_behaviour_from_pickle

#Maxime
class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite` 
    observation function in a format understood by the pytorch geometric data handlers.
    """
    def __init__(self, constraint_features, edge_indices, edge_features, variable_features,
                 candidates, candidate_choice, candidate_scores):
        super().__init__()
        self.constraint_features = torch.FloatTensor(constraint_features)
        self.edge_index = torch.LongTensor(edge_indices.astype(np.int64))
        self.edge_attr = torch.FloatTensor(edge_features).unsqueeze(1)
        self.variable_features = torch.FloatTensor(variable_features)
        self.candidates = candidates
        self.nb_candidates = len(candidates)
        self.candidate_choices = candidate_choice
        self.candidate_scores = candidate_scores

    def __inc__(self, key, value):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs 
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'candidates':
            return self.variable_features.size(0)
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

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        with gzip.open(self.sample_files[index], 'rb') as f:
            g1_data, g2_data, comp = load_behaviour_from_pickle(f) #generator of G1,G2, comp triplets
            #G1 is a quadruplet format (var_attributes, cons_attributes, edge_idx, edge attributes ) 


        graph1 = BipartiteNodeData(*g1_data)
        graph2 = BipartiteNodeData(*g2_data)

        
        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = sample_observation.row_features.shape[0]+sample_observation.column_features.shape[0]
        
        return graph
    

sample_files = [str(path) for path in Path('../behaviour_generation/data/GISP').glob('*.pkl')]
train_files = sample_files[:int(0.8*len(sample_files))]
valid_files = sample_files[int(0.8*len(sample_files)):]

train_data = GraphDataset(train_files)
train_loader = torch_geometric.data.DataLoader(train_data, batch_size=32, shuffle=True)
valid_data = GraphDataset(valid_files)
valid_loader = torch_geometric.data.DataLoader(valid_data, batch_size=128, shuffle=False)