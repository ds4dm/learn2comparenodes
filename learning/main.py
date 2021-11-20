#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 10:38:45 2021

@author: abdel
"""

def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("recorders", "../behaviour_generation/recorders.py" )

import os
import torch
import torch_geometric
from sklearn.model_selection import train_test_split
from tensorify import processed_data_files, GraphDataset 
from model import GNNPolicy
osp = os.path


LEARNING_RATE = 0.001
NB_EPOCHS = 50
PATIENCE = 10
EARLY_STOPPING = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



train_files, valid_files = train_test_split(processed_data_files, train_size=0.8)

train_data = GraphDataset(train_files)
valid_data = GraphDataset(valid_files)

train_loader = torch_geometric.data.DataLoader(train_data,batch_size=32, shuffle=True)
valid_loader = torch_geometric.data.DataLoader(valid_data, batch_size=128, shuffle=False)


observation = train_data[0]
policy = GNNPolicy()
policy(observation.constraint_features, observation.edge_index, observation.edge_attr, observation.variable_features)



