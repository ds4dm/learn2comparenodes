#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 10:38:45 2021

@author: abdel
"""

#Imports 
def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("recorders", "../behaviour_generation/recorders.py" )

import os
import torch
import torch.nn.functional as F
import torch_geometric
from sklearn.model_selection import train_test_split
from tensorify import processed_data_files, GraphDataset, BipartiteNodeData
from model import GNNPolicy
osp = os.path

#function definition
# https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation.ipynb
def process(policy, data_loader, optimizer=None):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    mean_loss = 0
    mean_acc = 0
    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in data_loader:
            batch = batch.to(DEVICE)
            
            y_true = batch.y > 0
            y_pred = policy(batch)
            
            # Compute the usual cross-entropy classification loss
            loss = F.binary_cross_entropy(y_pred, y_true )

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            accuracy = ((y_pred > 0.5) == y_true).float().mean().item()

            mean_loss += loss.item() * batch.num_graphs
            mean_acc += accuracy * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    mean_acc /= n_samples_processed
    return mean_loss, mean_acc


#main

LEARNING_RATE = 0.001
NB_EPOCHS = 10
PATIENCE = 10
EARLY_STOPPING = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



train_files, valid_files = train_test_split(processed_data_files, train_size=0.8)

train_data = GraphDataset(train_files)
valid_data = GraphDataset(valid_files)

train_loader = torch_geometric.loader.DataLoader(train_data,batch_size=32, shuffle=True)
valid_loader = torch_geometric.loader.DataLoader(valid_data, batch_size=128, shuffle=False)


policy = GNNPolicy().to(DEVICE)
optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE) #ADAM is the best

for epoch in range(NB_EPOCHS):
    print(f"Epoch {epoch+1}")
    
    train_loss, train_acc = process(policy, train_loader, optimizer)
    print(f"Train loss: {train_loss:0.3f}, accuracy {train_acc:0.3f}" )

    valid_loss, valid_acc = process(policy, valid_loader, None)
    print(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}" )

torch.save(policy.state_dict(), 'trained_params.pkl')


