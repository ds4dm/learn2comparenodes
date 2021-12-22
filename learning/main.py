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
from pathlib import Path
from model import GNNPolicy, GraphDataset
from sklearn.model_selection import train_test_split
osp = os.path

#function definition
# https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation.ipynb
def process(policy, data_loader, loss_fct, optimizer=None, balance=True):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    mean_loss = 0
    mean_acc = 0
    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for idx,batch in enumerate(data_loader):
            
            batch = batch.to(DEVICE)
            test1(batch)
            
            y_true = 0.5*batch.y + 0.5*torch.abs(batch.y) #0,1 labels
            y_proba = policy(batch)
            y_pred = torch.round(y_proba)
            
            
            # Compute the usual cross-entropy classification loss
            loss = loss_fct(y_proba, y_true )

            if optimizer is not None:
                if balance:
                    y_proba_inv = policy(batch, inv=True)
                    loss += loss_fct(y_proba_inv, -1*y_true + 1)
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
          
            accuracy = (y_pred == y_true).float().mean().item()

            mean_loss += loss.item() * batch.num_graphs
            mean_acc += accuracy * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    mean_acc /= n_samples_processed
    return mean_loss, mean_acc

#main
def test1(data):
    assert(not torch.allclose(data.variable_features_s, data.variable_features_t))
    #assert(not torch.allclose(data.constraint_features_s, data.constraint_features_t))
    #assert(not torch.allclose(data.edge_features_s, data.edge_features_t))
    
    assert( torch.max(data.variable_features_s) <= 1 and torch.min(data.variable_features_s) >= -1 )
    assert( torch.max(data.constraint_features_s) <= 1 and torch.min(data.constraint_features_s) >= -1 )
    assert( torch.max(data.edge_attr_s) <= 1 and torch.min(data.edge_attr_s) >= -1 )
    
    



problems = ["GISP"]
LEARNING_RATE = 0.005
NB_EPOCHS = 100
PATIENCE = 10
EARLY_STOPPING = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOSS = torch.nn.BCELoss()
OPTIMIZER = torch.optim.Adam

for problem in problems:

    train_files = [ str(path) for path in Path(f"../behaviour_generation/data/{problem}/train").glob("*.pt") ][:100]
    
    valid_files = [ str(path) for path in Path(f"../behaviour_generation/data/{problem}/valid").glob("*.pt") ][:20]
    
    train_data = GraphDataset(train_files)
    valid_data = GraphDataset(valid_files)
    
# TO DO : learn something from the data
    train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=16, shuffle=True, follow_batch=['constraint_features_s', 'constraint_features_t','variable_features_s','variable_features_t'])
    valid_loader = torch_geometric.loader.DataLoader(valid_data, batch_size=128, shuffle=False, follow_batch=['constraint_features_s', 'constraint_features_t', 'variable_features_s', 'variable_features_t'])
    
    policy = GNNPolicy().to(DEVICE)
    optimizer = OPTIMIZER(policy.parameters(), lr=LEARNING_RATE) #ADAM is the best
    
    
    for epoch in range(NB_EPOCHS):
        print(f"Epoch {epoch+1}")
        
        train_loss, train_acc = process(policy, train_loader, LOSS, optimizer)
        print(f"Train loss: {train_loss:0.3f}, accuracy {train_acc:0.3f}" )
    
        valid_loss, valid_acc = process(policy, valid_loader, LOSS, None)
        print(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}" )
    
    torch.save(policy.state_dict(),f'gnn_node_comparator_{problem}.pkl')



    
