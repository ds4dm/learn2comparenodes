#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 10:38:45 2021

@author: abdel
"""

#Imports 

import os
import torch
import torch.nn.functional as F
import torch_geometric
from pathlib import Path
from model import GNNPolicy, GraphDataset
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
            
            y_true = 0.5*batch.y + 0.5*torch.abs(batch.y) #0,1 labels
            y_pred_proba = policy(batch)
            y_pred = torch.round(y_pred_proba)
            
            # Compute the usual cross-entropy classification loss
            loss = F.binary_cross_entropy(y_pred_proba, y_true )

            if optimizer is not None:
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


problems = ["GISP"]
LEARNING_RATE = 0.001
NB_EPOCHS = 1
PATIENCE = 10
EARLY_STOPPING = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


for problem in problems:

    train_files = [ str(path) for path in Path(f"../behaviour_generation/data/{problem}/train").glob("*.pt") ]
    valid_files = [ str(path) for path in Path(f"../behaviour_generation/data/{problem}/valid").glob("*.pt") ] 

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
    
    torch.save(policy.state_dict(),f'gnn_node_comparator_{problem}.pkl')
    

