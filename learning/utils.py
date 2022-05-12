#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:04:12 2022

@author: aglabassi
"""

import torch
import torch_geometric

def normalize_graph(constraint_features, 
                    edge_index,
                    edge_attr,
                    variable_features,
                    bounds,
                    depth,
                    bound_normalizor = 1000):
    
    
    #SMART
    obj_norm = torch.max(torch.abs(variable_features[:,2]), axis=0)[0].item()
    var_max_bounds = torch.max(torch.abs(variable_features[:,:2]), axis=1, keepdim=True)[0]  
    
    var_max_bounds.add_(var_max_bounds == 0)
    
    var_normalizor = var_max_bounds[edge_index[0]]
    cons_normalizor = constraint_features[edge_index[1], 0:1]
    normalizor = var_normalizor/(cons_normalizor + (cons_normalizor == 0))
    
    variable_features[:,2].div_(obj_norm)
    variable_features[:,:2].div_(var_max_bounds)
    constraint_features[:,0].div_(constraint_features[:,0] + (constraint_features[:,0] == 0) )
    edge_attr.mul_(normalizor)
    bounds.div_(bound_normalizor)
        
    
    
    #cheap 

    
    # #normalize objective
    # #obj_norm = torch.max(torch.abs(variable_features[:,2]), axis=0)[0].item()
    # #var_max_bounds = torch.max(torch.abs(variable_features[:,:2]), axis=1, keepdim=True)[0]  
    
    # #var_max_bounds.add_(var_max_bounds == 0)
    
    # #var_normalizor = var_max_bounds[edge_index[0]]
    # #cons_normalizor = constraint_features[edge_index[1]]
    # #normalizor = var_normalizor/(cons_normalizor)

    # variable_features[:,2].div_(100)
    # variable_features[:,:2].div_(300)
    # constraint_features.div_(300)
    # #edge_attr.mul_(normalizor)
    # bounds.div_(bound_normalizor)
    
    return (constraint_features, edge_index, edge_attr, variable_features, bounds, depth)



#function definition
# https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation.ipynb
def process(policy, data_loader, loss_fct, device, optimizer=None, normalize=True):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    mean_loss = 0
    mean_acc = 0
    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for idx,batch in enumerate(data_loader):
            
            
            batch = batch.to(device)
            if normalize:
                #IN place operations
                (batch.constraint_features_s,
                 batch.edge_index_s, 
                 batch.edge_attr_s,
                 batch.variable_features_s,
                 batch.bounds_s,
                 batch.depth_s)  =  normalize_graph(batch.constraint_features_s,  batch.edge_index_s, batch.edge_attr_s,
                                                    batch.variable_features_s, batch.bounds_s,  batch.depth_s)
                
                (batch.constraint_features_t,
                 batch.edge_index_t, 
                 batch.edge_attr_t,
                 batch.variable_features_t,
                 batch.bounds_t,
                 batch.depth_t)  =  normalize_graph(batch.constraint_features_t,  batch.edge_index_t, batch.edge_attr_t,
                                                    batch.variable_features_t, batch.bounds_t,  batch.depth_t)
                                                    
        
            y_true = 0.5*batch.y + 0.5 #0,1 label from -1,1 label
            y_proba = policy(batch)
            y_pred = torch.round(y_proba)
            
            # Compute the usual cross-entropy classification loss
            #loss_fct.weight = torch.exp((1+torch.abs(batch.depth_s - batch.depth_t)) / 
                            #(torch.min(torch.vstack((batch.depth_s,  batch.depth_t)), axis=0)[0]))

            l = loss_fct(y_proba, y_true)
            loss_value = l.item()
            if optimizer is not None:
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
            
            accuracy = (y_pred == y_true).float().mean().item()

            mean_loss += loss_value * batch.num_graphs
            mean_acc += accuracy * batch.num_graphs
            n_samples_processed += batch.num_graphs
            #print(y_proba.item(), y_true.item())

    mean_loss /= (n_samples_processed + ( n_samples_processed == 0))
    mean_acc /= (n_samples_processed  + ( n_samples_processed == 0))
    return mean_loss, mean_acc


def process_ranknet(policy, X, y, loss_fct, device, optimizer=None):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    mean_loss = 0
    mean_acc = 0
    n_samples_processed = 0
    X.to(device)

    with torch.set_grad_enabled(optimizer is not None):
        for idx,x in enumerate(X):
            yi = y[idx].to(device)
            y_true = 0.5*yi + 0.5 #0,1 label from -1,1 label
            y_proba = policy(x[:20].to(device), x[20:].to(device))
            y_pred = torch.round(y_proba)
            
            # Compute the usual cross-entropy classification loss
            #loss_fct.weight = torch.exp((1+torch.abs(batch.depth_s - batch.depth_t)) / 
                            #(torch.min(torch.vstack((batch.depth_s,  batch.depth_t)), axis=0)[0]))
            #print(y_proba)
            l = loss_fct(y_proba, y_true)
            #print(l)
            loss_value = l.item()
            if optimizer is not None:
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
            
            accuracy = (y_pred == y_true).float().mean().item()

            mean_loss += loss_value
            mean_acc += accuracy 
            n_samples_processed += 1
            #print(y_proba.item(), y_true.item())

    mean_loss /= (n_samples_processed + ( n_samples_processed == 0))
    mean_acc /= (n_samples_processed  + ( n_samples_processed == 0))
    return mean_loss, mean_acc

