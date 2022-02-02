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

load_src("recorders", "../node_selection/recorders.py" )

import os
import sys
import torch
import torch_geometric
from pathlib import Path
from model import GNNPolicy, GraphDataset
osp = os.path



def normalize_graph(constraint_features, 
                    edge_index,
                    edge_attr,
                    variable_features,
                    bounds,
                    depth,
                    bound_normalizor = 1000):
    

    
    #normalize objective
    obj_norm = torch.max(torch.abs(variable_features[:,2]), axis=0)[0].item()
   
    var_max_bounds = torch.max(torch.abs(variable_features[:,:2]), axis=1, keepdim=True)[0] 
    var_max_bounds += var_max_bounds == 0 #remove division by 0
    var_normalizor = var_max_bounds[edge_index[0]]
    cons_normalizor = constraint_features[edge_index[1]]
    normalizor = var_normalizor / cons_normalizor

    variable_features[:,2]/= obj_norm
    variable_features[:,:2] /= var_max_bounds
    constraint_features /= constraint_features
    edge_attr *= normalizor
    bounds/= bound_normalizor
    
    return (constraint_features, edge_index, edge_attr, variable_features, bounds, depth)

#main
def test1(data):
    assert(not torch.allclose(data.variable_features_s, data.variable_features_t))
    
    assert(not ((torch.allclose(data.constraint_features_s, data.constraint_features_t) 
                and torch.allclose(data.edge_attr_s, data.edge_attr_t))))
    assert( torch.max(data.variable_features_s) <= 1 and torch.min(data.variable_features_s) >= -1 )
    assert( torch.max(data.constraint_features_s) <= 1 and torch.min(data.constraint_features_s) >= -1 )

def inspect(geom_dataset):
    
    sub_milps = {}
    for data in geom_dataset:
        graph0 = (data.constraint_features_s, 
                  data.edge_index_s, 
                  data.edge_attr_s, 
                  data.variable_features_s)
        
        bounds0 = (int(graph0[0][-2].item()), int(graph0[0][-1].item())) 
        
        graph1 = (data.constraint_features_t,
                  data.edge_index_t, 
                  data.edge_attr_t,
                  data.variable_features_t)
        bounds1 = (int(graph1[0][-2].item()), int(graph1[0][-1].item()))
        
        if bounds0 not in sub_milps:
            sub_milps[bounds0] = graph0
        if bounds1 not in sub_milps:
            sub_milps[bounds1] = graph1
            
    cs = []
    lbs = []
    ubs = []
    coeffs= []
    print(sub_milps.keys())
    for m in sub_milps.values():
        
        cs += list( m[0].squeeze())
        lbs += list( m[3][:,0].squeeze())
        ubs += list( m[3][:,1].squeeze())
        coeffs += list( m[3][:,2].squeeze())
    print(coeffs)
    
    import matplotlib.pyplot as plt
    plt.figure(0)
    plt.hist(cs)
    plt.title('constraint rhs histogramme')
    
    plt.figure(1)
    plt.hist(lbs)
    plt.title('variable lb histogramme')
    
    plt.figure(2)
    plt.hist(ubs)
    plt.title('variable ub histogramme')


    plt.figure(3)
    plt.hist(coeffs)
    plt.title('objective coeff histogramme')
    





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
                
                #test1(batch)
        
            y_true = 0.5*batch.y + 0.5 #0,1 label
            y_proba = policy(batch)
            y_pred = torch.round(y_proba)
            
            # Compute the usual cross-entropy classification loss
            loss_fct.weight = torch.exp((1+torch.abs(batch.depth_s - batch.depth_t)) / 
                                        (torch.min(torch.vstack((batch.depth_s,  batch.depth_t)), axis=0)[0]))

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




if __name__ == "__main__":
    
    problems = ["GISP"]
    lr = 0.01
    n_epoch = 5
    patience = 10
    early_stopping = 20
    normalize = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_train = 16
    batch_valid  = 256
    
    loss_fn = torch.nn.BCELoss()
    optimizer_fn = torch.optim.Adam
    
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-problems':
            problems = str(sys.argv[i + 1]).split(',')
        if sys.argv[i] == '-lr':
            lr = float(sys.argv[i + 1])
        if sys.argv[i] == '-n_epoch':
            n_epoch = int(sys.argv[i + 1])
        if sys.argv[i] == '-patience':
            patience = int(sys.argv[i + 1])
        if sys.argv[i] == '-early_stopping':
            early_stopping = int(sys.argv[i + 1])
        if sys.argv[i] == '-normalize':
            normalize = bool(int(sys.argv[i + 1]))
        if sys.argv[i] == '-device':
            device = str(sys.argv[i + 1])
        if sys.argv[i] == '-batch_train':
            batch_train = int(sys.argv[i + 1])
        if sys.argv[i] == '-batch_valid':
            batch_valid = int(sys.argv[i + 1])
            
  
    
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    for problem in problems:
    
        train_files = [ str(path) for path in Path(os.path.join(os.path.dirname(__file__), 
                                                                f"../node_selection/data/{problem}/train")).glob("*.pt") ]
        
        valid_files = [ str(path) for path in Path(os.path.join(os.path.dirname(__file__), 
                                                                f"../node_selection/data/{problem}/valid")).glob("*.pt") ]
        
        train_data = GraphDataset(train_files)
        valid_data = GraphDataset(valid_files)
        
        #inspect(train_data[:100])
        
    # TO DO : learn something from the data
        train_loader = torch_geometric.loader.DataLoader(train_data, 
                                                         batch_size=batch_train, 
                                                         shuffle=True, 
                                                         follow_batch=['constraint_features_s', 
                                                                       'constraint_features_t',
                                                                       'variable_features_s',
                                                                       'variable_features_t'])
        
        valid_loader = torch_geometric.loader.DataLoader(valid_data, 
                                                         batch_size=batch_valid, 
                                                         shuffle=False, 
                                                         follow_batch=['constraint_features_s',
                                                                       'constraint_features_t',
                                                                       'variable_features_s',
                                                                       'variable_features_t'])
        
        policy = GNNPolicy().to(device)
        optimizer = optimizer_fn(policy.parameters(), lr=lr) #ADAM is the best
        
        print("-------------------------")
        print(f"GNN for problem {problem}")
        print(f"Training on:         {len(train_data)} samples")
        print(f"Validating on:       {len(valid_data)} samples")
        print(f"Batch Size Train:    {batch_train}")
        print(f"Batch Size Valid     {batch_valid}")
        print(f"Learning rate:       {lr} ")
        print(f"Number of epochs:    {n_epoch}")
        print(f"Normalize:           {normalize}")
        print(f"Device:              {device}")
        print(f"Loss fct:            {loss_fn}")
        print(f"Optimizer:           {optimizer_fn}")  
        print("-------------------------")
        
        
        for epoch in range(n_epoch):
            print(f"Epoch {epoch + 1}")
            
            train_loss, train_acc = process(policy, 
                                            train_loader, 
                                            loss_fn,
                                            device,
                                            optimizer=optimizer, 
                                            normalize=normalize)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            print(f"Train loss: {train_loss:0.3f}, accuracy {train_acc:0.3f}" )
        
            valid_loss, valid_acc = process(policy, 
                                            valid_loader, 
                                            loss_fn, 
                                            device,
                                            optimizer=None,
                                            normalize=normalize)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)
            
            print(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}" )
        
        torch.save(policy.state_dict(),f'policy_{problem}.pkl')
    
    
    decisions = [ policy(dvalid.to(device)).item() for dvalid in valid_data ]
    
    import matplotlib.pyplot as plt
    plt.figure(0)
    plt.hist(decisions)
    plt.title('decisions histogramme for valid set')
    plt.savefig("./hist.png")
    
    plt.figure(1)
    plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    plt.title('losses')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig("./losses.png")
    
    
    plt.figure(2)
    plt.plot(train_accs, label='train')
    plt.plot(valid_accs, label='valid')
    plt.title('accuracies')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig("./accuracies.png")
    



