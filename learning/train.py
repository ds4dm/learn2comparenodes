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
import torch
import torch_geometric
from pathlib import Path
from model import GNNPolicy, GraphDataset
osp = os.path



def normalize_graph(constraint_features, 
                    edge_index,
                    edge_attr,
                    variable_features):
    
    # variable_normalizer = torch.max(torch.abs(variable_features[:,:2]))
    # variable_features[:,:2] /= variable_normalizer
    # variable_features[:,2] /= torch.max(torch.abs(variable_features[:,2]))
    
    # constraint_features /= variable_normalizer
    
    
    # def normalize_cons(c):
    #     associated_edges =  torch.where(edge_index[1] == c)[0]
    #     normalizer = max(torch.max(torch.abs(edge_attr[associated_edges]), axis=0)[0], 
    #                      torch.abs(constraint_features[c]))
    #     #normalize associated edges
    #     edge_attr[associated_edges] /= normalizer
        
    #     #normalize right hand side
    #     constraint_features[c] /= normalizer
    
    # vars_to_normalize = torch.where( torch.max(torch.abs(variable_features[:, :2]), axis=1)[0] > 1)[0]

    # coeffs = torch.max(torch.abs(variable_features[vars_to_normalize, :2]) , axis=1)[0]
    
    # for v, cf in zip(vars_to_normalize, coeffs):
     
    #   #normaize feature bound
    #   variable_features[ v, :2] = variable_features[ v, :2]/cf
     
    #   #update obj coeff and associated edges
    #   variable_features[ v, 2 ] = variable_features[ v, 2 ]*cf 
     
    #   associated_edges = torch.where(edge_index[0] == v)[0]
    #   edge_attr[associated_edges] = edge_attr[associated_edges]*cf
    
        
    # #Normalize constraints 
    # for c in range(constraint_features.shape[0]):
        
    #     associated_edges =  torch.where(edge_index[1] == c)[0]
    #     normalizer = max(torch.max(torch.abs(edge_attr[associated_edges]), axis=0)[0], 
    #                       torch.abs(constraint_features[c]))
        
    #     #normalize associated edges
    #     edge_attr[associated_edges] = edge_attr[associated_edges] / normalizer
        
    #     #normalize right hand side
    #     constraint_features[c] = constraint_features[c] / normalizer
    
    # #normalize objective
    # normalizer = torch.max(torch.abs(variable_features[:,2]), axis=0)[0]
    # variable_features[:,2] = variable_features[:,2] / normalizer
    
    constraint_features /= 300.0
    variable_features[:3] /= 300.0
    edge_attr /= 300.0
    
    return (constraint_features, edge_index, edge_attr, variable_features)

#main
def test1(data):
    assert(not torch.allclose(data.variable_features_s, data.variable_features_t))
    
    assert(not ((torch.allclose(data.constraint_features_s, data.constraint_features_t) 
                and torch.allclose(data.edge_attr_s, data.edge_attr_t))))
    assert( torch.max(data.variable_features_s) <= 1 and torch.min(data.variable_features_s) >= -1 )
    assert( torch.max(data.constraint_features_s) <= 1 and torch.min(data.constraint_features_s) >= -1 )
    assert( torch.max(data.edge_attr_s) <= 1 and torch.min(data.edge_attr_s) >= -1 )
    


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
            normalize_graph(batch.constraint_features_s, batch.edge_index_s, batch.edge_attr_s, batch.variable_features_s)
            normalize_graph(batch.constraint_features_t, batch.edge_index_t, batch.edge_attr_t, batch.variable_features_t)
            #test1(batch)
            
            y_true = 0.5*batch.y + 0.5 #0,1 labels

            y_proba = policy(batch)
            y_pred = torch.round(y_proba)
            
            
            # Compute the usual cross-entropy classification loss
            l = loss_fct(y_proba, y_true )
            loss_value = l.item()
            if optimizer is not None:
                if balance:
                    l += loss_fct(policy(batch, inv=True), -1*y_true+1)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                
          
            accuracy = (y_pred == y_true).float().mean().item()

            mean_loss += loss_value * batch.num_graphs
            mean_acc += accuracy * batch.num_graphs
            n_samples_processed += batch.num_graphs
            print(y_pred, y_true)

    mean_loss /= n_samples_processed + 1
    mean_acc /= n_samples_processed + 1
    return mean_loss, mean_acc


problems = ["GISP"]
LEARNING_RATE = 0.005
NB_EPOCHS = 50
PATIENCE = 10
EARLY_STOPPING = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOSS = torch.nn.BCELoss()
OPTIMIZER = torch.optim.Adam


train_losses = []
valid_losses = []
train_accs = []
valid_accs = []
for problem in problems:

    train_files = [ str(path) for path in Path(f"../node_selection/data/{problem}/train").glob("*.pt") ]
    
    valid_files = [ str(path) for path in Path(f"../node_selection/data/{problem}/valid").glob("*.pt") ]
    
    train_data = GraphDataset(train_files)
    valid_data = GraphDataset(valid_files)
    print(train_data)
    print(valid_data)
    #inspect(train_data[:100])
    
# TO DO : learn something from the data
    train_loader = torch_geometric.loader.DataLoader(train_data, 
                                                     batch_size=1, 
                                                     shuffle=True, 
                                                     follow_batch=['constraint_features_s', 
                                                                   'constraint_features_t',
                                                                   'variable_features_s',
                                                                   'variable_features_t'])
    
    valid_loader = torch_geometric.loader.DataLoader(valid_data, 
                                                     batch_size=1, 
                                                     shuffle=False, 
                                                     follow_batch=['constraint_features_s',
                                                                   'constraint_features_t',
                                                                   'variable_features_s',
                                                                   'variable_features_t'])
    
    policy = GNNPolicy().to(DEVICE)
    optimizer = OPTIMIZER(policy.parameters(), lr=LEARNING_RATE) #ADAM is the best
    
    
    for epoch in range(NB_EPOCHS):
        print(f"Epoch {epoch + 1}")
        
        train_loss, train_acc = process(policy, train_loader, LOSS, optimizer)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print(f"Train loss: {train_loss:0.3f}, accuracy {train_acc:0.3f}" )
    
        valid_loss, valid_acc = process(policy, valid_loader, LOSS, None)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        print(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}" )
    
    torch.save(policy.state_dict(),f'policy_{problem}.pkl')


decisions = [ policy(dvalid.to(DEVICE)).item() for dvalid in valid_data ]

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




