# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch_geometric
from pathlib import Path
from model import RankNet
from data_type import GraphDataset
from utils import process, process_ranknet
import numpy as np

def get_data(files):
    
    X = []
    y = []
    depths = []
    
    for file in files:
        
        f_array = np.loadtxt(file).tolist()
        features = f_array[:-1]
        comp_res = f_array[-1]
        X.append(features)
        y.append(comp_res)
        depths.append(np.array([f_array[18], f_array[-3]]))
    
    
    return torch.tensor(np.array(X, dtype=np.float), dtype=torch.float), torch.tensor(np.array(y, dtype=np.long), dtype=torch.long).unsqueeze(1), torch.tensor(np.array(depths))


    
if __name__ == "__main__":
    
    problem = "WPMS"
    lr = 0.005
    n_epoch = 2
    n_sample = -1
    patience = 10
    early_stopping = 20
    normalize = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_train = 1
    batch_valid  = 1
    
    loss_fn = torch.nn.BCELoss()
    optimizer_fn = torch.optim.Adam
    
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-problem':
            problem = str(sys.argv[i + 1])
        if sys.argv[i] == '-lr':
            lr = float(sys.argv[i + 1])
        if sys.argv[i] == '-n_epoch':
            n_epoch = int(sys.argv[i + 1])
        if sys.argv[i] == '-n_sample':
            n_sample = int(sys.argv[i + 1])
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


    train_files = [ str(path) for path in Path(os.path.join(os.path.dirname(__file__), 
                                                            f"../node_selection/data_svm/{problem}/train")).glob("*.csv") ][:n_sample]
    
    valid_files = [ str(path) for path in Path(os.path.join(os.path.dirname(__file__), 
                                                            f"../node_selection/data_svm/{problem}/valid")).glob("*.csv") ][:int(0.2*n_sample if n_sample != -1 else -1)]
    
    if problem == 'FCMCNF':
        train_files = train_files + valid_files[3000:]
        valid_files = valid_files[:3000]
    
    X_train, y_train, _ = get_data(train_files)
    X_valid, y_valid, _ = get_data(valid_files)
        
    
    X_train.to(device)
    y_train.to(device)
    X_valid.to(device)
    y_valid.to(device)    

    
    policy = RankNet().to(device)
    optimizer = optimizer_fn(policy.parameters(), lr=lr) #ADAM is the best
    
    print("-------------------------")
    print(f"Ranknet for problem {problem}")
    print(f"Training on:          {len(X_train)} samples")
    print(f"Validating on:        {len(X_valid)} samples")
    print(f"Batch Size Train:     {1}")
    print(f"Batch Size Valid      {1}")
    print(f"Learning rate:        {lr} ")
    print(f"Number of epochs:     {n_epoch}")
    print(f"Normalize:            {normalize}")
    print(f"Device:               {device}")
    print(f"Loss fct:             {loss_fn}")
    print(f"Optimizer:            {optimizer_fn}")  
    print(f"Model's Size:         {sum(p.numel() for p in policy.parameters())} parameters ")
    print("-------------------------") 
    
    
    for epoch in range(n_epoch):
        print(f"Epoch {epoch + 1}")
        
        train_loss, train_acc = process_ranknet(policy, 
                                        X_train, y_train, 
                                        loss_fn,
                                        device,
                                        optimizer=optimizer)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print(f"Train loss: {train_loss:0.3f}, accuracy {train_acc:0.3f}" )
    
        valid_loss, valid_acc = process_ranknet(policy, 
                                        X_valid, y_valid, 
                                        loss_fn, 
                                        device,
                                        optimizer=None)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        
        print(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}" )
    
    torch.save(policy.state_dict(),f'policy_{problem}_ranknet.pkl')
    
    

