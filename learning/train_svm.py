# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 12:23:29 2022

@author: aglabassi
"""

import os
import sys
import numpy as np
from pathlib import Path
import sklearn as sk
from sklearn import svm, datasets
from joblib import dump, load



def get_data(files):
    
    X = []
    y = []
    depths = []
    
    for file in files:
        
        f_array = np.loadtxt(file)
        features = f_array[:-1]
        comp_res = f_array[-1]
        X.append(features)
        y.append(comp_res)
        depths.append(np.array([f_array[18], f_array[-3]]))
        
    return np.array(X),np.array(y), np.array(depths)
        
        
    



if __name__ == '__main__':
    
    problem = 'FCMCNF'
    n_sample = -1
    n_epoch = 10
    
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-problem':
            problem = str(sys.argv[i + 1])
        if sys.argv[i] == '-n_epoch':
            n_epoch = int(sys.argv[i + 1])
        if sys.argv[i] == '-n_sample':
            n_sample = int(sys.argv[i + 1])
        
    

        
    train_files = [ str(path) for path in Path(os.path.join(os.path.dirname(__file__), 
                                                            f"../node_selection/data_svm/{problem}/train")).glob("*.csv") ][:n_sample]
    
    valid_files = [ str(path) for path in Path(os.path.join(os.path.dirname(__file__), 
                                                            f"../node_selection/data_svm/{problem}/valid")).glob("*.csv") ][:int(0.2*n_sample if n_sample != -1 else -1)]
    print(train_files)
    X,y,depths = get_data(train_files)
    X_valid, y_valid, depths_valid = get_data(valid_files)
    
    print(f"X shape {X.shape}")

    model = svm.LinearSVC()
    
    model.fit(X,y, np.exp(2.67/np.min(depths, axis=1)))
    
    valid_acc = model.score(X_valid,y_valid, np.min(depths_valid, axis=1))
    
    print(f"Accuracy on validation set : {valid_acc}")
    
    dump(model, f'policy_{problem}_svm.pkl')
        
        
        
        
        
        
        
    
        
