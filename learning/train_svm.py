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
    
    problems = ['FCMCNF']
    
    for problem in problems:
        
        train_files = [ str(path) for path in Path(os.path.join(os.path.dirname(__file__), 
                                                                f"../node_selection/data_svm/{problem}/train")).glob("*.csv") ]
        
        valid_files = [ str(path) for path in Path(os.path.join(os.path.dirname(__file__), 
                                                                f"../node_selection/data_svm/{problem}/valid")).glob("*.csv") ]

        X,y,depths = get_data(train_files)
    
        model = svm.LinearSVC()
        
        model.fit(X,y, np.exp(2.67/np.min(depths, axis=1)))
        
        dump(model, f'policy_{problem}_svm.pkl')
        
        
        
        
        
        
        
    
        