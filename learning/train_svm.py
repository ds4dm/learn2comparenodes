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
    #TODO
    
    return np.random.rand(100,20), np.random.randint(0,2,size=(100,)), np.random.randint(1,62,size=(100,2))



if __name__ == '__main__':
    
    problems = ['GISP']
    
    for problem in problems:
        
        train_files = [ str(path) for path in Path(os.path.join(os.path.dirname(__file__), 
                                                                f"../node_selection/data_svm/{problem}/train")).glob("*.pt") ]
        
        valid_files = [ str(path) for path in Path(os.path.join(os.path.dirname(__file__), 
                                                                f"../node_selection/data_svm/{problem}/valid")).glob("*.pt") ]
        
        X,y,depths = get_data(train_files)
    
        model = svm.LinearSVC()
        
        model.fit(X,y, np.exp(2.67/np.min(depths, axis=1)))
        
        dump(model, f'policy_{problem}_svm.pkl')
        
        
        
        
        
        
        
    
        