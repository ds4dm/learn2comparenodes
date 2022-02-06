#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:24:33 2021

@author: abdel
"""


#Test different nodesels under different problems

import sys
import os
import re
import numpy as np
import torch
from torch.multiprocessing import Process, set_start_method
from functools import partial
from utils import record_stats, display_stats
from pathlib import Path 



def distribute(n_instance, n_cpu):
    if n_cpu == 1:
        return [(0, n_instance)]
    
    k = n_instance //( n_cpu -1 )
    r = n_instance % (n_cpu - 1 )
    res = []
    for i in range(n_cpu -1):
        res.append( ((k*i), (k*(i+1))) )
    
    res.append(((n_cpu - 1) *k ,(n_cpu - 1) *k + r ))
    return res

       
if __name__ == "__main__":
    
    n_cpu = 4
    nodesels = ['gnn_untrained',
                'gnn_trained']
    
    problems = ["GISP"]
    normalize = True
    n_instance = -1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    verbose = True
    on_log = False
    default = True
    
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-n_cpu':
            n_cpu = int(sys.argv[i + 1])
        if sys.argv[i] == '-nodesels':
            nodesels = str(sys.argv[i + 1]).split(',')
        if sys.argv[i] == '-problems':
            problems = str(sys.argv[i + 1]).split(',')
        if sys.argv[i] == '-normalize':
            normalize = bool(int(sys.argv[i + 1]))
        if sys.argv[i] == '-n_instance':
            n_instance = int(sys.argv[i + 1])
        if sys.argv[i] == '-device':
            device = str(sys.argv[i + 1])
        if sys.argv[i] == '-verbose':
            verbose = bool(int(sys.argv[i + 1]))
        if sys.argv[i] == '-on_log':
            on_log = bool(int(sys.argv[i + 1]))    
        if sys.argv[i] == '-default':
            default = bool(int(sys.argv[i + 1]))    
            
    print("Evaluation")
    print(f"  Problem:                    {','.join(problems)}")
    print(f"  n_instance/problem:         {n_instance}")
    print(f"  Nodeselectors evaluated:    {','.join(nodesels)}")
    print(f"  Device for GNN inference:   {device}")
    print(f"  Normalize features:         {normalize}")
    print("----------------")
    
    
    if on_log:
        sys.stdout = open(os.path.join(os.path.dirname(__file__), 
                                       'evaluation.log'), 'w')
        

    for problem in problems:
        
        instances = list(Path(os.path.join(os.path.dirname(__file__), 
                                           f"./problem_generation/data/{problem}/test")).glob("*.lp"))
        if n_instance == -1 :
            n_instance = len(instances)
        
        instances = instances[:n_instance]
    
    
        processes = [  Process(name=f"worker {p}", 
                                        target=partial(record_stats,
                                                        nodesels=nodesels,
                                                        instances=instances[p1:p2], 
                                                        problem=problem,
                                                        device=torch.device(device),
                                                        normalize=normalize,
                                                        verbose=verbose,
                                                        default=default))
                        for p,(p1,p2) in enumerate(distribute(n_instance, n_cpu)) ]  
        
        
        try:
            set_start_method('spawn')
        except RuntimeError:
            ''

        a = list(map(lambda p: p.start(), processes)) #run processes
        b = list(map(lambda p: p.join(), processes)) #join processes
            
        min_n = min([ int( str(instance).split('=')[1].split('_m')[0] )  for instance in instances ] )
        max_n = max([ int( str(instance).split('=')[1].split('_m')[0] )  for instance in instances ] )
     
        display_stats(problem, nodesels, instances, min_n, max_n, default=default)
   

