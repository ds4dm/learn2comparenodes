#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:24:33 2021

@author: abdel
"""


#Test different nodesels under different problems



from pathlib import Path 
import numpy as np
import torch
import sys
import multiprocessing as md
from functools import partial
from utils import record_stats, display_stats, clear_records
import os
import re

       
if __name__ == "__main__":
    
    cpu_count = 4
    nodesels = ['breadthfirst', 'dfs', 'bfs', 'estimate','oracle']
    problems = ["GISP"]
    normalize = True
    n_instance = 4
    n_trial = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    verbose = False
    on_log = False
    
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-n_cpu':
            cpu_count = int(sys.argv[i + 1])
        if sys.argv[i] == '-nodesels':
            nodesels = str(sys.argv[i + 1]).split(',')
        if sys.argv[i] == '-problems':
            problems = str(sys.argv[i + 1]).split(',')
        if sys.argv[i] == '-normalize':
            normalize = bool(int(sys.argv[i + 1]))
        if sys.argv[i] == '-n_instance':
            n_instance = int(sys.argv[i + 1])
        if sys.argv[i] == '-n_trial':
            n_trial = int(sys.argv[i + 1])
        if sys.argv[i] == '-device':
            device = str(sys.argv[i + 1])
        if sys.argv[i] == '-verbose':
            verbose = bool(int(sys.argv[i + 1]))
        if sys.argv[i] == '-on_log':
            on_log = bool(int(sys.argv[i + 1]))    

    print("Evaluation")
    print(f"  Problem:                    {','.join(problems)}")
    print(f"  n_instance/problem:         {n_instance}")
    print(f"  n_trial/instance:           {n_trial}")
    print(f"  Nodeselectors evaluated:    {','.join(nodesels)}")
    print(f"  Device for GNN inference:   {device}")
    print(f"  Normalize features:         {normalize}")
    print("----------------")
    
    if device == 'cuda':
        nodesels_gpu = [ nodesel for nodesel in nodesels if re.match('gnn*', nodesel) ] 
        nodesels_cpu = [ nodesel for nodesel in nodesels if not re.match('gnn*', nodesel) ] 
    else:
        nodesels_gpu = []
        nodesels_cpu = nodesels
        
    if on_log:
        sys.stdout = open(os.path.join(os.path.dirname(__file__), 
                                       'evaluation.log'), 'w')
        

    for problem in problems:

        #clear records
        clear_records(problem, nodesels)

        instances = sorted(list(Path(os.path.join(os.path.dirname(__file__), 
                                           f"./problem_generation/data/{problem}/test")).glob("*.lp"))[:n_instance])
        for _ in range(n_trial):
            
            chunck_size = int(np.ceil(len(instances)/cpu_count))
            processes = [  md.Process(name=f"worker {p}", 
                                            target=partial(record_stats,
                                                            nodesels=nodesels_cpu,
                                                            instances=instances[ p*chunck_size : (p+1)*chunck_size], 
                                                            problem=problem,
                                                            device=torch.device('cpu'),
                                                            normalize=normalize,
                                                            verbose=verbose))
                            for p in range(cpu_count) ]
            
            
            
            a = list(map(lambda p: p.start(), processes)) #run processes
            
            record_stats(nodesels_gpu, instances, problem, torch.device('cuda'), normalize, verbose, default=False)
            
            
            b = list(map(lambda p: p.join(), processes)) #join processes
     
        print("==================================")
        print(f"SUMMARIES for {n_trial} trials with {n_instance} instances")
        display_stats(nodesels, problem)
   

