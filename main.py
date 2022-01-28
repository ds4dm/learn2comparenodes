#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:24:33 2021

@author: abdel
"""


#Test different nodesels under different problems



from pathlib import Path 
import pyscipopt.scip as sp
import numpy as np
import torch
import sys
import multiprocessing as md
from functools import partial
from utils import record_stats, display_stats

       
if __name__ == "__main__":
    
    cpu_count = 1
    nodesels = [ 'gnn_trained', 'gnn_untrained','oracle_0', 'estimate', 'random']
    problems = ["GISP"]
    normalize = True
    n_instance = 10
    n_trial = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    verbose = False
    
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
    

    print("Evaluation")
    print(f"  Problem:                    {','.join(problems)}")
    print(f"  n_instance/problem:         {n_instance}")
    print(f"  n_trial/instance:           {n_trial}")
    print(f"  Nodeselectors evaluated:    {','.join(nodesels)}")
    print(f"  Device for GNN inference:   {device}")
    print(f"  Normalize features:         {normalize}")

    for problem in problems:

        #clear records
        for nodesel in nodesels:
            with open(f"nnodes_{problem}_{nodesel}.csv", "w") as f:
                f.write("")
                f.close()
            with open(f"times_{problem}_{nodesel}.csv", "w") as f:
                f.write("")
                f.close()
            if nodesel == "gnn_trained" or nodesel == "gnn_untrained" or nodesel == "random":
                #clear decisions make by oracle
                with open(f"decisions_{nodesel}.csv", "w") as f:
                    f.write("")
                    f.close()


        instances = list(Path(f"./problem_generation/data/{problem}/test").glob("*.lp"))[:n_instance]
        
        for _ in range(n_trial):
            
            if cpu_count == 1:
                record_stats(nodesels, 
                             instances, 
                             problem,
                             verbose=verbose,
                             device=device,
                             normalize=normalize)
                continue
                
            
            chunck_size = int(np.ceil(len(instances)/cpu_count))
            processes = [  md.Process(name=f"worker {p}", 
                                            target=partial(record_stats,
                                                            nodesels=nodesels,
                                                            instances=instances[ p*chunck_size : (p+1)*chunck_size], 
                                                            problem=problem,
                                                            verbose=verbose,
                                                            device=device,
                                                            normalize=normalize))
                            for p in range(cpu_count) ]
            checker = []
            for p in range(cpu_count):
                checker +=instances[ p*chunck_size : (p+1)*chunck_size]
            print(f"Join n_instances parralelixed : {len(checker)}")
            a = list(map(lambda p: p.start(), processes)) #run processes
            b = list(map(lambda p: p.join(), processes)) #join processes
     
        print("==================================")
        print(f"SUMMARIES for {n_trial} trials with {n_instance} instances")
        display_stats(nodesels, problem)






"""
GISP
{'oracle': [(21980, 536.88903),
  (8485, 172.335328),
  (4880, 106.475179),
  (3868, 75.894912),
  (3107, 52.380571),
  (6803, 100.78423)],
 'estimate': [(29442, 481.002816),
  (9990, 154.707946),
  (8920, 131.88883),
  (6870, 82.019173),
  (5031, 62.321967),
  (8453, 117.592189)],
 'dfs': [(25974, 383.803651),
  (16747, 188.050372),
  (7569, 92.087239),
  (7759, 73.404249),
  (6688, 59.423487),
  (9597, 103.478273)]}
"""
   

