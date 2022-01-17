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



#take a list of nodeselectors to evaluate, a list of instance to test on, and the 
#problem type for printing purposes
def record_stats(nodesels, instances, problem):
    
    nodesels_record = dict((nodesel, []) for nodesel in nodesels)
    model = sp.Model()
    model.hideOutput()
    
    oracle_estimator = None
    oracle = None
    if "oracle_estimator" in nodesels:
        from node_selection.recorders import CompFeaturizer, LPFeatureRecorder
        from node_selection.node_selectors.oracle_selectors import OracleNodeSelectorEstimator
        comp_featurizer = CompFeaturizer()
        oracle_estimator = OracleNodeSelectorEstimator(problem, 
                                                       comp_featurizer,
                                                       DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                                       record_fpath="decisions.csv")
        model.includeNodesel(oracle_estimator, "oracle_estimator", 'testing',100, 100)
    
    if "oracle" in nodesels:
        from node_selection.node_selectors.oracle_selectors import OracleNodeSelectorAbdel
        oracle = OracleNodeSelectorAbdel("optimal_plunger")
        model.includeNodesel(oracle, "oracle", 'testing',100, 100)
        
    for instance in instances:
        
        instance = str(instance)
        model.readProblem(instance)
        
        if "oracle_estimator" in nodesels:
            oracle_estimator.set_LP_feature_recorder(LPFeatureRecorder(model.getVars(),
                                                                       model.getConss()))
        if "oracle" in nodesels:         
            optsol = model.readSolFile(instance.replace(".lp", ".sol"))
            oracle.setOptsol(optsol)
            
        print("----------------------------")
        print(f" {problem}  {instance.split('/')[-1].split('.lp')[0] } ")
       #test nodesels
        for nodesel in nodesels:
            
            model.freeTransform()
            model.readProblem(instance)
            
            #canceL all otheer nodesels, WORKS
            for other in nodesels:
                    model.setNodeselPriority(other, 100)
                    
            #activate this nodesel, WORKS
            model.setNodeselPriority(nodesel, 536870911)
            
            model.optimize()
            print(f"  Nodeselector : {nodesel}")
            print(f"    # of processed nodes : {model.getNNodes()} \n")
            print(f"    Time                 : {model.getSolvingTime()} \n")
            if nodesel == "oracle_estimator":
                print(f"fe time : {oracle_estimator.fe_time}")
                print(f"inference time : {oracle_estimator.inference_time}")
            
                
            with open(f"nnodes_{problem}_{nodesel}.csv", "a+") as f:
                f.write(f"{model.getNNodes()},")
                f.close()
            with open(f"times_{problem}_{nodesel}.csv", "a+") as f:
                f.write(f"{model.getSolvingTime()},")
                f.close()
            

    return nodesels_record, [4]




def display_stats(nodesels, problem):
   
   print("========================")
   print(f'{problem}') 
   for nodesel in nodesels:
        nnodes = np.genfromtxt(f"nnodes_{problem}_{nodesel}.csv", delimiter=",")[:-1]
        times = np.genfromtxt(f"times_{problem}_{nodesel}.csv", delimiter=",")[:-1]
        print(f"  {nodesel} ")
        print(f"    Number of instances solved    : {len(nnodes)}")
        print(f"    Mean number of node created   : {np.mean(nnodes):.2f}")
        print(f"    Mean solving time             : {np.mean(times):.2f}")
        print(f"    Median number of node created : {np.median(nnodes):.2f}")
        print(f"    Median solving time           : {np.median(times):.2f}")
        print("--------------------------")
    from scipy.stats import entropy
    decisions = np.genfromtxt(f"decisions.csv", delimiter=",")[:-1]
    print(f"Entropy of decisions (oracle estimator ) : { entropy(decisions):.3f }") 
   

if __name__ == "__main__":
    DEVICE = 'cpu'
    cpu_count = 2
    problems = ["GISP"]
    nodesels_gpu = ["oracle_estimator"]
    nodesels_cpu = ["oracle", "dfs", "bfs", "estimate"] 
    nodesels = nodesels_gpu + nodesels_cpu
    
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-n_cpu':
            cpu_count = int(sys.argv[i + 1])

    for problem in problems:

        #clear records
        for nodesel in nodesels:
            with open(f"nnodes_{problem}_{nodesel}.csv", "w") as f:
                f.write("")
                f.close()
            with open(f"times_{problem}_{nodesel}.csv", "w") as f:
                f.write("")
                f.close()
                
        #clear decisions make by oracle
        with open(f"decisions.csv", "w") as f:
            f.write("")
            f.close()
        

        instances = list(Path(f"./problem_generation/data/{problem}/test").glob("*.lp"))

        if cpu_count == 1:
            record_stats(nodesels, instances, problem)
        else:
            chunck_size = int(np.ceil(len(instances)/cpu_count))
            processes = [  md.Process(name=f"worker {p}", 
                                            target=partial(record_stats,
                                                            nodesels=nodesels_cpu,
                                                            instances=instances[ p*chunck_size : (p+1)*chunck_size], 
                                                            problem=problem))
                            for p in range(cpu_count) ]


            a = list(map(lambda p: p.start(), processes)) #run processes
            record_stats(nodesels_gpu, instances, problem)
            b = list(map(lambda p: p.join(), processes)) #join processes

        print("SUMMARIES")
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
   

