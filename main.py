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
def record_stats(nodesels, instances, problem, normalize=False, device='cpu', verbose=False):
    
    nodesels_record = dict((nodesel, []) for nodesel in nodesels)
    model = sp.Model()
    model.hideOutput()
    model.setIntParam('randomization/permutationseed', 9)
    model.setIntParam('randomization/randomseedshift',9)
    
    oracle_estimator_trained = None
    oracle_estimator_untrained = None
    oracle_0 = None
    oracle = None
    
    #creating random nodesel
    if "random" in nodesels:
        from node_selection.node_selectors.classic_selectors import Random
        random = Random(record_fpath='decisions_rand.csv')
        model.includeNodesel(random, "random", 'testing',100, 100)
        
    #creating oracle estimator nodesels
    if "gnn_trained" in nodesels:
        from node_selection.recorders import CompFeaturizer, LPFeatureRecorder
        from node_selection.node_selectors.oracle_selectors import OracleNodeSelectorEstimator
        comp_featurizer = CompFeaturizer(normalize=normalize)
        oracle_estimator_trained = OracleNodeSelectorEstimator(problem,
                                                       comp_featurizer,
                                                       DEVICE=device,
                                                       record_fpath="decisions_gnn_trained.csv",
                                                       use_trained_gnn=True)
        model.includeNodesel(oracle_estimator_trained, "gnn_trained", 'testing',100, 100)
    
    if "gnn_untrained" in nodesels:
        from node_selection.recorders import CompFeaturizer, LPFeatureRecorder
        from node_selection.node_selectors.oracle_selectors import OracleNodeSelectorEstimator
        comp_featurizer = CompFeaturizer(normalize=normalize)
        oracle_estimator_untrained = OracleNodeSelectorEstimator(problem,
                                                       comp_featurizer,
                                                       DEVICE=device,
                                                       record_fpath="decisions_gnn_untrained.csv",
                                                       use_trained_gnn=False)
        model.includeNodesel(oracle_estimator_untrained, "gnn_untrained", 'testing',100, 100)
        
    #creating appropriate oracle 
    if 'oracle 0' in nodesels:
        from node_selection.node_selectors.oracle_selectors import OracleNodeSelectorAbdel
        oracle_0 = OracleNodeSelectorAbdel("optimal_plunger", inv_proba=0)
        model.includeNodesel(oracle_0, 'oracle 0', 'testing',100, 100)
        
    def find_2d(matrix, elem):
        for idx, array in enumerate(matrix):
            if elem in array and float(array[1]) > 0:
                return idx, array
        return -1, ""
    
    oracle_idx, oracle_params = find_2d([ n.split(" ") for n in nodesels ], 'oracle')
    if oracle_idx != -1:
        from node_selection.node_selectors.oracle_selectors import OracleNodeSelectorAbdel
        inv_proba = 0 if len(oracle_params) == 1 else float(oracle_params[1]) 
        oracle = OracleNodeSelectorAbdel("optimal_plunger", inv_proba=inv_proba)
        model.includeNodesel(oracle, nodesels[oracle_idx], 'testing',100, 100)
        
    
    
    
    
    for instance in instances:
        
        instance = str(instance)
        model.readProblem(instance)
        optsol = model.readSolFile(instance.replace(".lp", ".sol"))
        
        for idx,o in enumerate([oracle_estimator_trained, oracle_estimator_untrained, oracle_0, oracle]):
            if o != None:
                if idx < 2 : #estimators
                    o.set_LP_feature_recorder(LPFeatureRecorder(model.getVars(), model.getConss()))
                o.setOptsol(optsol)
    
            
        if verbose:    
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
            if verbose:
                print(f"  Nodeselector : {nodesel}")
                print(f"    # of processed nodes : {model.getNNodes()} \n")
                print(f"    Time                 : {model.getSolvingTime()} \n")
            if nodesel == "gnn_trained":
                if verbose:
                    print(f"fe time : {oracle_estimator_trained.fe_time}")
                    print(f"inference time : {oracle_estimator_trained.inference_time}")
            
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
        #print(f"    Number of instances solved    : {len(nnodes)}")
        print(f"    Mean number of node created   : {np.mean(nnodes):.2f}")
        #print(f"    Mean solving time             : {np.mean(times):.2f}")
        #print(f"    Median number of node created : {np.median(nnodes):.2f}")
        #print(f"    Median solving time           : {np.median(times):.2f}")
        print("--------------------------")
   from scipy.stats import entropy
   import matplotlib.pyplot as plt
   
   if 'gnn_trained' in nodesels:
       decisions_gnn_trained = np.genfromtxt("decisions_gnn_trained.csv", delimiter=",")
       plt.figure()
       plt.title("decisions gnn trained")
       plt.hist(decisions_gnn_trained[:,0])
       plt.savefig("decisions_gnn_trained.png")
       print(" % Oracle 1's decision : { np.mean(decisions_gnn_trained[:,1] == 1) : 0.3f}")
       print(f"Error in trained GNN : {1-np.mean( np.round(decisions_gnn_trained[:,0]) == 0.5*decisions_gnn_trained[:,1] + 0.5 ):0.3f}"  )
       
   if 'gnn_untrained' in nodesels:
       decisions_gnn_untrained = np.genfromtxt("decisions_gnn_untrained.csv", delimiter=",")
       plt.figure()
       plt.title("decisions gnn untrained")
       plt.hist(decisions_gnn_untrained[:,0])
       plt.savefig("decisions_gnn_untrained.png")
       print(f" % Oracle 1's decision : { np.mean(decisions_gnn_untrained[:,1] == 1) : 0.3f}")
       print(f"Error in untrained GNN : {1-np.mean( np.round(decisions_gnn_untrained[:,0]) == 0.5*decisions_gnn_untrained[:,1]+0.5  ):0.3f}"  )

     
   if 'random' in nodesels:
       decisions_rand = np.genfromtxt("decisions_rand.csv", delimiter=",")[:-1]
       plt.figure()
       plt.title("decisions rand")
       plt.hist(decisions_rand)
       plt.savefig("decisions_rand.png")
   
if __name__ == "__main__":
    
    cpu_count = 4
    nodesels_cpu = ['random', 'estimate', 'oracle 0', 'oracle 0.05']
    nodesels_gpu = ['gnn_untrained']
    problems = ["GISP"]
    normalize = True
    n_instance = 5
    n_trial = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    verbose = False
    
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-n_cpu':
            cpu_count = int(sys.argv[i + 1])
        if sys.argv[i] == '-nodesels_cpu':
            nodesels_cpu = str(sys.argv[i + 1]).split(',')
        if sys.argv[i] == '-nodesels_gpu':
            nodesels_gpu = str(sys.argv[i + 1]).split(',')
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
            
    nodesels = nodesels_cpu + nodesels_gpu
    
    print(f"Testing {n_instance} instances with {cpu_count} CPUs node selection methods {nodesels} on problems {problems}, ML {'with' if normalize else 'without'} feature normalization on {device}")
    
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
        with open("decisions_gnn_trained.csv", "w") as f:
            f.write("")
            f.close()
            #clear decisions make by oracle
        with open("decisions_gnn_untrained.csv", "w") as f:
            f.write("")
            f.close()
        #clear decisions make by random
        with open("decisions_rand.csv", "w") as f:
            f.write("")
            f.close()
     

        instances = list(Path(f"./problem_generation/data/{problem}/test").glob("*.lp"))[:n_instance]
        
        for _ in range(n_trial):
            
            if cpu_count == 1:
                record_stats(nodesels,
                             instances,
                             problem, 
                             normalize=normalize, 
                             device=device,
                             verbose=verbose)
            else:
        
                chunck_size = int(np.ceil(len(instances)/cpu_count))
                processes = [  md.Process(name=f"worker {p}", 
                                                target=partial(record_stats,
                                                                nodesels=nodesels_cpu,
                                                                instances=instances[ p*chunck_size : (p+1)*chunck_size], 
                                                                problem=problem,
                                                                verbose=verbose))
                                for p in range(cpu_count) ]
                checker = []
                for p in range(cpu_count):
                    checker +=instances[ p*chunck_size : (p+1)*chunck_size]
                print(f"len instances parralelixed : {len(checker)}")
        
        
                a = list(map(lambda p: p.start(), processes)) #run processes
                   
                record_stats(nodesels_gpu,
                             instances,
                             problem, 
                             normalize=normalize, 
                             device=device,
                             verbose=verbose)
                
                b = list(map(lambda p: p.join(), processes)) #join processes
     

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
   

