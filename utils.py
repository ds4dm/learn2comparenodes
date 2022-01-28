#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 12:36:43 2022

@author: aglabassi
"""

import pyscipopt.scip as sp
import sys

#take a list of nodeselectors to evaluate, a list of instance to test on, and the 
#problem type for printing purposes
def record_stats(nodesels, instances, problem, normalize=True, device='cpu', verbose=False):
    
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
    if "estimate_custom" in nodesels:
        from node_selection.node_selectors.classic_selectors import EstimateComp
        comp = EstimateComp()
        model.includeNodesel(comp, "estimate_custom", 'testing',100, 100)
        
    #creating oracle estimator nodesels
    if "gnn_trained" in nodesels:
        from node_selection.recorders import CompFeaturizer, LPFeatureRecorder
        from node_selection.node_selectors.oracle_selectors import OracleNodeSelectorEstimator
        from learning.train import normalize_graph
        
        comp_featurizer = CompFeaturizer(normalizor=normalize_graph if normalize else None)
        oracle_estimator_trained = OracleNodeSelectorEstimator(problem,
                                                       comp_featurizer,
                                                       DEVICE=device,
                                                       record_fpath="decisions_gnn_trained.csv",
                                                       use_trained_gnn=True)
        
        model.includeNodesel(oracle_estimator_trained, "gnn_trained", 'testing',100, 100)
    
    if "gnn_untrained" in nodesels:
        from node_selection.recorders import CompFeaturizer, LPFeatureRecorder
        from node_selection.node_selectors.oracle_selectors import OracleNodeSelectorEstimator
        from learning.train import normalize_graph
        
        comp_featurizer = CompFeaturizer(normalizor=normalize_graph if normalize else None)
        oracle_estimator_untrained = OracleNodeSelectorEstimator(problem,
                                                       comp_featurizer,
                                                       DEVICE=device,
                                                       record_fpath="decisions_gnn_untrained.csv",
                                                       use_trained_gnn=False)
        
        model.includeNodesel(oracle_estimator_untrained, "gnn_untrained", 'testing',100, 100)
        
    #creating appropriate oracle 
    if 'oracle_0' in nodesels:
        from node_selection.node_selectors.oracle_selectors import OracleNodeSelectorAbdel
        oracle_0 = OracleNodeSelectorAbdel("optimal_plunger", inv_proba=0)
        model.includeNodesel(oracle_0, 'oracle_0', 'testing',100, 100)
        
    def find_2d(matrix, elem):
        for idx, array in enumerate(matrix):
            if elem in array and float(array[1]) > 0:
                return idx, array
        return -1, ""
    
    oracle_idx, oracle_params = find_2d([ n.split("_") for n in nodesels ], 'oracle')
    if oracle_idx != -1:
        from node_selection.node_selectors.oracle_selectors import OracleNodeSelectorAbdel
        inv_proba = 0 if len(oracle_params) == 1 else float(oracle_params[1]) 
        oracle = OracleNodeSelectorAbdel("optimal_plunger", inv_proba=inv_proba)
        model.includeNodesel(oracle, nodesels[oracle_idx], 'testing',100, 100)
        
    
    
    
    
    for instance in instances:
        
        instance = str(instance)
        model.readProblem(instance)
        optsol = model.readSolFile(instance.replace(".lp", ".sol"))
        
        
        #setup oracles
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
            print("------------------------")
            
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




def display_stats(nodesels, problem, alternative_stdout=None):
    from scipy.stats import entropy
    import matplotlib.pyplot as plt
    import numpy as np
        
    print("========================")
    print(f'{problem}') 
    for nodesel in nodesels:
         nnodes = np.genfromtxt(f"nnodes_{problem}_{nodesel}.csv", delimiter=",")[:-1]
         times = np.genfromtxt(f"times_{problem}_{nodesel}.csv", delimiter=",")[:-1]
         print(f"  {nodesel} ")
         print(f"    Number of instances solved    : {len(nnodes)}")
         print(f"    Mean number of node created   : {np.mean(nnodes):.2f}")
         print(f"    Mean solving time             : {np.mean(times):.2f}")
         #print(f"    Median number of node created : {np.median(nnodes):.2f}")
         #print(f"    Median solving time           : {np.median(times):.2f}")
         print("--------------------------")
         
    if 'gnn_trained' in nodesels:
        decisions_gnn_trained = np.genfromtxt("decisions_gnn_trained.csv", delimiter=",")
        plt.figure()
        plt.title("decisions gnn trained")
        plt.hist(decisions_gnn_trained[:,0])
        plt.savefig("decisions_gnn_trained.png")
        print(f"Accuracy in trained GNN : {np.mean( np.round(decisions_gnn_trained[:,0]) == 0.5*decisions_gnn_trained[:,1] + 0.5 ):0.3f}"  )
        
    if 'gnn_untrained' in nodesels:
        decisions_gnn_untrained = np.genfromtxt("decisions_gnn_untrained.csv", delimiter=",")
        plt.figure()
        plt.title("decisions gnn untrained")
        plt.hist(decisions_gnn_untrained[:,0])
        plt.savefig("decisions_gnn_untrained.png")
        print(f"Accuracy in untrained GNN : {np.mean( np.round(decisions_gnn_untrained[:,0]) == 0.5*decisions_gnn_untrained[:,1]+0.5  ):0.3f}"  )
    
      
    if 'random' in nodesels:
        decisions_rand = np.genfromtxt("decisions_rand.csv", delimiter=",")[:-1]
        plt.figure()
        plt.title("decisions rand")
        plt.hist(decisions_rand)
        plt.savefig("decisions_rand.png")
    
    
    if alternative_stdout != None:
        sys.stdout = alternative_stdout
        display_stats(nodesels, problem, alternative_stdout=None)
        