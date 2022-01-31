#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 12:36:43 2022

@author: aglabassi
"""

import pyscipopt.scip as sp
import sys
from node_selection.recorders import CompFeaturizer, LPFeatureRecorder
from node_selection.node_selectors import CustomNodeSelector,OracleNodeSelectorEstimator, OracleNodeSelectorAbdel
from learning.train import normalize_graph
import re


def setup_oracles(model, optsol, oracles, device):
    for o in oracles:
        o.setOptsol(optsol)
        try:
            o.set_LP_feature_recorder(LPFeatureRecorder(model.getVars(), model.getConss(), device))
        except AttributeError: #oracle Abdel has no lp feature recorder
            ''
        


#take a list of nodeselectors to evaluate, a list of instance to test on, and the 
#problem type for printing purposes
def record_stats(nodesels, instances, problem, device, normalize, verbose=False):
    
    nodesels_record = dict((nodesel, []) for nodesel in nodesels)
    model = sp.Model()
    model.hideOutput()
    model.setIntParam('randomization/permutationseed', 9)
    model.setIntParam('randomization/randomseedshift',9)
    
    oracles = []
    
    for nodesel in nodesels :
        comp = None
        if re.match('custom_*', nodesel):
            name = nodesel.split("_")[-1]
            comp = CustomNodeSelector(name)
            
        elif nodesel in ['gnn_trained', 'gnn_untrained']:
            trained = nodesel.split('_')[-1] == "trained"            
            comp_featurizer = CompFeaturizer()
            feature_normalizor = normalize_graph if normalize else lambda x: x
            comp = OracleNodeSelectorEstimator(problem,
                                               comp_featurizer,
                                               device,
                                               feature_normalizor,
                                               use_trained_gnn=trained)
            oracles.append(comp)
        
        elif re.match('oracle*', nodesel) :
            try:
                inv_proba = float(nodesel.split('_')[-1])
            except:
                inv_proba = 0
            comp = OracleNodeSelectorAbdel('optimal_plunger', optsol=0,inv_proba=inv_proba)
            oracles.append(comp)
        
        if comp != None:
            model.includeNodesel(comp, nodesel, 'testing', 100, 100)
    
    
    for instance in instances:
        
        instance = str(instance)
        model.readProblem(instance)
        optsol = model.readSolFile(instance.replace(".lp", ".sol"))
        
        
        #setup oracles
        setup_oracles(model, optsol, oracles, device)
            
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
            
            with open(f"nnodes_{problem}_{nodesel}.csv", "a+") as f:
                f.write(f"{model.getNNodes()},")
                f.close()
            with open(f"times_{problem}_{nodesel}.csv", "a+") as f:
                f.write(f"{model.getSolvingTime()},")
                f.close()
            

    return nodesels_record, [4]




def display_stats(nodesels, problem, n_instance=-1, alternative_stdout=None):
    import matplotlib.pyplot as plt
    import numpy as np
        
    print("========================")
    print(f'{problem}') 
    for nodesel in nodesels:
         nnodes = np.genfromtxt(f"nnodes_{problem}_{nodesel}.csv", delimiter=",")[:n_instance]
         times = np.genfromtxt(f"times_{problem}_{nodesel}.csv", delimiter=",")[:n_instance]

         print(f"  {nodesel} ")
         print(f"    Number of instances solved    : {len(nnodes)}")
         print(f"    Mean number of node created   : {np.mean(nnodes):.2f}")
         print(f"    Mean solving time             : {np.mean(times):.2f}")
         #print(f"    Median number of node created : {np.median(nnodes):.2f}")
         #print(f"    Median solving time           : {np.median(times):.2f}")

    
    if alternative_stdout != None:
        sys.stdout = alternative_stdout
        display_stats(nodesels, problem, alternative_stdout=None)
        
