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
import numpy as np
import matplotlib.pyplot as plt



def clear_records(problem, nodesels):    
    
    for nodesel in nodesels + ['default']:
        with open(f"nnodes_{problem}_{nodesel}.csv", "w") as f:
            f.write("")
            f.close()
        with open(f"times_{problem}_{nodesel}.csv", "w") as f:
            f.write("")
            f.close()
        with open(f"ncomp_{problem}_{nodesel}.csv", "w") as f:
            f.write("")
            f.close()
            
        if nodesel in ['gnn_trained', 'gnn_untrained']:
            with open(f"decisions_{problem}_{nodesel}.csv", "w") as f:
                f.write("")
                f.close()
                

            
            
            
def setup_oracles(model, optsol, oracles, device):
    for o in oracles:
        try:
            o.setOptsol(optsol)
            o.set_LP_feature_recorder(LPFeatureRecorder(model.getVars(), model.getConss(), device))
        except AttributeError: 
            ''
def put_missing_nodesels(model, nodesels, problem, normalize, device,):
    
    putted = []
    
    for nodesel in nodesels:
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
                                               record_fpath=f"decisions_{problem}_{nodesel}.csv",
                                               use_trained_gnn=trained)
        
        elif re.match('oracle*', nodesel) :
            try:
                inv_proba = float(nodesel.split('_')[-1])
            except:
                inv_proba = 0
            comp = OracleNodeSelectorAbdel('optimal_plunger', optsol=0,inv_proba=inv_proba)
            
            
        if comp != None: #dont include something already included
            putted.append(comp)
            model.includeNodesel(comp, nodesel, 'testing', 100, 100)
        
        
            
    return putted


def activate_nodesel(model, nodesel_to_activate, all_nodesels):
    
    #canceL all nodesels, WORKS
    for nodesel in all_nodesels:
        model.setNodeselPriority(nodesel, 100)
                
    #activate this nodesel, WORKS
    model.setNodeselPriority(nodesel_to_activate, 536870911)
    


#take a list of nodeselectors to evaluate, a list of instance to test on, and the 
#problem type for printing purposes
def record_stats(nodesels, instances, problem, device, normalize, verbose=False, default=True):
    
    if default:  
        default_model = sp.Model()
        default_model.hideOutput()
        default_model.setIntParam('randomization/permutationseed',9) 
        default_model.setIntParam('randomization/randomseedshift',9)
        
    model = sp.Model()
    model.hideOutput()
    model.setIntParam('randomization/permutationseed', 9)
    model.setIntParam('randomization/randomseedshift',9)
    
    putted = put_missing_nodesels(model,nodesels, problem, normalize, device)
    
    for instance in instances:
        
        instance = str(instance)
        if default:
            default_model.readProblem(instance)
        model.readProblem(instance)
        
        #setup oracles
        setup_oracles(model, model.readSolFile(instance.replace(".lp", ".sol")), putted, device)
            
        if verbose:    
            print("----------------------------")
            print(f" {problem}  {instance.split('/')[-1].split('.lp')[0] } ")
       #test nodesels
        for nodesel in nodesels:
            
            model.freeTransform()
            
            activate_nodesel(model, nodesel, nodesels)       

            model.optimize()
            
            with open(f"nnodes_{problem}_{nodesel}.csv", "a+") as f:
                f.write(f"{model.getNNodes()},")
                f.close()
            with open(f"times_{problem}_{nodesel}.csv", "a+") as f:
                f.write(f"{model.getSolvingTime()},")
                f.close()
                
        if default:
            default_model.optimize()
            with open(f"nnodes_{problem}_default.csv", "a+") as f:
                f.write(f"{default_model.getNNodes()},")
                f.close()
            with open(f"times_{problem}_default.csv", "a+") as f:
                f.write(f"{default_model.getSolvingTime()},")
                f.close()
            




def display_stats(nodesels, problem, n_instance=-1, alternative_stdout=None):
    
    print("========================")
    print(f'{problem}') 

    for nodesel in ['default'] + nodesels:
        nnodes = np.genfromtxt(f"nnodes_{problem}_{nodesel}.csv", delimiter=",")[:n_instance]
        times = np.genfromtxt(f"times_{problem}_{nodesel}.csv", delimiter=",")[:n_instance]
    
        print(f"  {nodesel} ")
        print(f"    Number of instances solved    : {len(nnodes)}")
        print(f"    Mean number of node created   : {np.mean(nnodes):.2f}")
        print(f"    Mean solving time             : {np.mean(times):.2f}")
        #print(f"    Median number of node created : {np.median(nnodes):.2f}")
        #print(f"    Median solving time           : {np.median(times):.2f}""
    
    
                
        if nodesel in ['gnn_trained', 'gnn_untrained']:
            decisions = np.genfromtxt(f'decisions_{problem}_{nodesel}.csv', delimiter=',')
            plt.figure()
            plt.title(f'decisions {nodesel}')
            plt.hist(decisions[:,0])
            plt.savefig(f'decisions_{nodesel}.png')
            print(f'    Number of comparaison         : {len(decisions)}')
            print(f'    Accuracy                      : {np.mean( np.round(decisions[:,0]) == 0.5*decisions[:,1]+0.5  ):0.3f}' )
            
    
        print("--------------------------")
     
    