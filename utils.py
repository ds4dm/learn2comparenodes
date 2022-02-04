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
import os

def setup_oracles(model, optsol, name2nodeselector, device):
    
    for nodesel in name2nodeselector:
        if re.match('oracle*', nodesel):
            name2nodeselector[nodesel].setOptsol(optsol)
            
        elif re.match('gnn*', nodesel):
            name2nodeselector[nodesel].set_LP_feature_recorder(LPFeatureRecorder(model.getVars(), model.getConss(), device))
            
def put_missing_nodesels(model, nodesels, problem, normalize, device,):
    
    putted = dict()
    
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
            
            
        putted[nodesel] = comp 
        if comp != None: #dont include something already included
            model.includeNodesel(comp, nodesel, 'testing', 100, 100)
        
        
            
    return putted


def activate_nodesel(model, nodesel_to_activate, all_nodesels):
    
    #canceL all nodesels, WORKS
    for nodesel in all_nodesels:
        model.setNodeselPriority(nodesel, 100)
                
    #activate this nodesel, WORKS
    model.setNodeselPriority(nodesel_to_activate, 536870911)
    

def get_record_file(problem, nodesel, instance):
    save_dir = os.path.join(os.path.dirname(__file__),  f'stats/{problem}/{nodesel}/')
    
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        ""
        
    instance = str(instance).split('/')[-1]
    file = os.path.join(save_dir, instance.replace('.lp', '.csv'))
    return file

def record_stats_instance(problem, nodesel, model, instance, nodesel_obj=None):
    nnode = model.getNNodes()
    time = model.getSolvingTime()
    
    if re.match('gnn*', nodesel):
        fe_time = nodesel_obj.fe_time
        fn_time = nodesel_obj.fn_time
        inference_time = nodesel_obj.inference_time
    else:
        fe_time, fn_time, inference_time = -1, -1, -1
    
    file = get_record_file(problem, nodesel, instance)
    np.savetxt(file, np.array([nnode, time, fe_time, fn_time, inference_time]), delimiter=',')
    
 

    
def print_infos(problem, nodesel, instance):
    print("------------------------------------------")
    print(f"   |----Solving:  {problem}")
    print(f"   |----Instance: {instance}")
    print(f"   |----Nodesel: {nodesel}")

    

def solve_default(problem, instance, verbose):
    default_model = sp.Model()
    default_model.hideOutput()
    default_model.setIntParam('randomization/permutationseed',9) 
    default_model.setIntParam('randomization/randomseedshift',9)
    default_model.readProblem(instance)
    if verbose:
        print_infos(problem, 'default', instance)
    
    default_model.optimize()        
    record_stats_instance(problem, 'default', default_model, instance, nodesel_obj=None)

    


#take a list of nodeselectors to evaluate, a list of instance to test on, and the 
#problem type for printing purposes
def record_stats(nodesels, instances, problem, device, normalize, verbose=False, default=True):
        
    model = sp.Model()
    model.hideOutput()
    model.setIntParam('randomization/permutationseed', 9)
    model.setIntParam('randomization/randomseedshift',9)
    
    name2nodeselector = put_missing_nodesels(model,nodesels, problem, normalize, device)
    
    for instance in instances:
        
        instance = str(instance)
        
        model.readProblem(instance)
        
        #setup oracles
        setup_oracles(model, model.readSolFile(instance.replace(".lp", ".sol")), name2nodeselector, device)
            
       #test nodesels
        for nodesel in nodesels:
            if os.path.isfile(get_record_file(problem, nodesel, instance)): #no need to resolve 
                continue
            
            model.freeTransform()
            activate_nodesel(model, nodesel, nodesels)     
            
            if verbose:
                print_infos(problem, nodesel, instance)
            model.optimize()
            record_stats_instance(problem, nodesel, model, instance, nodesel_obj=name2nodeselector[nodesel])

        if default and not os.path.isfile(get_record_file(problem,'default', instance)):
            solve_default(problem, instance, verbose)
           



def get_mean(problem, nodesel, instances, stat_type):
    res = 0
    stat_idx = ['nnode', 'time', 'fe', 'fn', 'inf'].index(stat_type)
    for instance in instances:
        file = get_record_file(problem, nodesel, instance)
        res += np.genfromtxt(file)[stat_idx]
        
    return res/(len(instances) + (len(instances)==0))
        

def display_stats(problem, nodesels, instances, min_n, max_n, default=False):
    
    print("======================================================")
    print(f'Statistics on {problem} over {len(instances)} instances for n in [{min_n}, {max_n}]') 
    print("======================================================")
    
    for nodesel in (['default'] if default else []) + nodesels:
        
        nnode_mean = get_mean(problem, nodesel, instances, 'nnode')
        time_mean =  get_mean(problem, nodesel, instances, 'time')
        
    
        print(f"  {nodesel} ")
        print(f"    Mean NNode Created   : {nnode_mean:.2f}")
        print(f"    Mean Solving Time    : {time_mean:.2f}")
        #print(f"    Median number of node created : {np.median(nnodes):.2f}")
        #print(f"    Median solving time           : {np.median(times):.2f}""
    
    
                
        if re.match('gnn*', nodesel):
            fe_mean = get_mean(problem, nodesel, instances, 'fe')
            fn_mean = get_mean(problem, nodesel, instances, 'fn')
            inf_mean = get_mean(problem, nodesel, instances, 'inf')
            print(f"     |---   Feature Extraction  Mean Time:      {fe_mean:.2f}")
            print(f"     |---   Feature Normalization Mean Time:    {fn_mean:.2f}")
            print(f"     |---   Inference Mean Time:                {inf_mean:.2f}")
            
        print("-------------------------------------------------")
            
        
     
    