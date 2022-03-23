#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 12:36:43 2022

@author: aglabassi
"""
import os
import re
import numpy as np
import pyscipopt.scip as sp
from node_selection.recorders import CompFeaturizerSVM, CompFeaturizer, LPFeatureRecorder
from node_selection.node_selectors import (CustomNodeSelector,
                                           OracleNodeSelectorAbdel, 
                                           OracleNodeSelectorEstimator_SVM,
                                           OracleNodeSelectorEstimator)
from learning.utils import normalize_graph

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


def get_nodesels2models(nodesels, instance, problem, normalize, device):
    
    res = dict()
    nodesels2nodeselectors = dict()
    
    for nodesel in nodesels:
        model = sp.Model()
        model.hideOutput()
        model.readProblem(instance)
        model.setIntParam('randomization/permutationseed', 9)
        model.setIntParam('randomization/randomseedshift',9)
        model.setParam('constraints/linear/upgrade/logicor',0)
        model.setParam('constraints/linear/upgrade/indicator',0)
        model.setParam('constraints/linear/upgrade/knapsack', 0)
        model.setParam('constraints/linear/upgrade/setppc', 0)
        model.setParam('constraints/linear/upgrade/xor', 0)
        model.setParam('constraints/linear/upgrade/varbound', 0)
    
        
        comp = None
        
        if re.match('custom_*', nodesel):
            name = nodesel.split("_")[-1]
            comp = CustomNodeSelector(name)
        
        elif nodesel in ['svm']:
            comp_featurizer = CompFeaturizerSVM(model)
            comp = OracleNodeSelectorEstimator_SVM(problem, comp_featurizer)
            
        elif nodesel in ['gnn_trained', 'gnn_untrained']:
            trained = nodesel.split('_')[-1] == "trained"  
            comp_featurizer = CompFeaturizer()
            feature_normalizor = normalize_graph if normalize else lambda x: x
            comp = OracleNodeSelectorEstimator(problem,
                                               comp_featurizer,
                                               device,
                                               feature_normalizor,
                                               use_trained_gnn=trained)
            comp.set_LP_feature_recorder(LPFeatureRecorder(model, device))
        
        elif re.match('oracle*', nodesel) :
            try:
                inv_proba = float(nodesel.split('_')[-1])
            except:
                inv_proba = 0
            comp = OracleNodeSelectorAbdel('optimal_plunger', optsol=0, inv_proba=inv_proba)
            optsol = model.readSolFile(instance.replace(".lp", ".sol"))
            comp.setOptsol(optsol)
            
            
        res[nodesel] = model
        if comp != None: #dont include something already included
            model.includeNodesel(comp, nodesel, 'testing',  536870911,  536870911)
        else:
            model.setNodeselPriority(nodesel,536870911)
            
        
        res[nodesel] = model
        nodesels2nodeselectors[nodesel] = comp
        
        
        
            
    return res, nodesels2nodeselectors



def get_record_file(problem, nodesel, instance):
    save_dir = os.path.join(os.path.abspath(''),  f'stats/{problem}/{nodesel}/')
    
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
        counter = nodesel_obj.counter
    else:
        fe_time, fn_time, inference_time, counter = -1, -1, -1, -1
    
    file = get_record_file(problem, nodesel, instance)
    np.savetxt(file, np.array([nnode, time, fe_time, fn_time, inference_time, counter]), delimiter=',')
    
 

    
def print_infos(problem, nodesel, instance):
    print("------------------------------------------")
    print(f"   |----Solving:  {problem}")
    print(f"   |----Instance: {instance}")
    print(f"   |----Nodesel: {nodesel}")

    

def solve_and_record_default(problem, instance, verbose):
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
    

    for instance in instances:       
        instance = str(instance)
        
        if default and not os.path.isfile(get_record_file(problem,'default', instance)):
            solve_and_record_default(problem, instance, verbose)
        
        nodesels2models, nodesels2nodeselectors = get_nodesels2models(nodesels, instance, problem, normalize, device)
        
        
        for nodesel in nodesels:  
            
            model = nodesels2models[nodesel]
            nodeselector = nodesels2nodeselectors[nodesel]
                
           #test nodesels
            if os.path.isfile(get_record_file(problem, nodesel, instance)): #no need to resolve 
                continue
        
            
            if verbose:
                print_infos(problem, nodesel, instance)
                
            model.optimize()
            record_stats_instance(problem, nodesel, model, instance, nodesel_obj=nodeselector)
    
 
               



def get_mean(problem, nodesel, instances, stat_type):
    res = 0
    n = 0
    stat_idx = ['nnode', 'time', 'fe', 'fn', 'inf', 'ncomp'].index(stat_type)
    for instance in instances:
        try:
            file = get_record_file(problem, nodesel, instance)
            res += np.genfromtxt(file)[stat_idx]
            n += 1
        except:
            ''
    return res/(n + int(n==0)),n
        

def display_stats(problem, nodesels, instances, min_n, max_n, default=False):
    
    print("======================================================")
    print(f'Statistics on {problem} over {len(instances)} instances for problem size in [{min_n}, {max_n}]') 
    print("======================================================")
    
    for nodesel in nodesels + (['default'] if default else []):
            
        nnode_mean, n = get_mean(problem, nodesel, instances, 'nnode')
        time_mean, n =  get_mean(problem, nodesel, instances, 'time')
        
    
        print(f"  {nodesel}, n={n} ")
        print(f"    Mean NNode Created   : {nnode_mean:.2f}")
        print(f"    Mean Solving Time    : {time_mean:.2f}")
        #print(f"    Median number of node created : {np.median(nnodes):.2f}")
        #print(f"    Median solving time           : {np.median(times):.2f}""
    
    
                
        if re.match('gnn*', nodesel):
            fe_mean = get_mean(problem, nodesel, instances, 'fe')[0]
            fn_mean = get_mean(problem, nodesel, instances, 'fn')[0]
            inf_mean = get_mean(problem, nodesel, instances, 'inf')[0]
            ncomp = get_mean(problem, nodesel, instances, 'ncomp')[0]
            print(f"     |---   Feature Extraction  Mean Time:      {fe_mean:.2f}")
            print(f"     |---   Feature Normalization Mean Time:    {fn_mean:.2f}")
            print(f"     |---   Inference Mean Time:                {inf_mean:.2f}")
            print(f"     |---   Ncomp Mean:                         {ncomp:.2f}")
            
        print("-------------------------------------------------")
            
        
     
     
    
