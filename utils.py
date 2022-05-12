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
                                           OracleNodeSelectorEstimator_RankNet,
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
        
        if not re.match('default*', nodesel):
            try:
                comp_policy, sel_policy, other = nodesel.split("_")
            except:
                comp_policy, sel_policy = nodesel.split("_")
                


            if comp_policy == 'gnn':
                comp_featurizer = CompFeaturizer()
                
                feature_normalizor = normalize_graph if normalize else lambda x: x
                
                n_primal = int(other.split('=')[-1])
                       
                
                comp = OracleNodeSelectorEstimator(problem,
                                                   comp_featurizer,
                                                   device,
                                                   feature_normalizor,
                                                   use_trained_gnn=True,
                                                   sel_policy=sel_policy,
                                                   n_primal=n_primal)
                fr = LPFeatureRecorder(model, device)
                comp.set_LP_feature_recorder(fr)

            elif comp_policy == 'svm':
                comp_featurizer = CompFeaturizerSVM(model)
                n_primal = int(other.split('=')[-1])
                comp = OracleNodeSelectorEstimator_SVM(problem, comp_featurizer, sel_policy=sel_policy, n_primal=n_primal)
                
            elif comp_policy == 'ranknet':
                comp_featurizer = CompFeaturizerSVM(model)
                n_primal = int(other.split('=')[-1])
                comp = OracleNodeSelectorEstimator_RankNet(problem, comp_featurizer, device, sel_policy=sel_policy, n_primal=n_primal)
        

            elif comp_policy == 'expert':
                comp = OracleNodeSelectorAbdel('optimal_plunger', optsol=0,inv_proba=0)
                optsol = model.readSolFile(instance.replace(".lp", ".sol"))
                comp.setOptsol(optsol)

            else:
                comp = CustomNodeSelector(comp_policy=comp_policy, sel_policy=sel_policy)

            model.includeNodesel(comp, nodesel, 'testing',  536870911,  536870911)
        
        else:
            _, nsel_name, priority = nodesel.split("_")
            assert(nsel_name in ['estimate', 'dfs', 'bfs']) #to do add other default methods 
            priority = int(priority)
            model.setNodeselPriority(nsel_name, priority)
            

            
        
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

def record_stats_instance(problem, nodesel, model, instance, nodesel_obj):
    nnode = model.getNNodes()
    time = model.getSolvingTime()
    
    if nodesel_obj != None:    
        comp_counter = nodesel_obj.comp_counter
        sel_counter = nodesel_obj.sel_counter
    else:
        comp_counter = sel_counter = -1
    
    
    if re.match('gnn*', nodesel):
        init1_time = nodesel_obj.init_solver_cpu
        init2_time = nodesel_obj.init_cpu_gpu
        fe_time = nodesel_obj.fe_time
        fn_time = nodesel_obj.fn_time
        inference_time = nodesel_obj.inference_time
        inf_counter = nodesel_obj.inf_counter
        
    else:
        init1_time, init2_time, fe_time, fn_time, inference_time, inf_counter = -1, -1, -1, -1, -1, -1
    
    
    if re.match('svm*', nodesel) or re.match('expert*', nodesel) or re.match('ranknet*', nodesel):
        inf_counter = nodesel_obj.inf_counter
    
    
        
    
    file = get_record_file(problem, nodesel, instance)
    np.savetxt(file, np.array([nnode, time, comp_counter, sel_counter, init1_time, init2_time, fe_time, fn_time, inference_time, inf_counter]), delimiter=',')
    
 

    
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
    record_stats_instance(problem, 'default', default_model, instance, None)

    


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
            record_stats_instance(problem, nodesel, model, instance, nodeselector)
    
 
               



def get_mean(problem, nodesel, instances, stat_type):
    res = 0
    n = 0
    means = dict()
    stat_idx = ['nnode', 'time', 'ncomp','nsel', 'init1', 'init2', 'fe', 'fn', 'inf','ninf'].index(stat_type)
    for instance in instances:
        try:
            file = get_record_file(problem, nodesel, instance)
            res += np.genfromtxt(file)[stat_idx]
            n += 1
            means[str(instance)] = np.genfromtxt(file)[stat_idx]
        except:
            ''
    return res/(n + int(n==0)),n, means
        

def display_stats(problem, nodesels, instances, min_n, max_n, default=False):
    
    print("======================================================")
    print(f'Statistics on {problem} for problem size in [{min_n}, {max_n}]') 
    print("======================================================")
    means_nodes = dict()
    for nodesel in (['default'] if default else []) + nodesels:
        
            
        nnode_mean, n, nnode_means = get_mean(problem, nodesel, instances, 'nnode')
        time_mean  =  get_mean(problem, nodesel, instances, 'time')[0]
        ncomp_mean = get_mean(problem, nodesel, instances, 'ncomp')[0]
        nsel_mean = get_mean(problem, nodesel, instances, 'nsel')[0]
        
        
        means_nodes[nodesel] = nnode_means
        
    
        print(f"  {nodesel} ")
        print(f"      Mean over n={n} instances : ")
        print(f"        |- B&B Tree Size   :  {nnode_mean:.0f}")
        if re.match('gnn*', nodesel):
            in1_mean = get_mean(problem, nodesel, instances, 'init1')[0]
            in2_mean = get_mean(problem, nodesel, instances, 'init2')[0]
            print(f"        |- Presolving A,b,c Feature Extraction Time :  ")
            print(f"           |---   Init. Solver to CPU:           {in1_mean:.2f}")
            print(f"           |---   Init. CPU to GPU   :           {in2_mean:.2f}")
        print(f"        |- Solving Time    :  {time_mean:.2f}")
        
        #print(f"    Median number of node created : {np.median(nnodes):.2f}")
        #print(f"    Median solving time           : {np.median(times):.2f}""
    
    
                
        if re.match('gnn*', nodesel):
            fe_mean = get_mean(problem, nodesel, instances, 'fe')[0]
            fn_mean = get_mean(problem, nodesel, instances, 'fn')[0]
            inf_mean = get_mean(problem, nodesel, instances, 'inf')[0]
            print(f"           |---   On-GPU Feature Updates:        {fe_mean:.2f}")
            print(f"           |---   Feature Normalization:         {fn_mean:.2f}")
            print(f"           |---   Inference     :                {inf_mean:.2f}")
            
        if not re.match('default*', nodesel):
            print(f"        |- nodecomp calls  :  {ncomp_mean:.0f}")
            if re.match('gnn*', nodesel) or re.match('svm*', nodesel) or re.match('expert*', nodesel) or re.match('ranknet*', nodesel):
                inf_counter_mean = get_mean(problem, nodesel, instances, 'ninf')[0]
                print(f"           |---   inference nodecomp calls:      {inf_counter_mean:.0f}")
            print(f"        |- nodesel calls   :  {nsel_mean:.0f}")
        print("-------------------------------------------------")
        
    return means_nodes
     
     
    
