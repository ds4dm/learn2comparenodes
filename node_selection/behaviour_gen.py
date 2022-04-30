#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:26:18 2021

@author: abdel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 12:54:57 2021

@author: abdel

"""

import os
import sys
import random
import numpy as np
import pyscipopt.scip as sp
from pathlib import Path 
from functools import partial
from node_selectors import OracleNodeSelectorAbdel
from recorders import LPFeatureRecorder, CompFeaturizer, CompFeaturizerSVM
from torch.multiprocessing import Process, set_start_method




class OracleNodeSelRecorder(OracleNodeSelectorAbdel):
    
    def __init__(self, oracle_type, comp_behaviour_saver, comp_behaviour_saver_svm):
        super().__init__(oracle_type)
        self.counter = 0
        self.comp_behaviour_saver = comp_behaviour_saver
        self.comp_behaviour_saver_svm = comp_behaviour_saver_svm
    
    def set_LP_feature_recorder(self, LP_feature_recorder):
        self.comp_behaviour_saver.set_LP_feature_recorder(LP_feature_recorder)

        
        
    def nodecomp(self, node1, node2):
        comp_res, comp_type = super().nodecomp(node1, node2, return_type=True)
        
        if comp_type in [-1,1]:
            self.comp_behaviour_saver.save_comp(self.model, 
                                                node1, 
                                                node2,
                                                comp_res,
                                                self.counter) 
            
            self.comp_behaviour_saver_svm.save_comp(self.model, 
                                                node1, 
                                                node2,
                                                comp_res,
                                                self.counter) 
        
            #print("saved comp # " + str(self.counter))
            self.counter += 1
        
        #make it bad to generate more data !
        if comp_type in [-1,1]:
            comp_res = -1 if comp_res == 1 else 1
        else:
            comp_res = 0
            
        return comp_res



def run_episode(oracle_type, instance,  save_dir, save_dir_svm, device):
    
    model = sp.Model()
    model.hideOutput()
    
    
    #Setting up oracle selector
    instance = str(instance)
    model.readProblem(instance)
    model.setParam('constraints/linear/upgrade/logicor',0)
    model.setParam('constraints/linear/upgrade/indicator',0)
    model.setParam('constraints/linear/upgrade/knapsack', 0)
    model.setParam('constraints/linear/upgrade/setppc', 0)
    model.setParam('constraints/linear/upgrade/xor', 0)
    model.setParam('constraints/linear/upgrade/varbound', 0)
    
    
    optsol = model.readSolFile(instance.replace(".lp", ".sol"))
    
    comp_behaviour_saver = CompFeaturizer(f"{save_dir}", instance_name=str(instance).split("/")[-1])
    comp_behaviour_saver_svm = CompFeaturizerSVM(model, f"{save_dir_svm}", instance_name=str(instance).split("/")[-1])
    
    oracle_ns = OracleNodeSelRecorder(oracle_type, comp_behaviour_saver, comp_behaviour_saver_svm)
    oracle_ns.setOptsol(optsol)
    oracle_ns.set_LP_feature_recorder(LPFeatureRecorder(model, device))
        
    
    model.includeNodesel(oracle_ns, "oracle_recorder", "testing",
                         536870911,  536870911)


    # Run the optimizer
    model.optimize()
    print(f"Got behaviour for instance  "+ str(instance).split("/")[-1] + f' with {oracle_ns.counter} comparisons' )
    
    with open("nnodes.csv", "a+") as f:
        f.write(f"{model.getNNodes()},")
        f.close()
    with open("times.csv", "a+") as f:
        f.write(f"{model.getSolvingTime()},")
        f.close()
        
    return 1


def run_episodes(oracle_type, instances, save_dir, save_dir_svm, device):
    
    for instance in instances:
        run_episode(oracle_type, instance, save_dir, save_dir_svm, device)
        
    print("finished running episodes for process")
        
    return 1
    
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


if __name__ == "__main__":
    

    
    oracle = 'optimal_plunger'
    problem = 'GISP'
    data_partitions = ['train', 'valid'] #dont change
    n_cpu = 10
    n_instance = -1
    device = 'cpu'
    
    with open("nnodes.csv", "w") as f:
        f.write("")
        f.close()
    with open("times.csv", "w") as f:
        f.write("")
        f.close()
        
    
    #Initializing the model 
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-oracle':
            oracle = str(sys.argv[i + 1])
        if sys.argv[i] == '-problem':
            problem = str(sys.argv[i + 1])
        if sys.argv[i] == '-n_cpu':
            n_cpu = int(sys.argv[i + 1])
        if sys.argv[i] == '-n_instance':
            n_instance = int(sys.argv[i + 1])
        if sys.argv[i] == '-device':
            device = str(sys.argv[i + 1])
   
   
  
    for data_partition in data_partitions:
        

        save_dir = os.path.join(os.path.dirname(__file__), f'./data/{problem}/{data_partition}')
        save_dir_svm = os.path.join(os.path.dirname(__file__), f'./data_svm/{problem}/{data_partition}')
        
        try:
            os.makedirs(save_dir)
        except FileExistsError:
            ""
        
        n_keep  = n_instance if data_partition == 'train' or n_instance == -1 else int(0.2*n_instance)
        
        instances = list(Path(os.path.join(os.path.dirname(__file__), 
                                           f"../problem_generation/data/{problem}/{data_partition}")).glob("*.lp"))
        random.shuffle(instances)
        instances = instances[:n_keep]
        
        print(f"Generating {data_partition} samples from {len(instances)} instances using oracle {oracle}")
        
      
        processes = [  Process(name=f"worker {p}", 
                                        target=partial(run_episodes,
                                                        oracle_type=oracle,
                                                        instances=instances[ p1 : p2], 
                                                        save_dir=save_dir,
                                                        save_dir_svm=save_dir_svm,
                                                        device=device))
                        for p,(p1,p2) in enumerate(distribute(len(instances), n_cpu))]
        
        
        try:
            set_start_method('spawn')
        except RuntimeError:
            ''
            
        a = list(map(lambda p: p.start(), processes)) #run processes
        b = list(map(lambda p: p.join(), processes)) #join processes
        
            
    nnodes = np.genfromtxt("nnodes.csv", delimiter=",")[:-1]
    times = np.genfromtxt("times.csv", delimiter=",")[:-1]
        
    print(f"Mean number of node created  {np.mean(nnodes)}")
    print(f"Mean solving time  {np.mean(times)}")
    print(f"Median number of node created  {np.median(nnodes)}")
    print(f"Median solving time  {np.median(times)}")
    
    
                         
            
        

        
