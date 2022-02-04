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
import numpy as np
import pyscipopt.scip as sp
import multiprocessing as md
from pathlib import Path 
from functools import partial
from node_selectors import OracleNodeSelectorAbdel
from recorders import LPFeatureRecorder, CompFeaturizer


class OracleNodeSelRecorder(OracleNodeSelectorAbdel):
    
    def __init__(self, oracle_type, comp_behaviour_saver=None):
        super().__init__(oracle_type)
        self.counter = 0
        self.comp_behaviour_saver = comp_behaviour_saver
    
    def set_LP_feature_recorder(self, LP_feature_recorder):
        self.comp_behaviour_saver.set_LP_feature_recorder(LP_feature_recorder)
        self.counter = 0
        
        
    def nodecomp(self, node1, node2):
        comp_res, comp_type = super().nodecomp(node1, node2, return_type=True)
        
        if comp_type in [-1,1]:
            self.comp_behaviour_saver.save_comp(self.model, 
                                                node1, 
                                                node2,
                                                comp_res,
                                                self.counter) 
        
            print("saved comp # " + str(self.counter))
            self.counter += 1
        
        #make it shit to generate more data !
        if comp_type in [-1,1]:
            comp_res = -1 if comp_res == 1 else 1
        else:
            comp_res = 0
            
        return comp_res



def run_episode(oracle_type, instance,  save_dir):
    
    model = sp.Model()
    model.hideOutput()
    
    #Setting up oracle selector
    instance = str(instance)
    model.readProblem(instance)
    
    optsol = model.readSolFile(instance.replace(".lp", ".sol"))
    comp_behaviour_saver = CompFeaturizer(f"{save_dir}", 
                                              instance_name=str(instance).split("/")[-1])
    oracle_ns = OracleNodeSelRecorder(oracle_type, comp_behaviour_saver)
    oracle_ns.setOptsol(optsol)
    oracle_ns.set_LP_feature_recorder(LPFeatureRecorder(model.getVars(),
                                                        model.getConss(),
                                                        'cpu'))
    
    model.includeNodesel(oracle_ns, "oracle_recorder", "testing",
                         536870911,  536870911)

    #print(f"Getting behaviour for instance {problem} "+ str(instance).split("/")[-1] )

    # Run the optimizer
    model.freeTransform()
    model.readProblem(instance)
    model.optimize()
    with open("nnodes.csv", "a+") as f:
        f.write(f"{model.getNNodes()},")
        f.close()
    with open("times.csv", "a+") as f:
        f.write(f"{model.getSolvingTime()},")
        f.close()
        
    return 1


def run_episodes(oracle_type, instances, save_dir):
    
    for instance in instances:
        run_episode(oracle_type, instance, save_dir)
        
    print("finished running episodes for process " + str(md.current_process()))
        
    return 1
    

if __name__ == "__main__":
    

    
    oracle = 'optimal_plunger'
    problem = 'GISP'
    data_partitions = ['train', 'valid']
    n_cpu = 1
    
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
   
  
    for data_partition in data_partitions:
        
        save_dir = lp_dir= os.path.join(os.path.dirname(__file__), f"./data/{problem}/{data_partition}")
    
        try:
            os.makedirs(save_dir)
        except FileExistsError:
            ""
        
        
        instances = list(Path(os.path.join(os.path.dirname(__file__), 
                                           f"../problem_generation/data/{problem}/{data_partition}")).glob("*.lp"))
        
        print(f"Geneating {data_partition} samples from {len(instances)} instances using oracle {oracle}")
        
        chunck_size = int(np.floor(len(instances)/n_cpu))
        processes = [  md.Process(name=f"worker {p}", 
                                        target=partial(run_episodes,
                                                        oracle_type=oracle,
                                                        instances=instances[ p*chunck_size : (p+1)*chunck_size], 
                                                        save_dir=save_dir))
                        for p in range(n_cpu+1) ]
            
        a = list(map(lambda p: p.start(), processes)) #run processes
        b = list(map(lambda p: p.join(), processes)) #join processes
        
            
    nnodes = np.genfromtxt("nnodes.csv", delimiter=",")[:-1]
    times = np.genfromtxt("times.csv", delimiter=",")[:-1]
        
    print(f"Mean number of node created  {np.mean(nnodes)}")
    print(f"Mean solving time  {np.mean(times)}")
    print(f"Median number of node created  {np.median(nnodes)}")
    print(f"Median solving time  {np.median(times)}")
    
    
                         
            
        

        
