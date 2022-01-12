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
from node_selectors.oracle_selectors import OracleNodeSelectorAbdel
from recorders import LPFeatureRecorder, CompFeaturizer
from pathlib import Path 
import pyscipopt.scip as sp
import numpy as np
import multiprocessing as md
from functools import partial


class OracleNodeSelRecorder(OracleNodeSelectorAbdel):
    
    def __init__(self, oracle_type, comp_behaviour_saver=None):
        super().__init__(oracle_type)
        self.counter = 0
        self.comp_behaviour_saver = comp_behaviour_saver
    
    def set_LP_feature_recorder(self, LP_feature_recorder):
        self.comp_behaviour_saver.set_LP_feature_recorder(LP_feature_recorder)
        
        
    def nodecomp(self, node1, node2):
        comp_res = super().nodecomp(node1, node2)
        
        self.comp_behaviour_saver.save_comp(self.model, node1, 
                                            node2,
                                            comp_res,
                                            self.counter)
        
        print("saved comp # " + str(self.counter))
        self.counter += 1
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
                                                        model.getConss()))
    
    model.includeNodesel(oracle_ns, "oracle_recorder", "testing",
                         536870911,  536870911)

    #print(f"Getting behaviour for instance {problem} "+ str(instance).split("/")[-1] )

    # Run the optimizer
    model.freeTransform()
    model.readProblem(instance)
    model.optimize()
    
    return 1#sucess

def run_episodes(oracle_type, instances, save_dir):
    for instance in instances:
        run_episode(oracle_type, instance, save_dir)
    print("finished running episodes for process " + str(md.current_process()))
    

if __name__ == "__main__":
    

    
    oracle = "optimal_plunger"
    problem = "GISP"
    data_partition = "train"
    cpu_count = 1
    
    #Initializing the model 
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-oracle':
            oracle = str(sys.argv[i + 1])
        if sys.argv[i] == '-problem':
            problem = str(sys.argv[i + 1])
        if sys.argv[i] == '-data_partition':
            data_partition = str(sys.argv[i + 1])
        if sys.argv[i] == '-n_cpu':
            cpu_count = int(sys.argv[i + 1])
   
  
        
       
    save_dir = f"./data/{problem}/{data_partition}"

    try:
        os.makedirs(save_dir)
    except FileExistsError:
        ""
    
    
    instances = list(Path(f"../problem_generation/data/{problem}/{data_partition}").glob("*.lp"))
    
    if cpu_count == 1:
        run_episodes(oracle, instances, save_dir)
    else:
        chunck_size = int(np.ceil(len(instances)/cpu_count))
        
        processes = [  md.Process(name=f"worker {p}", 
                                        target=partial(run_episodes,
                                                       oracle_type=oracle,
                                                       instances=instances[ p*chunck_size : (p+1)*chunck_size], 
                                                       save_dir=save_dir))
                       for p in range(cpu_count) ]
        
        a = list(map(lambda p: p.start(), processes)) #run processes
        b = list(map(lambda p: p.join(), processes)) #join processes
        
        
                         
            
        

        
