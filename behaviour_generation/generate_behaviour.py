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
from node_selectors.oracle_selectors import OracleNodeSelectorAbdel
from recorders import LPFeatureRecorder, CompBehaviourSaver
from pathlib import Path 
import pyscipopt.scip as sp
import numpy as np
import multiprocessing as md
from functools import partial


class OracleNodeSelRecorder(OracleNodeSelectorAbdel):
    
    def __init__(self, comp_behaviour_saver=None):
        super().__init__()
        self.counter = 0
        self.comp_behaviour_saver = comp_behaviour_saver
    
    def set_LP_feature_recorder(self, LP_feature_recorder):
        self.comp_behaviour_saver.set_LP_feature_recorder(LP_feature_recorder)
        
        
    def nodecomp(self, node1, node2):
        
        comp_res = super().nodecomp(node1, node2)
        self.comp_behaviour_saver.append_data(self.model, node1, node2, comp_res)
        print("saved comp # " + str(self.counter))
        self.counter += 1
        return comp_res



def run_episode(instance,  save_dir):
    
    model = sp.Model()
    model.hideOutput()
    
    #Setting up oracle selector
    instance = str(instance)
    model.readProblem(instance)
    
    optsol = model.readSolFile(instance.replace(".lp", ".sol"))
    comp_behaviour_saver = CompBehaviourSaver(f"{save_dir}", instance_name=str(instance).split("/")[-1])
    oracle_ns = OracleNodeSelRecorder(comp_behaviour_saver)
    oracle_ns.setOptsol(optsol)
    oracle_ns.set_LP_feature_recorder(LPFeatureRecorder(model.getVars(), model.getConss()))
    
    model.includeNodesel(oracle_ns, "oracle_recorder", "testing", 536870911,  536870911)

    #print(f"Getting behaviour for instance {problem} "+ str(instance).split("/")[-1] )

    # Run the optimizer and save behaviour
    model.freeTransform()
    model.readProblem(instance)
    model.optimize()
    
    comp_behaviour_saver.save_dataset()
    
    return 1#sucess

def run_episodes(instances, save_dir):
    for instance in instances:
        run_episode(instance, save_dir)
    print("finished running episodes for process " + str(md.current_process()))

if __name__ == "__main__":
    
    #Defining some variables
    problems = ["GISP"]
    
    
    #Initializing the model 
    
            
            
    
    for problem in problems:
        try:
            os.makedirs(f"./data/{problem}")
            
        except FileExistsError:
            ""

                
        
        instances_train = list(Path(f"../problem_generation/data/{problem}/train").glob("*.lp"))
        instances_valid = list(Path(f"../problem_generation/data/{problem}/valid").glob("*.lp"))
        save_train_dir = f"./data/{problem}/train"
        save_valid_dir = f"./data/{problem}/valid"
        cpu_count = md.cpu_count()
        chunck_size_train = int(np.ceil(len(instances_train)/cpu_count))
        chunck_size_valid = int(np.ceil(len(instances_valid)/cpu_count))
        
        processes_train = [  md.Process(name=f"worker {p}", target=partial(run_episodes, 
                                                                     instances=instances_train[ p*chunck_size_train : (p+1)*chunck_size_train], 
                                                                     save_dir=save_train_dir))
                       for p in range(cpu_count)]
        
        processes_valid = [  md.Process(name=f"worker {p}", target=partial(run_episodes, 
                                                                     instances=instances_valid[ p*chunck_size_valid : (p+1)*chunck_size_valid], 
                                                                     save_dir=save_valid_dir))
                       for p in range(cpu_count)]
        
        a = list(map(lambda p: p.start(), processes_train)) #run processes
        b = list(map(lambda p: p.join(), processes_train)) #join processes
        c = list(map(lambda p: p.start(), processes_valid)) #run processes
        d = list(map(lambda p: p.join(), processes_valid)) #join processes
        
        
            
        
                             
            
        

        
