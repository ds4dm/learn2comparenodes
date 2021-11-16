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



def run_episode(instance, problem):
    
    model = sp.Model()
    model.hideOutput()
    
    #Setting up oracle selector
    instance = str(instance)
    model.readProblem(instance)
    
    optsol = model.readSolFile(instance.replace(".lp", ".sol"))
    comp_behaviour_saver = CompBehaviourSaver(f"./data/{problem}", instance_name=str(instance).split("/")[-1])
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

def run_episodes(instances, problem):
    for instance in instances:
        run_episode(instance, problem)
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

                
        
        instances = list(Path(f"../problem_generation/data/{problem}/train").glob("*n=4*.lp"))[0:10]
        cpu_count = md.cpu_count()
        chunck_size = int(np.ceil(len(instances)/cpu_count))
        
        processes = [  md.Process(name=f"worker {p}", target=partial(run_episodes, instances=instances[ p*chunck_size : (p+1)*chunck_size], problem=problem[:]))
                       for p in range(cpu_count)]
        
        a = list(map(lambda p: p.start(), processes)) #run processes
        b = list(map(lambda p: p.join(), processes)) #join processes
        
        
            
        
                             
            
        

        