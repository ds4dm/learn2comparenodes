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
from node_selectors.oracle_selectors import OracleNodeSelectorAbdel
from recorders import LPFeatureRecorder, CompBehaviourSaver
from pathlib import Path 
import pyscipopt.scip as sp
from multiprocessing import Pool
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
    
    comp_behaviour_saver = CompBehaviourSaver(f"./data/{problem}", instance_name=str(instance).split("/")[-1])
    
    model = sp.Model()
    model.hideOutput()
    
    oracle_ns = OracleNodeSelRecorder(comp_behaviour_saver)
    model.includeNodesel(oracle_ns, "oracle_recorder", "testing", 536870911,  536870911)

    print(f"Getting behaviour for instance {problem} "+ str(instance).split("/")[-1] )
    
    instance = str(instance)
    model.readProblem(instance)
    optsol = model.readSolFile(instance.replace(".lp", ".sol"))
    #record solution : WORKS
    oracle_ns.setOptsol(optsol)
    feature_recorder = LPFeatureRecorder(model.getVars(), model.getConss())
    oracle_ns.set_LP_feature_recorder(feature_recorder)
    

    model.freeTransform()
    model.readProblem(instance)
    
    model.optimize()
    
    comp_behaviour_saver.save_dataset()
    
    return 1#sucess

    
if __name__ == "__main__":
    
    #Defining some variables
    problems = ["GISP"]
    
    
    #Initializing the model 
    
    
    for problem in problems:
        
        instances = list(Path(f"../problem_generation/data/{problem}/train").glob("*.lp"))[0:1]
        
        
        pool = Pool()
        run_episode_closure = partial(run_episode, problem=problem)
        
        comp_savers = pool.map(run_episode_closure, instances)
        
