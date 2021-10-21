#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:24:33 2021

@author: abdel
"""


#Test different nodesels under different problems

from oracle_nodesel import OracleNodeSelectorAbdel
from recorder import Node2Grapher, CompFeatureSaver
from pathlib import Path 
import pyscipopt.scip as sp
import numpy as np

class OracleNodeSelRecorder(OracleNodeSelectorAbdel):
    
    def __init__(self, comp_feature_saver =None):
        super().__init__()
        self.comp_feature_saver = comp_feature_saver
    
    def set_LP_feature_recorder(self, LP_feature_recorder):
        self.comp_feature_saver.set_LP_feature_recorder(LP_feature_recorder)
        
        
    def nodecomp(self, node1, node2):
        
        comp_res = super().nodecomp(node1, node2)
        self.comp_feature_saver.append_data(self.model, node1, node2, comp_res)
        
        return comp_res
    
    
def get_stats(nodesels, instances):
    
    
    nodesels_record = dict((nodesel, []) for nodesel in nodesels)
    
    for instance in instances:
        
        instance = str(instance)
        model.readProblem(instance)
        optsol = model.readSolFile(instance.replace(".lp", ".sol"))
        
        #record solution : WORKS
        oracle_ns.setOptsol(optsol)
        feature_recorder = Node2Grapher(model.getVars(), model.getConss())
        oracle_ns.set_LP_feature_recorder(feature_recorder)
        
        
        #test nodesels
        for nodesel in nodesels:
            
            model.freeTransform()
            model.readProblem(instance)
            
            #canceL all otheer nodesels, WORKS
            for other in nodesels:
                    model.setNodeselPriority(other, 100)
                    
            #activate this nodesel, WORKS
            model.setNodeselPriority(nodesel, 536870911)
            
            
            print( nodesel + " on GISP " + instance.split("/")[-1].split(".lp")[0] + '\n') 
            model.optimize()
            print("    # of processed nodes : " + str(model.getNNodes()) +"\n")
            print("    Time                 : " + str(model.getSolvingTime()) +"\n")
            nodesels_record[nodesel].append((model.getNNodes(), model.getSolvingTime()))

    return nodesels_record



def display_stats(nodesels_record):
    
    for k in nodesels_record:
        nnode_mean, time_mean = np.mean(nodesels_record[k], axis=0)
        nnode_med, time_med = np.median(nodesels_record[k], axis=0)
        print( k + f"\n \t Means : NNode {int(nnode_mean)}, time {int(time_mean)}" + 
              f"\n \t Medians : NNodes {int(nnode_med)}, time {int(time_med)}" )
        
#Defining some variables
gisp_instances = Path("./problem_generation/GISP/").glob("*.lp")
oracle_name = 'oracle'
nodesels = [oracle_name]



#Initializing the model 
model = sp.Model()
model.hideOutput()
comp_feature_saver = CompFeatureSaver()
oracle_ns = OracleNodeSelRecorder(comp_feature_saver)
model.includeNodesel(oracle_ns, oracle_name, 'testing',100, 100)



display_stats(get_stats(nodesels, gisp_instances))

print(len(comp_feature_saver.get_dataset()))





"""
GISP
{'oracle': [(21980, 536.88903),
  (8485, 172.335328),
  (4880, 106.475179),
  (3868, 75.894912),
  (3107, 52.380571),
  (6803, 100.78423)],
 'estimate': [(29442, 481.002816),
  (9990, 154.707946),
  (8920, 131.88883),
  (6870, 82.019173),
  (5031, 62.321967),
  (8453, 117.592189)],
 'dfs': [(25974, 383.803651),
  (16747, 188.050372),
  (7569, 92.087239),
  (7759, 73.404249),
  (6688, 59.423487),
  (9597, 103.478273)]}
"""
   

