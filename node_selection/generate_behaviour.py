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
from recorders import LPFeatureRecorder, CompFeaturizer, CompFeaturizerSVM




class OracleNodeSelRecorder(OracleNodeSelectorAbdel):
    
    def __init__(self, oracle_type, comp_behaviour_saver=None):
        super().__init__(oracle_type)
        self.counter = 0
        self.comp_behaviour_saver = comp_behaviour_saver
    
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
        
            print("saved comp # " + str(self.counter))
            self.counter += 1
        
        #make it bad to generate more data !
        if comp_type in [-1,1]:
            comp_res = -1 if comp_res == 1 else 1
        else:
            comp_res = 0
            
        return comp_res



def run_episode(oracle_type, instance,  save_dir, svm):
    
    model = sp.Model()
    model.hideOutput()
    
    #Setting up oracle selector
    instance = str(instance)
    model.readProblem(instance)
    model.setIntParam('separating/maxrounds', 0)
    
    optsol = model.readSolFile(instance.replace(".lp", ".sol"))
    
    
    CompFeaturizerConstrucor = CompFeaturizerSVM if svm else CompFeaturizer
    comp_behaviour_saver = CompFeaturizerConstrucor(f"{save_dir}", 
                                              instance_name=str(instance).split("/")[-1])
    
    
    oracle_ns = OracleNodeSelRecorder(oracle_type, comp_behaviour_saver)
    oracle_ns.setOptsol(optsol)
    if isinstance(comp_behaviour_saver, CompFeaturizer): #gnn
        oracle_ns.set_LP_feature_recorder(LPFeatureRecorder(model, 'cuda'))
        
        print(oracle_ns)
    
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


def run_episodes(oracle_type, instances, save_dir, svm=False):
    
    for instance in instances:
        run_episode(oracle_type, instance, save_dir, svm)
        
    print("finished running episodes for process " + str(md.current_process()))
        
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
    n_cpu = 1
    svm = False
    
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
        if sys.argv[i] == '-svm':
            n_cpu = bool(int(sys.argv[i + 1]))
   
   
  
    for data_partition in data_partitions:
        

        save_dir = lp_dir= os.path.join(os.path.dirname(__file__), f'./data{"_svm" if svm else ""}/{problem}/{data_partition}')

    
        try:
            os.makedirs(save_dir)
        except FileExistsError:
            ""
        
        
        instances = list(Path(os.path.join(os.path.dirname(__file__), 
                                           f"../problem_generation/data/{problem}/{data_partition}")).glob("*.lp"))
        n_instance = len(instances)
        
        print(f"Geneating {data_partition} samples from {n_instance} instances using oracle {oracle}")
        
        chunck_size = int(np.floor(len(instances)/n_cpu))
        processes = [  md.Process(name=f"worker {p}", 
                                        target=partial(run_episodes,
                                                        oracle_type=oracle,
                                                        instances=instances[ p1 : p2], 
                                                        save_dir=save_dir,
                                                        svm=svm))
                        for p,(p1,p2) in enumerate(distribute(n_instance, n_cpu))]
        
        
            
        a = list(map(lambda p: p.start(), processes)) #run processes
        b = list(map(lambda p: p.join(), processes)) #join processes
        
            
    nnodes = np.genfromtxt("nnodes.csv", delimiter=",")[:-1]
    times = np.genfromtxt("times.csv", delimiter=",")[:-1]
        
    print(f"Mean number of node created  {np.mean(nnodes)}")
    print(f"Mean solving time  {np.mean(times)}")
    print(f"Median number of node created  {np.median(nnodes)}")
    print(f"Median solving time  {np.median(times)}")
    
    
                         
            
        

        
