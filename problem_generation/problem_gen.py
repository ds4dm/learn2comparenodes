import os
import sys
import networkx as nx
import random
import pyscipopt as sp
import numpy as np
import multiprocessing as md
from functools import partial
import imp

import gisp
import fcmcnf



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
    instance = None
    n_cpu = 4
    n_instance = 16
    
    problem = 'FCMCNF'
    
    exp_dir = f"data/{problem}/"
    data_partition = 'test'
    min_n = 15
    max_n = 20
    n_nodes = 50
    er_prob = 0.6
    whichSet = 'SET2'
    setParam = 100.0
    alphaE2 = 0.5
    timelimit = 7200.0
    solveInstance = True
    
    seed = 0
    
    
    min_n_nodes = 20
    max_n_nodes = 20
    
    min_n_arcs = 120
    max_n_arcs = 140
    
    min_n_commodities = 30
    max_n_commodities = 40
    
    if problem == 'FCMCNF':
        min_n, max_n = min_n_arcs, max_n_arcs
    
    
    
    

    # seed = 0
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-instance':
            instance = sys.argv[i + 1]
        if sys.argv[i] == '-data_partition':
            data_partition = sys.argv[i + 1]
        if sys.argv[i] == '-min_n':
            min_n = int(sys.argv[i + 1])
        if sys.argv[i] == '-max_n':
            max_n = int(sys.argv[i + 1])
        if sys.argv[i] == '-er_prob':
            er_prob = float(sys.argv[i + 1])
        if sys.argv[i] == '-whichSet':
            whichSet = sys.argv[i + 1]
        if sys.argv[i] == '-setParam':
            setParam = float(sys.argv[i + 1])
        if sys.argv[i] == '-alphaE2':
            alphaE2 = float(sys.argv[i + 1])
        if sys.argv[i] == '-timelimit':
            timelimit = float(sys.argv[i + 1])
        if sys.argv[i] == '-solve':
            solveInstance = bool(int(sys.argv[i + 1]))
        if sys.argv[i] == '-seed_start':
            seed = int(sys.argv[i + 1])
        if sys.argv[i] == '-n_instance':
            n_instance = int(sys.argv[i + 1])
        if sys.argv[i] == '-n_cpu':
            n_cpu = int(sys.argv[i + 1])
    
    assert exp_dir is not None
    if instance is None:
        assert min_n is not None
        assert max_n is not None
    
    exp_dir = exp_dir + data_partition
    lp_dir= os.path.join(os.path.dirname(__file__), exp_dir)
    try:
        os.makedirs(lp_dir)
    except FileExistsError:
        ""
    
    print(f"Summary for {problem} generation")
    print(f"n_instance    :     {n_instance}")
    print(f"size interval :     {min_n, max_n}")
    print(f"n_cpu         :     {n_cpu} ")
    print(f"solve         :     {solveInstance}")
    print(f"saving dir    :     {lp_dir}")
    
        
            
    cpu_count = md.cpu_count()//2 if n_cpu == None else n_cpu
    



    if problem == 'GISP':
        processes = [  md.Process(name=f"worker {p}", target=partial(gisp.generate_instances,
                                                                      seed + p1, 
                                                                      seed + p2, 
                                                                      whichSet, 
                                                                      setParam, 
                                                                      alphaE2, 
                                                                      min_n, 
                                                                      max_n, 
                                                                      er_prob, 
                                                                      instance, 
                                                                      lp_dir, 
                                                                      solveInstance))
                     
                     
                     for p,(p1,p2) in enumerate(distribute(n_instance, n_cpu)) ]
        
    elif problem == 'FCMCNF':
        
#=============================================================================
        processes = [  md.Process(name=f"worker {p}", target=partial(fcmcnf.generate_instances,
                                                                      seed + p1, 
                                                                      seed + p2, 
                                                                      min_n_nodes,
                                                                      max_n_nodes,
                                                                      min_n_arcs,
                                                                      max_n_arcs,
                                                                      min_n_commodities,
                                                                      max_n_commodities,
                                                                      lp_dir, 
                                                                      solveInstance))
                      
                      
                      for p,(p1,p2) in enumerate(distribute(n_instance, n_cpu)) ]
        
#=============================================================================
        #generate_fcmcnf.generate_instances(0, n_instance, min_n_nodes, max_n_nodes, min_n_arcs, max_n_arcs, min_n_commodities, max_n_commodities, lp_dir, solveInstance)
        
    
 
    a = list(map(lambda p: p.start(), processes)) #run processes
    b = list(map(lambda p: p.join(), processes)) #join processes
    print('Generated')

            