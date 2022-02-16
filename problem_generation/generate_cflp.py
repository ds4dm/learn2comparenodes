# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 15:23:22 2022

@author: aglabassi
"""

import numpy as np
import sys
import os
import multiprocessing as md
from functools import partial
import pyscipopt.scip as sp


def generate_capacited_facility_location(rng, filename, n_customers, n_facilities, ratio=1):
    """
    Generate a Capacited Facility Location problem following
        Cornuejols G, Sridharan R, Thizy J-M (1991)
        A Comparison of Heuristics and Relaxations for the Capacitated Plant Location Problem.
        European Journal of Operations Research 50:280-297.

    Saves it as a CPLEX LP file.

    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the file to save.
    n_customers: int
        The desired number of customers.
    n_facilities: int
        The desired number of facilities.
    ratio: float
        The desired capacity / demand ratio.
    """
    c_x = rng.rand(n_customers)
    c_y = rng.rand(n_customers)

    f_x = rng.rand(n_facilities)
    f_y = rng.rand(n_facilities)

    demands = rng.randint(5, 35+1, size=n_customers)
    capacities = rng.randint(10, 160+1, size=n_facilities)
    fixed_costs = rng.randint(100, 110+1, size=n_facilities) * np.sqrt(capacities) \
            + rng.randint(90+1, size=n_facilities)
    fixed_costs = fixed_costs.astype(int)

    total_demand = demands.sum()
    total_capacity = capacities.sum()

    # adjust capacities according to ratio
    capacities = capacities * ratio * total_demand / total_capacity
    capacities = capacities.astype(int)
    total_capacity = capacities.sum()

    # transportation costs
    trans_costs = np.sqrt(
            (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 \
            + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2) * 10 * demands.reshape((-1, 1))

    # write problem
    with open(filename, 'w') as file:
        file.write("minimize\nOBJ:")
        file.write("".join([f" +{trans_costs[i, j]}x_{i+1}_{j+1}" for i in range(n_customers) for j in range(n_facilities)]))
        file.write("".join([f" +{fixed_costs[j]}y_{j+1}" for j in range(n_facilities)]))

        file.write("\nSubject to\n")
        for i in range(n_customers):
            file.write(f"demand_{i+1}:" + "".join([f" -1x_{i+1}_{j+1}" for j in range(n_facilities)]) + f" <= -1\n")
        for j in range(n_facilities):
            file.write(f"capacity_{j+1}:" + "".join([f" +{demands[i]}x_{i+1}_{j+1}" for i in range(n_customers)]) + f" -{capacities[j]}y_{j+1} <= +0\n")

        # optional constraints for LP relaxation tightening
        file.write("total_capacity:" + "".join([f" -{capacities[j]}y_{j+1}" for j in range(n_facilities)]) + f" <= -{total_demand}\n")
        for i in range(n_customers):
            for j in range(n_facilities):
                file.write(f"affectation_{i+1}_{j+1}: +1x_{i+1}_{j+1} -1y_{j+1} <= +0\n")

        file.write("\nBounds\n")
        for i in range(n_customers):
            for j in range(n_facilities):
                file.write(f" 0 <= x_{i+1}_{j+1} <= +1\n")

        file.write("\nBinaries\n")
        for j in range(n_facilities):
            file.write(f" y_{j+1}")
        file.write('\nEnd\n')
        file.close()
    print(filename)


def generate_instances(start_seed, end_seed, min_n, max_n, lp_dir, solveInstance):
    
    for seed in range(start_seed, end_seed):
        n_customer =  np.random.randint(min_n, max_n+1)
        n_facility = np.random.randint(min_n, max_n+1)
        ratio = 5
        rng = np.random.RandomState(seed)
        instance_id = np.random.rand()*100
        instance_name = f'n_customer={n_customer}_n_facility={n_facility}_ratio={ratio}_id_{instance_id:0.2f}'
        instance_path = lp_dir +  "/" + instance_name
        generate_capacited_facility_location(rng, instance_path + ".lp", n_customer, n_facility, ratio)
        

        try:
            model = sp.Model()
            model.hideOutput()
            model.readProblem(instance_path + ".lp")
            if solveInstance:
                model.optimize()
                model.writeBestSol(instance_path + ".sol")  
        except:
            os.remove(instance_path + ".lp" )

    

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
    
    n_cpu = 4
    n_instance = 4
    
    exp_dir = "data/CFLP/"
    data_partition = 'test'
    min_n = 60
    max_n = 70
    solveInstance = True
    seed = 0
    
    

    # seed = 0
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-data_partition':
            data_partition = sys.argv[i + 1]
        if sys.argv[i] == '-min_n':
            min_n = int(sys.argv[i + 1])
        if sys.argv[i] == '-max_n':
            max_n = int(sys.argv[i + 1])
        if sys.argv[i] == '-solve':
            solveInstance = bool(int(sys.argv[i + 1]))
        if sys.argv[i] == '-n_instance':
            n_instance = int(sys.argv[i + 1])
        if sys.argv[i] == '-n_cpu':
            n_cpu = int(sys.argv[i + 1])
    
    
    exp_dir = exp_dir + data_partition
    lp_dir= os.path.join(os.path.dirname(__file__), exp_dir)
    try:
        os.makedirs(lp_dir)
    except FileExistsError:
        ""
    
    print("Summary for CFLP generation")
    print(f"n_instance    :     {n_instance}")
    print(f"size interval :     {min_n, max_n}")
    print(f"n_cpu         :     {n_cpu} ")
    print(f"solve         :     {solveInstance}")
    print(f"saving dir    :     {lp_dir}")
    
        
            
    cpu_count = md.cpu_count()//2 if n_cpu == None else n_cpu
    

    
    processes = [  md.Process(name=f"worker {p}", target=partial(generate_instances,
                                                                  seed + p1, 
                                                                  seed + p2, 
                                                                  min_n, 
                                                                  max_n, 
                                                                  lp_dir, 
                                                                  solveInstance))
                 for p,(p1,p2) in enumerate(distribute(n_instance, n_cpu)) ]
    
 
    a = list(map(lambda p: p.start(), processes)) #run processes
    b = list(map(lambda p: p.join(), processes)) #join processes
    print('Generated')
 
    

