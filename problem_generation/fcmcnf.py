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
import networkx as nx
import matplotlib.pyplot as plt


def get_random_uniform_graph(rng, n_nodes, n_arcs, c_range, d_range, ratio, k_max):
    adj_mat = [[0 for _ in range(n_nodes) ] for _ in range(n_nodes)]
    edge_list = []
    incommings = dict([ (j, []) for j in range(n_nodes) ])
    outcommings = dict([(i, []) for i in range(n_nodes) ])
        
    added_arcs = 0
    #gen network, todo: use 
    while(True):
        i = rng.randint(0,n_nodes) 
        j = rng.randint(0,n_nodes)
        if i ==j or adj_mat[i][j] != 0:
            continue
        else:
            c_ij = rng.uniform(*c_range)
            f_ij = rng.uniform(c_range[0]*ratio, c_range[1]*ratio)
            u_ij = rng.uniform(1,k_max+1)* rng.uniform(*d_range)
            adj_mat[i][j] = (c_ij, f_ij, u_ij)
            added_arcs += 1
            edge_list.append((i,j))
            
            outcommings[i].append(j)
            incommings[j].append(i)

            
            
            
        if added_arcs == n_arcs:
            break
        
    G = nx.DiGraph()
    G.add_nodes_from([i for i in range(n_nodes)])
    G.add_edges_from(edge_list)
    
    return G, adj_mat, edge_list, incommings, outcommings


def get_erdos_graph(rng,n_nodes, c_range, d_range, ratio, k_max, er_prob=0.25):
    
    G = nx.erdos_renyi_graph(n=n_nodes, p=er_prob, seed=int(rng.get_state()[1][0]), directed=True)
    adj_mat = [[0 for _ in range(n_nodes) ] for _ in range(n_nodes)]
    edge_list = []
    incommings = dict([ (j, []) for j in range(n_nodes) ])
    outcommings = dict([(i, []) for i in range(n_nodes) ])
    
    for i,j in G.edges:
        c_ij = int(rng.uniform(*c_range))
        f_ij = int(rng.uniform(c_range[0]*ratio, c_range[1]*ratio))
        u_ij = int(rng.uniform(1,k_max+1)* rng.uniform(*d_range))
        adj_mat[i][j] = (c_ij, f_ij, u_ij)
        edge_list.append((i,j))
        
        outcommings[i].append(j)
        incommings[j].append(i)
    
    return G, adj_mat, edge_list, incommings, outcommings
        

    



def generate_fcmcnf(rng, filename, n_nodes, n_commodities, c_range, d_range, k_max, ratio):
    
    G, adj_mat, edge_list, incommings, outcommings = get_erdos_graph(rng, n_nodes, c_range, d_range, ratio, k_max)

    #nx.draw(G)
    print(G)
    #plt.savefig(filename+'.png')
     
    commodities = [ 0 for _ in range(n_commodities) ]
    for k in range(n_commodities):
        while True:
            o_k = rng.randint(0, n_nodes)
            d_k = rng.randint(0, n_nodes)
            
            if nx.has_path(G, o_k, d_k) and o_k != d_k:
                break
        
        demand_k = int(rng.uniform(*d_range))
        commodities[k] = (o_k, d_k, demand_k)
        
    with open(filename, 'w') as file:
        file.write("minimize\nOBJ:")
        file.write("".join([f" + {commodities[k][2]*adj_mat[i][j][0]}x_{i+1}_{j+1}_{k+1}" for (i,j) in edge_list for k in range(n_commodities)]))
        file.write("".join([f" + {adj_mat[i][j][1]}y_{i+1}_{j+1}" for (i,j) in edge_list ]))
        
        
        file.write("\nSubject to\n")
        
        for i in range(n_nodes):
            for k in range(n_commodities):
                
                delta_i = 1 if (commodities[k][0] == i ) else (-1 if commodities[k][1] == i else 0) #1 if source, -1 if sink, 0 if else
                
                file.write(f"flow_{i+1}_{k+1}:" + 
                           "".join([f" +x_{i+1}_{j+1}_{k+1}" for j in outcommings[i] ]) +
                           "".join([f" -x_{j+1}_{i+1}_{k+1}" for j in incommings[i] ]) + f" = {delta_i}\n"   )
                
        
        for (i,j) in edge_list:
            file.write(f"arc_{i+1}_{j+1}:" + 
                       "".join([f" +{commodities[k][2]}x_{i+1}_{j+1}_{k+1}" for k in range(n_commodities) ]) + f"-{adj_mat[i][j][2]}y_{i+1}_{j+1} <= +0\n" )
            

        file.write("\nBinaries\n")
        for (i,j) in edge_list:
            file.write(f" y_{i+1}_{j+1}")
            
        file.write('\nEnd\n')
        file.close()
        
    

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


def generate_instances(start_seed, end_seed, min_n_nodes, max_n_nodes, min_n_commodities, max_n_commodities, lp_dir, solveInstance):
    
    for seed in range(start_seed, end_seed):
        ratio = 5
        rng = np.random.RandomState(seed)
        instance_id = rng.uniform(0,1)*100
        
        
        n_nodes =  rng.randint(min_n_nodes, max_n_nodes+1)
        n_commodities = rng.randint(min_n_commodities, max_n_commodities+1)
        
        
        c_range = (11,50)
        d_range = (10,100)
        
        k_max = n_commodities #loose
        ratio = 100
        
        
        instance_name = f'n_nodes={n_nodes}_n_commodities={n_commodities}_id_{instance_id:0.2f}'
        instance_path = lp_dir +  "/" + instance_name
        filename = instance_path+'.lp'
        
        
        
        
        generate_fcmcnf(rng, filename, n_nodes, n_commodities, c_range, d_range, k_max, ratio)
        
        model = sp.Model()
        model.hideOutput()
        model.readProblem(instance_path + ".lp")
        
        if solveInstance:
            model.optimize()
            model.writeBestSol(instance_path + ".sol")  
            print(model.getNNodes())
            
            if model.getNNodes() <= 1:
                os.remove(instance_path+ ".lp" )
                os.remove(instance_path+ ".sol")
            






