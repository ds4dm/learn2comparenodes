#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 17:16:39 2021

@author: abdel

Contains utilities to save and load comparaison behavioural data


"""


import pickle
import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse


class CompBehaviourSaver():
    
    def __init__(self, file_path, instance_name):
        self.instance_name = instance_name
        self.file_path = file_path
        self.dataset = []
    
    def set_file_path(self, file_path):
        self.file_path = file_path
        return self
        
        
    def get_dataset(self):
        return self.dataset
    
    def set_dataset(self, dataset):
        self.dataset = dataset
        return self
    
    def set_LP_feature_recorder(self, LP_feature_recorder):
        self.LP_feature_recorder = LP_feature_recorder
        return self
    
    
    def append_data(self, model, node1, node2, comp_res):
        
        self.LP_feature_recorder.record_sub_milp_graph(model, node1)
        self.LP_feature_recorder.record_sub_milp_graph(model, node2)
        
        self.dataset.append((node1.getNumber(), node2.getNumber(), comp_res))
        
        return self
        
   
    
    def _get_graph_pair_data( var_attributes0, var_attributes1, cons_block_idxs1, cons_block_idxs2, all_conss_blocks, all_conss_blocks_features, comp_res ):
             
        var_attributes = np.hstack((var_attributes0, var_attributes1)) #n_var x (2*dim_var)
        
        adjacency_matrixes = np.array(all_conss_blocks)[list(set(cons_block_idxs1 + cons_block_idxs2))]
        cons_attributes_blocks = np.array(all_conss_blocks_features)[list(set(cons_block_idxs1 + cons_block_idxs2))]
        
        adjacency_matrix = sparse.hstack(tuple(adjacency_matrixes), format="coo") #n_var x card(conss1 U conss1)
        
        cons_attributes = np.vstack(tuple(cons_attributes_blocks)) #card(conss1 U conss2) X cons_dim
        
        
        return var_attributes, cons_attributes, np.vstack( (adjacency_matrix.row, adjacency_matrix.col)), adjacency_matrix.data[:, np.newaxis], comp_res
        
    
    def save_dataset(self):
        
        with open(self.file_path+f"/{self.instance_name}.pickle", 'wb') as handle:
            
            pickle.dump(self.LP_feature_recorder.recorded_light, handle, protocol=pickle.HIGHEST_PROTOCOL) #dict nodenumbers to node0attributes, conssidxs
            pickle.dump(self.LP_feature_recorder.all_conss_blocks, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.LP_feature_recorder.all_conss_blocks_features ,handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

            handle.close()
        
        return self
    
    def load_behaviour_from_pickle(pickle_file_path):
        
        with open(pickle_file_path, 'rb') as handle:
            
            graphidx2graphdata = pickle.load(handle)
            all_conss_blocks = pickle.load(handle)
            all_conss_blocks_features = pickle.load(handle)
            dataset = pickle.load(handle)
            
            handle.close()
            
                    
        for data in dataset:
            g1_idx, g2_idx, comp_res = data[0], data[1], data[2]
            var_attributes0, cons_block_idxs0 = graphidx2graphdata[g1_idx]
            var_attributes1, cons_block_idxs1 = graphidx2graphdata[g2_idx]
            
            g = CompBehaviourSaver._get_graph_pair_data(var_attributes0, var_attributes1, cons_block_idxs0, cons_block_idxs1, all_conss_blocks, all_conss_blocks_features, comp_res)
        
            yield g #binary classification supervised learning
    
    
 
    
        

  


# params_to_set_false = ["constraints/linear/upgrade/indicator",
#                        "constraints/linear/upgrade/logicor",
#                        "constraints/linear/upgrade/knapsack",
#                        "constraints/linear/upgrade/setppc",
#                        "constraints/linear/upgrade/xor",
#                        "constraints/linear/upgrade/varbound"]
# model = scip.Model()


#Converts a branch and bound node, aka a sub-LP, to a bipartite var/constraint 
#graph representation
#1LP recorder per problem
class LPFeatureRecorder():
    
    def __init__(self, varrs, original_conss):
        
        self.n0 = len(varrs)
        
        self.varrs = varrs
        self.var2idx = dict([ (str_var, idx) for idx, var in enumerate(self.varrs) for str_var in [str(var), "t_" + str(var) ]   ])
        
        self.original_conss = original_conss
        
        self.recorded = dict()
        self.recorded_light = dict()
        self.all_conss_blocks = []
        self.all_conss_blocks_features = []
        
    
    def get_graph(self, model, sub_milp):
        
        sub_milp_number = sub_milp.getNumber()
        if sub_milp_number in self.recorded:
            return self.recorded[ sub_milp_number]
        else:
            self.record_sub_milp_graph(model, sub_milp)
            return self.recorded[ sub_milp_number ]
        
    
    def record_sub_milp_graph(self, model, sub_milp):
        
             
        if sub_milp.getNumber() not in self.recorded:
            
            parent = sub_milp.getParent()

            if parent == None: #Root
                graph = self.get_root_graph(model)
                
            else:
               
                graph = self.get_graph(model, parent).copy()
                
                self._add_conss_to_graph(graph, model, parent.getAddedConss())
                self._change_branched_bounds(graph, sub_milp)
                
            self.recorded[sub_milp.getNumber()] = graph
            self.recorded_light[sub_milp.getNumber()] = (graph.var_attributes, graph.cons_block_idxs)
    
    
    def get_root_graph(self, model):
        
        graph = BipartiteGraphStatic0(self.n0)
        
        self._add_vars_to_graph(graph, model)
        self._add_conss_to_graph(graph, model, self.original_conss)
    
        
        return graph
    
                
    def _add_vars_to_graph(self, graph, model):
        #add vars
        
        for idx, var in enumerate(self.varrs):
            graph.var_attributes[idx] = self._get_feature_var(model, var)

    
    def _add_conss_to_graph(self, graph, model, conss):

        cons_attributes = np.zeros((len(conss), graph.d1))        
        
        var_idxs = []
        cons_block_idxs = []
        weigths = []
        for cons_idx, cons in enumerate(conss):
            #vstack the cons n graph
            cons_attributes[cons_idx] =  self._get_feature_cons(model, cons)
            #h stack the coeff

            for var, coeff in model.getValsLinear(cons).items():
                
                var_idxs.append(self.var2idx[str(var)] )
                cons_block_idxs.append(cons_idx)
                weigths.append(coeff)

            
        
        adjacency_matrix =  csr_matrix( (weigths, (var_idxs, cons_block_idxs) ), shape=(self.n0, len(conss))) 
        
        #add idx to graph
        graph.cons_block_idxs.append(len(self.all_conss_blocks_features)) #carreful with parralelization
        
        #add appropriate structure to self
        self.all_conss_blocks_features.append(cons_attributes)
        self.all_conss_blocks.append(adjacency_matrix)

    
        
        
    def _change_branched_bounds(self, graph, sub_milp):
        
        bvars, bbounds, btypes = sub_milp.getParentBranchings()
        
        for bvar, bbound, btype in zip(bvars, bbounds, btypes): 
            var_idx = self.var2idx[str(bvar)]
            graph.var_attributes[var_idx, int(btype) ] = bbound
            
        
    
    def _get_feature_cons(self, model, cons):
        rhs = model.getRhs(cons)
        return np.array([ rhs ])
    
    def _get_feature_var(self, model, var):
        
        lb, ub = var.getLbOriginal(), var.getUbOriginal()
        
        objective_coeff = model.getObjective()[var]
        
        binary, integer, continuous = self._one_hot_type(var)
    
        
        return np.array([ lb, ub, objective_coeff, binary, integer, continuous ])
    
    
    def _one_hot_type(self, var):
        vtype = var.vtype()
        binary, integer, continuous = 0,0,0
        
        if vtype == 'BINARY':
            binary = 1
        elif vtype == 'INTEGER':
            integer = 1
        elif vtype == 'CONTINUOUS':
            continuous = 1
            
        return binary, integer,  continuous
        
        
        
class BipartiteGraphStatic0():
    
    #Defines the structure of the problem solved. Invariant toward problems
    def __init__(self, n0, d0=6, d1=1):
        
        self.n0, self.d0, self.d1 = n0, d0, d1
        
        self.var_attributes = np.zeros((n0,d0))
        self.cons_block_idxs = []
    
    
    def copy(self):
        
        copy = BipartiteGraphStatic0(self.n0)
        
        copy.var_attributes = self.var_attributes[:,:]
        copy.cons_block_idxs = self.cons_block_idxs[:]
        
        return copy
   
        
        