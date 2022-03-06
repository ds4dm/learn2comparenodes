#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 17:16:39 2021

@author: abdel

Contains utilities to save and load comparaison behavioural data


"""

import os
import imp
import torch
import numpy as np
import re

def load_src(name, fpath):
     return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("data_type", "../learning/data_type.py" )

from data_type import BipartiteGraphPairData


class CompFeaturizerSVM():
    def __init__(self, save_dir=None, instance_name=None):
        self.instance_name = instance_name
        self.save_dir = save_dir
        
    def save_comp(self, model, node1, node2, comp_res, comp_id):
        
        f1,f2 = self.get_features(model, node1), self.get_features(model, node2)
        
        
        file_path = os.path.join(self.save_dir, f"{self.instance_name}_{comp_id}.csv")
        file = open(file_path, 'a')
        
        np.savetxt(file, f1, delimiter=',')
        np.savetxt(file, f2, delimiter=',')
        file.write(str(comp_res))
        file.close()
        
        return self
    
    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
        return self

    def get_features(self, model, node):
        f = np.ones((1,10))
        #TODO

        return f
    




class CompFeaturizer():
    
    def __init__(self, save_dir=None, instance_name=None):
        self.instance_name = instance_name
        self.save_dir = save_dir
        
    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
        return self

    
    def set_LP_feature_recorder(self, LP_feature_recorder):
        self.LP_feature_recorder = LP_feature_recorder
        return self
        
    
    def save_comp(self, model, node1, node2, comp_res, comp_id):
        
        torch_geometric_data = self.get_torch_geometric_data(model, node1, node2, comp_res)
        file_path = os.path.join(self.save_dir, f"{self.instance_name}_{comp_id}.pt")
        torch.save(torch_geometric_data, file_path, _use_new_zipfile_serialization=False)
        
        return self
    
    def get_torch_geometric_data(self, model, node1, node2, comp_res=0):
        
        triplet = self.get_triplet_tensors(model, node1, node2, comp_res)
        
        return BipartiteGraphPairData(*triplet[0], *triplet[1], triplet[2])
    
    
    
    
    def get_graph_for_inf(self,model, node):
        
        self.LP_feature_recorder.record_sub_milp_graph(model, node)
        graphidx2graphdata = self.LP_feature_recorder.recorded_light
        all_conss_blocks = self.LP_feature_recorder.all_conss_blocks
        all_conss_blocks_features = self.LP_feature_recorder.all_conss_blocks_features
        
        g_idx = node.getNumber()
        
        var_attributes, cons_block_idxs = graphidx2graphdata[g_idx]
        
    
        g_data = self._get_graph_data(var_attributes, cons_block_idxs, all_conss_blocks, all_conss_blocks_features)
        
        variable_features = g_data[0]
        constraint_features = g_data[1]
        edge_indices = g_data[2]
        edge_features = g_data[3]
        
        lb, ub = node.getLowerbound(), node.getEstimate()
        depth = node.getDepth()
        
        if model.getObjectiveSense() == 'maximize':
            lb,ub = ub,lb
            
        g = (constraint_features,
              edge_indices, 
              edge_features, 
              variable_features, 
              torch.tensor([[lb, -1*ub]], device=self.LP_feature_recorder.device).float(),
              torch.tensor([depth], device=self.LP_feature_recorder.device).float()
              )
            
        return g
        
        
        
        
        
        
        
        
    
    
    
    def get_triplet_tensors(self, model, node1, node2, comp_res=0):
                
        self.LP_feature_recorder.record_sub_milp_graph(model, node1)
        self.LP_feature_recorder.record_sub_milp_graph(model, node2)
        graphidx2graphdata = self.LP_feature_recorder.recorded_light
        all_conss_blocks = self.LP_feature_recorder.all_conss_blocks
        all_conss_blocks_features = self.LP_feature_recorder.all_conss_blocks_features
        
        g0_idx, g1_idx, comp_res = node1.getNumber(), node2.getNumber(), comp_res
        
        var_attributes0, cons_block_idxs0 = graphidx2graphdata[g0_idx]
        var_attributes1, cons_block_idxs1 = graphidx2graphdata[g1_idx]
        
        g_data = self._get_graph_pair_data(var_attributes0, 
                                           var_attributes1, 
                                                         
                                           cons_block_idxs0, 
                                           cons_block_idxs1, 

                                           all_conss_blocks, 
                                           all_conss_blocks_features, 
                                           comp_res)
        
        bounds0 = [node1.getLowerbound(), node1.getEstimate()]
        bounds1 = [node2.getLowerbound(), node2.getEstimate()]
        
        if model.getObjectiveSense() == 'maximize':
            bounds0[1], bounds0[0] = bounds0
            bounds1[1], bounds1[0] = bounds1
            
        return self._to_triplet_tensors(g_data, node1.getDepth(), node2.getDepth(), bounds0, bounds1, self.LP_feature_recorder.device)
    
       
    
    def _get_graph_pair_data(self, var_attributes0, var_attributes1, cons_block_idxs0, cons_block_idxs1, all_conss_blocks, all_conss_blocks_features, comp_res ):
        
        g1 = self._get_graph_data(var_attributes0, cons_block_idxs0, all_conss_blocks, all_conss_blocks_features)
        g2 = self._get_graph_data(var_attributes1, cons_block_idxs1, all_conss_blocks, all_conss_blocks_features)
     
        return list(zip(g1,g2)) + [comp_res]
    
    def _get_graph_data(self, var_attributes, cons_block_idxs, all_conss_blocks, all_conss_blocks_features):
        
        
        adjacency_matrixes = map(all_conss_blocks.__getitem__, cons_block_idxs)
        
        cons_attributes_blocks = map(all_conss_blocks_features.__getitem__, cons_block_idxs)
        
        #TO DO ACCELERATE HSTACK VSTACK
        # adjacency_matrix = torch.hstack(tuple(adjacency_matrixes))
        # cons_attributes = torch.vstack(tuple(cons_attributes_blocks))
        adjacency_matrix = tuple(adjacency_matrixes)[0]
        cons_attributes = tuple(cons_attributes_blocks)[0]
        
        edge_idxs = adjacency_matrix._indices()
        edge_features =  adjacency_matrix._values().unsqueeze(1)
            
        
        return var_attributes, cons_attributes, edge_idxs, edge_features
        
    
    def _to_triplet_tensors(self, g_data, depth0, depth1, bounds0, bounds1, device ):
        
        variable_features = g_data[0]
        constraint_features = g_data[1]
        edge_indices = g_data[2]
        edge_features = g_data[3]
        y = g_data[4]
        lb0, ub0 = bounds0
        lb1, ub1 = bounds1
        
        g1 = (constraint_features[0],
              edge_indices[0], 
              edge_features[0], 
              variable_features[0], 
              torch.tensor([[lb0, -1*ub0]], device=device).float(),
              torch.tensor([depth0], device=device).float()
              )
        g2 = (constraint_features[1], 
              edge_indices[1], 
              edge_features[1], 
              variable_features[1], 
              torch.tensor([[lb1, -1*ub1]], device=device).float(),
              torch.tensor([depth1], device=device).float()
              )
        
        
        
        return (g1,g2,y)
    
        
        
        

    


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
    
    def __init__(self, model, device):
        
        varrs = model.getVars()
        original_conss = model.getConss()
        
        self.model = model
        
        self.n0 = len(varrs)
        
        self.varrs = varrs
        self.var2idx = dict([ (str_var, idx) for idx, var in enumerate(self.varrs) for str_var in [str(var), "t_" + str(var) ]  ])
        
        self.original_conss = original_conss
        
        self.recorded = dict()
        self.recorded_light = dict()
        self.all_conss_blocks = []
        self.all_conss_blocks_features = []
        self.obj_adjacency  = None
        
        self.device = device
        
        # root_graph = self.get_root_graph(model)
        # self.recorded[1] = root_graph
        # self.recorded_light[1] = (root_graph.var_attributes, root_graph.cons_block_idxs)

   
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
                #self._add_conss_to_graph(graph, model, sub_milp.getAddedConss())
                self._change_branched_bounds(graph, sub_milp)
                
            #self._add_scip_obj_cons(model, sub_milp, graph)
            self.recorded[sub_milp.getNumber()] = graph
            self.recorded_light[sub_milp.getNumber()] = (graph.var_attributes, 
                                                         graph.cons_block_idxs)
    
    def get_root_graph(self, model):
        
        graph = BipartiteGraphStatic0(self.n0, self.device)
        
        self._add_vars_to_graph(graph, model)
        self._add_conss_to_graph(graph, model, self.original_conss)
    
        
        return graph
    
    
    # def _get_obj_adjacency(self, model):
    
    #    if self.obj_adjacency  == None:
    #        var_coeff = { self.var2idx[ str(t[0]) ]:c for (t,c) in model.getObjective().terms.items() if c != 0.0 }
    #        var_idxs = list(var_coeff.keys())
    #        weigths = list(var_coeff.values())
    #        cons_idxs = [0]*len(var_idxs)
           
    #        self.obj_adjacency =  torch.torch.sparse_coo_tensor([var_idxs, cons_idxs], weigths, (self.n0, 1), device=self.device)
    #        self.obj_adjacency = torch.hstack((-1*self.obj_adjacency, self.obj_adjacency))
           
    #    return self.obj_adjacency         
       
    
    # def _add_scip_obj_cons(self, model, sub_milp, graph):
    #     adjacency_matrix = self._get_obj_adjacency(model)
    #     cons_feature = torch.tensor([[ sub_milp.getEstimate() ], [ -sub_milp.getLowerbound() ]], device=self.device).float()
    #     graph.cons_block_idxs.append(len(self.all_conss_blocks_features))
    #     self.all_conss_blocks_features.append(cons_feature)
    #     self.all_conss_blocks.append(adjacency_matrix)
  
    
                
    def _add_vars_to_graph(self, graph, model):
        #add vars
        
        for idx, var in enumerate(self.varrs):
            graph.var_attributes[idx] = self._get_feature_var(model, var)

    
    def _add_conss_to_graph(self, graph, model, conss):
        
        if len(conss) == 0:
            return

        cons_attributes = torch.zeros(len(conss), graph.d1, device=self.device).float()
        var_idxs = []
        cons_idxs = []
        weigths = []
        for cons_idx, cons in enumerate(conss):

            cons_attributes[cons_idx] =  self._get_feature_cons(model, cons)
          
            for var, coeff in model.getValsLinear(cons).items():
                var_idxs.append(self.var2idx[str(var)] )
                cons_idxs.append(cons_idx)
                weigths.append(coeff)


        adjacency_matrix =  torch.sparse_coo_tensor([var_idxs, cons_idxs], weigths, (self.n0, len(conss)), device=self.device) 
        
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
        
        try:
            
            cons_n = str(cons)
            
            if re.match('flow', cons_n):
                rhs = model.getRhs(cons)
                leq = 0
                eq = 1
                geq = 0
            elif re.match('arc', cons_n):
                rhs = 0
                leq = eq =  1
                geq = 0
            else:
                rhs = model.getRhs(cons)
                leq = eq = 1
                geq = 0
        except:
            'logicor no repr'
            rhs = 0
            leq = eq = 1
            geq = 0
  
        return torch.tensor([ rhs, leq, eq, geq ], device=self.device).float()
    
    def _get_feature_var(self, model, var):
        
        lb, ub = var.getLbOriginal(), var.getUbOriginal()
        
        if lb <= - 0.999e+20:
            lb = -300
        if ub >= 0.999e+20:
            ub = 300
            
        objective_coeff = model.getObjective()[var]
        
        binary, integer, continuous = self._one_hot_type(var)
    
        
        return torch.tensor([ lb, ub, objective_coeff, binary, integer, continuous ], device=self.device).float()
    
    
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
    def __init__(self, n0, device, d0=6, d1=4, allocate=True):
        
        self.n0, self.d0, self.d1 = n0, d0, d1
        self.device = device
        
        if allocate:
            self.var_attributes = torch.zeros(n0,d0, device=self.device)
            self.cons_block_idxs = []
        else:
            self.var_attributes = None
            self.cons_block_idxs = None
    
    
    def copy(self):
        
        copy = BipartiteGraphStatic0(self.n0, self.device, allocate=False)
        
        copy.var_attributes = self.var_attributes.clone()
        copy.cons_block_idxs = self.cons_block_idxs #no scip bonds
        
        return copy
