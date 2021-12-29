#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 17:16:39 2021

@author: abdel

Contains utilities to save and load comparaison behavioural data


"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse
import torch
import torch_geometric
import os.path as osp


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
    
    
    def get_inference_features(self, model, node1, node2):
        
        self.LP_feature_recorder.record_sub_milp_graph(model, node1)
        self.LP_feature_recorder.record_sub_milp_graph(model, node2)
        graphidx2graphdata = self.LP_feature_recorder.recorded_light
        all_conss_blocks = self.LP_feature_recorder.all_conss_blocks
        all_conss_blocks_features = self.LP_feature_recorder.all_conss_blocks_features
        
        comp_res = 0 #useless data
        
        g0_idx, g1_idx, comp_res = node1.getNumber(), node2.getNumber(), comp_res
        
        
        var_attributes0, cons_block_idxs0, objbound0 = graphidx2graphdata[g0_idx]
        var_attributes1, cons_block_idxs1, objbound1 = graphidx2graphdata[g1_idx]
        
        g_data = CompFeaturizer._get_graph_pair_data(var_attributes0, 
                                                         var_attributes1, 
                                                         
                                                         cons_block_idxs0, 
                                                         cons_block_idxs1, 

                                                         all_conss_blocks, 
                                                         all_conss_blocks_features, 
                                                         comp_res)
        
        data = self._to_tensors(g_data, objbound0, objbound1)
        
        return data
        
        
    
    def save_comp(self, model, node1, node2, comp_res, comp_id):
        
        self.LP_feature_recorder.record_sub_milp_graph(model, node1)
        self.LP_feature_recorder.record_sub_milp_graph(model, node2)
        
        graphidx2graphdata = self.LP_feature_recorder.recorded_light
        all_conss_blocks = self.LP_feature_recorder.all_conss_blocks
        all_conss_blocks_features = self.LP_feature_recorder.all_conss_blocks_features
        
        g0_idx, g1_idx, comp_res = node1.getNumber(), node2.getNumber(), comp_res
        
        
        var_attributes0, cons_block_idxs0, objbound0 = graphidx2graphdata[g0_idx]
        var_attributes1, cons_block_idxs1, objbound1 = graphidx2graphdata[g1_idx]
        
        g_data = CompFeaturizer._get_graph_pair_data(var_attributes0, 
                                                         var_attributes1, 
                                                         
                                                         cons_block_idxs0, 
                                                         cons_block_idxs1, 

                                                         all_conss_blocks, 
                                                         all_conss_blocks_features, 
                                                         comp_res)
        
        file_path = osp.join(self.save_dir, f"{self.instance_name}_{comp_id}.pt")
        data = self._to_tensors(g_data, objbound0, objbound1)    
        torch.save(data, file_path, _use_new_zipfile_serialization=False)
        
        return self
    
    def _to_tensors(self, g_data, objbound0, objbound1):

        variable_features = [ torch.FloatTensor(g_data[0][0]), torch.FloatTensor(g_data[0][1]) ]
        constraint_features = [ torch.FloatTensor(g_data[1][0]),torch.FloatTensor(g_data[1][1]) ]
        edge_indices = [ torch.LongTensor(g_data[2][0]), torch.LongTensor(g_data[2][1]) ] 
        edge_features =  [ torch.FloatTensor(g_data[3][0]), torch.FloatTensor(g_data[3][1]) ]
        y = g_data[4]
        
        
        constraint_features[0] = torch.cat((constraint_features[0], 
                                          torch.FloatTensor([[objbound0]])))
        
        constraint_features[1] = torch.cat((constraint_features[1], 
                                          torch.FloatTensor([[objbound1]])))
        
        obj_bound_weight = variable_features[0][:,2].unsqueeze(1), variable_features[1][:,2].unsqueeze(1)
        
        edge_features[0] = torch.cat((edge_features[0], obj_bound_weight[0]))
        edge_features[1] = torch.cat((edge_features[1], obj_bound_weight[1]))
        
        
        edge_to_stack0 = torch.LongTensor([range(0,variable_features[0].shape[0]), 
                                          [constraint_features[0].shape[0] - 1 for _ in range(variable_features[0].shape[0]) ]])
        
        edge_to_stack1 = torch.LongTensor([range(0,variable_features[1].shape[0]), 
                                          [constraint_features[1].shape[0] - 1 for _ in range(variable_features[1].shape[0]) ]])
        
        edge_indices[0] = torch.cat((edge_indices[0], edge_to_stack0), dim=1)
        edge_indices[1] = torch.cat((edge_indices[1], edge_to_stack1), dim=1)

        g1 = normalize_graph(variable_features[0], constraint_features[0], edge_indices[0], edge_features[0])
        g2 = normalize_graph(variable_features[1], constraint_features[1], edge_indices[1], edge_features[1])
        
        return BipartiteGraphPairData(*g1, *g2, y)
        
        
        
   
    
    def _get_graph_pair_data( var_attributes0, var_attributes1, cons_block_idxs0, cons_block_idxs1, all_conss_blocks, all_conss_blocks_features, comp_res ):
        
        adjacency_matrixes0 = np.array(all_conss_blocks)[cons_block_idxs0]
        adjacency_matrixes1 = np.array(all_conss_blocks)[cons_block_idxs1]
        
        cons_attributes_blocks0 = np.array(all_conss_blocks_features)[cons_block_idxs0]
        cons_attributes_blocks1 = np.array(all_conss_blocks_features)[cons_block_idxs1]
        
        adjacency_matrix0 = sparse.hstack(tuple(adjacency_matrixes0), format="coo") #n_var x card(conss1 )
        adjacency_matrix1 = sparse.hstack(tuple(adjacency_matrixes1), format="coo") #n_var x card(conss1
        
        cons_attributes0 = np.vstack(tuple(cons_attributes_blocks0)) #card(conss1 U conss2) X cons_dim
        cons_attributes1 = np.vstack(tuple(cons_attributes_blocks1))
        
        
        var_attributes = (var_attributes0, var_attributes1)
        cons_attributes = (cons_attributes0, cons_attributes1)
        edge_idxs = (np.vstack( (adjacency_matrix0.row, adjacency_matrix0.col)), np.vstack( (adjacency_matrix1.row, adjacency_matrix1.col)))
        edge_features =  (adjacency_matrix0.data[:, np.newaxis], adjacency_matrix1.data[:, np.newaxis])
            
        return var_attributes, cons_attributes, edge_idxs, edge_features, comp_res
                
    


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
                
            graph.objbounds = sub_milp.getEstimate()
            
            self.recorded[sub_milp.getNumber()] = graph
            self.recorded_light[sub_milp.getNumber()] = (graph.var_attributes, 
                                                         graph.cons_block_idxs,
                                                         graph.objbounds)
    
    
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
        
        if lb <= - 0.999e+20:
            lb = -300
        if ub >= 0.999e+20:
            ub = 300
            
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
        
        copy.var_attributes = np.copy(self.var_attributes)
        copy.cons_block_idxs = self.cons_block_idxs[:]
        
        return copy
   


    
class BipartiteGraphPairData(torch_geometric.data.Data):
    """
    This class encode a pair of node bipartite graphs observation 
    """
    def __init__(self, variable_features_s=None, constraint_features_s=None, edge_indices_s=None, edge_features_s=None, 
                 variable_features_t=None, constraint_features_t=None, edge_indices_t=None, edge_features_t=None, 
                 y=None):
        
        super().__init__()
        
        self.variable_features_s, self.constraint_features_s, self.edge_index_s, self.edge_attr_s  =  (
            variable_features_s, constraint_features_s, edge_indices_s, edge_features_s)
        
        self.variable_features_t, self.constraint_features_t, self.edge_index_t, self.edge_attr_t  = (
            variable_features_t, constraint_features_t, edge_indices_t, edge_features_t)

        
        self.y = y
        

   
    def __inc__(self, key, value, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs 
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == 'edge_index_s':
            return torch.tensor([[self.variable_features_s.size(0)], [self.constraint_features_s.size(0)]])
        elif key == 'edge_index_t':
            return torch.tensor([[self.variable_features_t.size(0)], [self.constraint_features_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)





        
    
    
def normalize_graph(variable_features, constraint_features, edge_index, edge_attr):
    

    #Normalize variable bounds to value between 0,1
    vars_to_normalize = torch.where( torch.max(torch.abs(variable_features[:, :2]), axis=1)[0] > 1)[0]

    coeffs = torch.max(torch.abs(variable_features[vars_to_normalize, :2]) , axis=1)[0]
    
    for v, cf in zip(vars_to_normalize, coeffs):
        
        #normaize feature bound
        variable_features[ v, :2] /= cf
        
        #update obj coeff and associated edges
        variable_features[ v, 2 ] *= cf 
        
        associated_edges = torch.where(edge_index[0] == v)[0]
        edge_attr[associated_edges] *= cf
        
    
    
    #Normalize constraints 
    for c in range(constraint_features.shape[0]):
        
        associated_edges =  torch.where(edge_index[1] == c)[0]
        normalizer = max(torch.max(torch.abs(edge_attr[associated_edges]), axis=0)[0], torch.abs(constraint_features[c]))
        
        #normalize associated edges
        edge_attr[associated_edges] /= normalizer
        
        #normalize right hand side
        constraint_features[c] /= normalizer
    
    #normalize objective
    normalizer = torch.max(torch.abs(variable_features[:,2]), axis=0)[0]
    variable_features[:,2] /= normalizer


    return variable_features, constraint_features, edge_index, edge_attr
    
