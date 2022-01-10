#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 17:16:39 2021

@author: abdel

Contains utilities to save and load comparaison behavioural data


"""

from scipy.sparse import csr_matrix
from scipy import sparse
import torch
import torch_geometric
import os.path as osp
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        
        
        var_attributes0, cons_block_idxs0 = graphidx2graphdata[g0_idx]
        var_attributes1, cons_block_idxs1 = graphidx2graphdata[g1_idx]
        
        g_data = CompFeaturizer._get_graph_pair_data(var_attributes0, 
                                                         var_attributes1, 
                                                         
                                                         cons_block_idxs0, 
                                                         cons_block_idxs1, 

                                                         all_conss_blocks, 
                                                         all_conss_blocks_features, 
                                                         comp_res)
        
        data = self._to_tensors(g_data)
        
        return data
        
        
    
    def save_comp(self, model, node1, node2, comp_res, comp_id):
        
        self.LP_feature_recorder.record_sub_milp_graph(model, node1)
        self.LP_feature_recorder.record_sub_milp_graph(model, node2)
        
        graphidx2graphdata = self.LP_feature_recorder.recorded_light
        all_conss_blocks = self.LP_feature_recorder.all_conss_blocks
        all_conss_blocks_features = self.LP_feature_recorder.all_conss_blocks_features
        
        g0_idx, g1_idx, comp_res = node1.getNumber(), node2.getNumber(), comp_res
        
        
        var_attributes0, cons_block_idxs0 = graphidx2graphdata[g0_idx]
        var_attributes1, cons_block_idxs1 = graphidx2graphdata[g1_idx]
        
        g_data = CompFeaturizer._get_graph_pair_data(var_attributes0,
                                                    var_attributes1, 
                                                    
                                                    cons_block_idxs0, 
                                                    cons_block_idxs1, 

                                                    all_conss_blocks, 
                                                    all_conss_blocks_features, 
                                                    comp_res)
        
        file_path = osp.join(self.save_dir, f"{self.instance_name}_{comp_id}.pt")
        data = self._to_tensors(g_data)
        torch.save(data, file_path, _use_new_zipfile_serialization=False)
        
        return self
    
    def _to_tensors(self, g_data):

        variable_features = g_data[0]
        constraint_features = g_data[1]
        edge_indices = g_data[2]
        edge_features = g_data[3]
        y = g_data[4]

        g1 = normalize_graph(variable_features[0], constraint_features[0], edge_indices[0], edge_features[0])
        g2 = normalize_graph(variable_features[1], constraint_features[1], edge_indices[1], edge_features[1])
        
        return BipartiteGraphPairData(*g1, *g2, y)
        
        
        
   
    
    def _get_graph_pair_data( var_attributes0, var_attributes1, cons_block_idxs0, cons_block_idxs1, all_conss_blocks, all_conss_blocks_features, comp_res ):
        
        adjacency_matrixes0 = map(all_conss_blocks.__getitem__, cons_block_idxs0)
        adjacency_matrixes1 = map(all_conss_blocks.__getitem__, cons_block_idxs1)
        
        cons_attributes_blocks0 = map(all_conss_blocks_features.__getitem__, cons_block_idxs0)
        cons_attributes_blocks1 = map(all_conss_blocks_features.__getitem__, cons_block_idxs1)
        
        adjacency_matrix0 = torch.hstack(tuple(adjacency_matrixes0)) #n_var x card(conss )
        adjacency_matrix1 = torch.hstack(tuple(adjacency_matrixes1)) 
        
        cons_attributes0 = torch.vstack(tuple(cons_attributes_blocks0)) #card(conss) X cons_dim
        cons_attributes1 = torch.vstack(tuple(cons_attributes_blocks1))
        
        
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
        self.obj_adjacency  == None

   
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
                self._add_conss_to_graph(graph, model, sub_milp.getAddedConss())
                self._change_branched_bounds(graph, sub_milp)
            
            self._add_scip_estimate_cons(sub_milp, graph)
            
            self.recorded[sub_milp.getNumber()] = graph
            self.recorded_light[sub_milp.getNumber()] = (graph.var_attributes, 
                                                         graph.cons_block_idxs)
    
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

        cons_attributes = torch.FloatTensor(len(conss), graph.d1, device=DEVICE)
        var_idxs = []
        cons_idxs = []
        weigths = []
        for cons_idx, cons in enumerate(conss):

            cons_attributes[cons_idx] =  self._get_feature_cons(model, cons)
          
            for var, coeff in model.getValsLinear(cons).items():
                var_idxs.append(self.var2idx[str(var)] )
                cons_idxs.append(cons_idx)
                weigths.append(coeff)


        adjacency_matrix =  torch.sparse_coo_tensor([var_idxs, cons_idxs], weigths, (self.n0, len(conss))) 
        
        #add idx to graph
        graph.cons_block_idxs.append(len(self.all_conss_blocks_features)) #carreful with parralelization
        
        #add appropriate structure to self
        self.all_conss_blocks_features.append(cons_attributes)
        self.all_conss_blocks.append(adjacency_matrix)

    def _get_obj_adjacency(self, model):
    
       if self.obj_adjacency  == None:
           var_coeff = { self.var2idx[ str(t) ]:c for (t,c) in model.getObjective().terms.items() if c != 0.0 }
           var_idxs = list(var_coeff.keys())
           weigths = list(var_coeff.values())
           cons_idxs = [0]*len(var_idxs)
    
           self.obj_adjacency =  torch.sparse_coo_tensor([var_idxs, cons_idxs], weigths, (self.n0, 1))
      
       return self.obj_adjacency         
       
    
    def _add_scip_estimate_cons(self, sub_milp, graph):
        adjacency_matrix = self._get_obj_adjacency()
        cons_feature = torch.FloatTensor([[sub_milp.getEstimate()]], device=DEVICE)
        graph.cons_block_idxs.append(len(self.all_conss_blocks_features))
        self.all_conss_blocks_featureså.append(cons_feature)
        self.all_conss_blocks.append(adjacency_matrix)

    def _change_branched_bounds(self, graph, sub_milp):
        
        bvars, bbounds, btypes = sub_milp.getParentBranchings()
        
        for bvar, bbound, btype in zip(bvars, bbounds, btypes): 
            var_idx = self.var2idx[str(bvar)]
            graph.var_attributes[var_idx, int(btype) ] = bbound
            
        
    
    def _get_feature_cons(self, model, cons):
        rhs = model.getRhs(cons)
        return torch.FloatTensor([ rhs ], device=DEVICE)
    
    def _get_feature_var(self, model, var):
        
        lb, ub = var.getLbOriginal(), var.getUbOriginal()
        
        if lb <= - 0.999e+20:
            lb = -300
        if ub >= 0.999e+20:
            ub = 300
            
        objective_coeff = model.getObjective()[var]
        
        binary, integer, continuous = self._one_hot_type(var)
    
        
        return torch.FloatTensor([ lb, ub, objective_coeff, binary, integer, continuous ], device=DEVICE)
    
    
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
    def __init__(self, n0, d0=6, d1=1, allocate=True):
        
        self.n0, self.d0, self.d1 = n0, d0, d1
        
        if allocate:
            self.var_attributes = torch.zeros(n0,d0)
            self.cons_block_idxs = []
        else:
            self.var_attributes = None
            self.cons_block_idxs = None
    
    
    def copy(self):
        
        copy = BipartiteGraphStatic0(self.n0, allocate=False)
        
        copy.var_attributes = self.var_attributes.clone()
        copy.cons_block_idxs = self.cons_block_idxs[:]
        
        return copy        
    
    
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
    

