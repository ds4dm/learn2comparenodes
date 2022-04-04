#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:12:41 2022

@author: aglabassi
"""

import numpy as np
import pyscipopt.scip as sp

#Ramon Bejar
#https://computational-sustainability.cis.cornell.edu/cpaior2013/pdfs/ansotegui.pdf
def get_clauses(rng,H,W, n_piece, n_obstacle):

    pieces = [(rng.randint(1,H), rng.randint(1,W)) for _ in range(n_piece) ]
    
    #Generate obstacles, aka filled pieces
    obstacles = []
    while(len(obstacles) != n_obstacle):
        candidat = (rng.randint(0,H+1), rng.randint(0,W+1))
        if candidat not in obstacles:
            obstacles.append(candidat)
            
    piece_and_obstacles = pieces + obstacles

        
    
    #Soft clauses
    clauses = [(f'x{k}', piece[0]*piece[1]) for k,piece in enumerate(pieces)]
    
    

    #Hard clauses
    #No  two times same row/colomn per piece
    for k,piece in enumerate(pieces):
        h,w = piece
        clauses.append((f'-x{k},' + ','.join([ f'r{k}_{h+i}' for i in range(0,H-h+1)  ]),np.inf))

        for i in range(h, H+1):
            for j in range(i+1, H+1):
                clauses.append((f'-x{k},-r{k}_{i},-r{k}_{j}',np.inf))

        
        clauses.append((f'-x{k},' + ','.join([ f'c{k}_{i}' for i in range(0,W-w+1)  ]), np.inf))

        for i in range(0, W-w+1):
            for j in range(i+1,W-w+1):
                clauses.append((f'-x{k},-c{k}_{i},-c{k}_{j}',np.inf))

    #Place obstacles
    for k,piece in enumerate(obstacles):
        i,j = piece
        reajusted_k = len(pieces) + k
        clauses.append((f'r{reajusted_k}_{i}', np.inf))
        clauses.append((f'c{reajusted_k}_{j}', np.inf))
        clauses.append((f'x{reajusted_k}', np.inf))

    #No overlapping between pieces(and obstacles)
    for s,piece1 in enumerate( piece_and_obstacles ):
        h,w = piece1

        for k in range(s+1, len( piece_and_obstacles)):

            for i in range(h, H+1):
                for j in range(w, W+1):
                    for l in range(i-h+1, i+1):
                        for m in range(j,j+w):
                            clauses.append((f'-r{s}_{i},-c{s}_{j},-r{k}_{l},-c{k}_{m}', np.inf))


    

    return clauses



def write_lp(clauses, filename):
    
    ''' 
        clauses (in conj normal form )  : list of clauses to be "and-ed" with their weiths
        
        Clause  : string representing a conjunctive close, variable seperated by ',', 
        negation of variable  represented by -.
        
        
        
        Ex : 2*(A1 or not(A2)) and 1*(not(C)) == [ ('A,-A2', 2) , ('-C',1) ]
        
        '''
    
    var_names = dict() #maps var appearing in order i in clauses (whatever the name ) to y_i
    counter = 0

    with open(filename, 'w') as file:
        file.write("maximize\nOBJ:")
        file.write("".join([f" +{clause[1]}c_{idx}" for idx,clause in enumerate(clauses) if clause[1] < np.inf ]))

        
        
        file.write("\n\nSubject to\n")
    
        for idx,clause in enumerate(clauses):
            varrs = clause[0].split(',')
            
            neg_varrs = []
            pos_varrs = []
            
            for var in varrs:
                if var != '':
                    if var[0] == '-':
                        if var[1:] not in var_names:
                            var_names[var[1:]] = f"y_{counter}"
                            counter += 1
                        neg_varrs.append(var_names[var[1:]])
                        
                    else:
                        if var[0:] not in var_names:
                            var_names[var[0:]] = f"y_{counter}"
                            counter += 1
                        pos_varrs.append(var_names[var[0:]])
                        
                            
            last_part = f' +c_{idx} <= {len(neg_varrs)} \n' if clause[1] < np.inf else f' <= {len(neg_varrs) - 1} \n'           
                    
            
            file.write(f"clause_{idx}:" + ''.join([ f" -{yi}" for yi in pos_varrs]) + ''.join([ f" +{yi}" for yi in neg_varrs]) + last_part) 
                       
            
            

                
  
        file.write("\nBinaries\n")
        for idx in range(len(clauses)):
            if clauses[idx][1] < np.inf:
                file.write(f" c_{idx}")
            
        for var_name in var_names.keys():
            file.write(f" {var_names[var_name]}")
            
        file.write('\nEnd\n')
        file.close()


seed = 0
rng = np.random.RandomState(seed)
        
clauses = get_clauses(rng,10,10,100,0) 
file_name = '34.lp'


write_lp(clauses, file_name)
m = sp.Model()
m.readProblem(file_name)
m.optimize()
