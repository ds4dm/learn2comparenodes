#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:12:41 2022

@author: aglabassi
"""



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
        file.write("".join([f" + {clause[1]}c_{idx}" for idx,clause in enumerate(clauses) ]))

        
        
        file.write("\nSubject to\n")

        

        for idx,clause in enumerate(clauses):
            varrs = clause[0].split(',')
            
            neg_varrs = []
            pos_varrs = []
            
            for var in varrs:
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
                    
                        
                    
                    
            
            file.write(f"clause_{idx}:" + ''.join([ f" + {yi}" for yi in pos_varrs]) + ''.join([ f" + 1 - {yi}" for yi in neg_varrs]) + f' >= c_{idx}\n')
            
            

                
  
        file.write("\nBinaries\n")
        for idx in range(len(clauses)):
            file.write(f" c_{idx}")
            
        for var_name in var_names.keys():
            file.write(f" {var_names[var_name]}")
            
        file.write('\nEnd\n')
        file.close()
        
clauses = [ ('A1,-A2', 2) , ('-C,-A1,A2',1) ]
file_name = '34.lp'

write_lp(clauses, file_name)