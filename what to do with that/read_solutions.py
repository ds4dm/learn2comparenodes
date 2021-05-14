from pathlib import Path
from pyscipopt import Model, Nodesel, Eventhdlr
import pyscipopt
# import math
# import re
def nodeCheckOptimal(model, node, opt_sol, optimal_node):  
    parent = node.getParent()
    print(parent.getNumber())
    # root -> trivially optimal 
    if(parent.getDepth() > 0 and not optimal_node[parent.getNumber()]): 
        return True  
    nbranchvars = model.getNParentBranchings()
    branchvars, branchbounds, boundtypes = node.getParentBranchings()
    assert(nbranchvars >= 1) 
    for i in range(nbranchvars):
        optval = model.getSolVal(opt_sol, branchvars[i])
        if boundtypes[i] == 0 and optval < branchbounds[i] or boundtypes[i] == 1 and optval > branchbounds[i]: 
            return False 
    return True 

# THese are generators and hence are not subscriptable 
# You can however run loops over them 
list_solutions_bounded = Path('./mik.data/bounded/temp_sol/').rglob('*')
list_problems_bounded = Path('./mik.data/bounded/').rglob('*.mps.gz')

# root = '/Users/work/Desktop/Learn2SelectNodes/mik.data/'
problem_path = '/Users/work/Desktop/Learn2SelectNodes/mik.data/unbounded/mik.500-1-100.5.mps.gz'
solution_path = '/Users/work/Desktop/Learn2SelectNodes/mik.data/unbounded/temp_sol/mik.500-1-100.5.'
MAXDEPTH = 0

class Default(Eventhdlr):
    def __init__(self): 
        self.feat = {'maxdepth':[]}
        self.optfeat = {'maxdepth':[]}
        self.optnodenumber = -1
        self.negate = True
        self.traj_file = open('/Users/work/Desktop/Learn2SelectNodes/mik.data/trajfile', "a")
        self.weight_file = open('/Users/work/Desktop/Learn2SelectNodes/mik.data/trajfile.weight', "a")
        self.opt_checked = {} 
        self.opt_nodes = {}
        
    def eventinit(self):
        self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.NODEFOCUSED, self)

    def eventexit(self):
        self.model.dropEvent(pyscipopt.SCIP_EVENTTYPE.NODEFOCUSED, self)
    
    def eventexec(self, event):
        self.nodeselinit()
        self.nodeselselect()
        
    def nodeselinit(self): 
        self.feat['maxdepth'].append(self.model.getNBinVars() + self.model.getNintVars())
        self.optfeat['maxdepth'].append(self.model.getNBinVars() + self.model.getNintVars())
        self.opt_sol = self.model.readSolFile('/Users/work/Desktop/Learn2SelectNodes/mik.data/unbounded/temp_sol/mik.500-1-100.5.') 
    
    def nodeselselect(self): 
        leaves, children, siblings = self.model.getOpenNodes()
        self.optchild = -1
        print(len(children))
        print(len(leaves))
        print(len(siblings))
        for idx, child in enumerate(children):
            print("loop activated")
            if child.getNumber() not in self.opt_checked: 
                parent = child.getParent()
                print(parent.getNumber())
                # root -> trivially optimal 
                if(parent.getDepth() > 0 and not self.optimal_node[parent.getNumber()]): 
                    optimal = True   
                nbranchvars = self.model.getNParentBranchings()
                print(nbranchvars)
                branchvars, branchbounds, boundtypes = child.getParentBranchings()
                optimal = True 
                for i in range(nbranchvars):
                    optval = self.model.getSolVal(self.opt_sol, branchvars[i])
                    print(optval)
                    if boundtypes[i] == 0 and optval < branchbounds[i] or boundtypes[i] == 1 and optval > branchbounds[i]: 
                        optimal = False 
                self.opt_checked[child.getNumber()] = True
                self.opt_nodes[child.getNumber()] = optimal 
            
            if child.getNumber() in self.opt_nodes:
                self.optnodenumber = child.getNumber()
                self.optchild = idx          
                print(self.optnodenumber)
                print(self.optchild)

            #TODO: Lines 461 through to 503 of the nodesel_oracle.c file need to implemented here. 

            node = self.model.getBestNode()
            return {"selnode": node}


            

        

class Oracle(Nodesel):
    def __init__(self): 
        self.opt_checked = {} 
        self.opt_nodes = {}
        self.feat = {'maxdepth':[]}
        self.optfeat = {'maxdepth':[]}
        self.optnodenumber = -1
        self.negate = True 
        self.traj_file = open('/Users/work/Desktop/Learn2SelectNodes/mik.data/trajfile', "a")
        self.weight_file = open('/Users/work/Desktop/Learn2SelectNodes/mik.data/trajfile.weight', "a")
        self.feat['maxdepth'].append(self.model.getNBinVars() + self.model.getNintVars())
        self.optfeat['maxdepth'].append(self.model.getNBinVars() + self.model.getNintVars())
        self.opt_sol = self.model.readSolFile('/Users/work/Desktop/Learn2SelectNodes/mik.data/unbounded/temp_sol/mik.500-1-100.5.') 

    def nodeselect(self): 
        leaves, children, siblings = self.model.getOpenNodes()
        self.optchild = -1
        print(len(children))
        print(len(leaves))
        print(len(siblings))
        for idx, child in enumerate(children):
            print("loop activated")
            if child.getNumber() not in self.opt_checked: 
                parent = child.getParent()
                print(parent.getNumber())
                # root -> trivially optimal 
                if(parent.getDepth() > 0 and not self.optimal_node[parent.getNumber()]): 
                    optimal = True   
                nbranchvars = self.model.getNParentBranchings()
                print(nbranchvars)
                branchvars, branchbounds, boundtypes = child.getParentBranchings()
                optimal = True 
                for i in range(nbranchvars):
                    optval = self.model.getSolVal(self.opt_sol, branchvars[i])
                    print(optval)
                    if boundtypes[i] == 0 and optval < branchbounds[i] or boundtypes[i] == 1 and optval > branchbounds[i]: 
                        optimal = False 
                self.opt_checked[child.getNumber()] = True
                self.opt_nodes[child.getNumber()] = optimal 
            
            if child.getNumber() in self.opt_nodes:
                self.optnodenumber = child.getNumber()
                self.optchild = idx          
                print(self.optnodenumber)
                print(self.optchild)

            #TODO: Lines 461 through to 503 of the nodesel_oracle.c file need to implemented here. 

            node = self.model.getBestNode()
            return {"selnode": node}
        
    def nodecomp(self): 
        pass


m = Model()
# m.readProblem(root + 'bounded/' + str(problem.name))
# solution_path = root + 'bounded/temp_sol/' + str(problem.name)[:len(problem.name) - 6]
# m.includeNodesel(Default(), "dummy_sel", "Testing a node selector.", 1073741823, 536870911)
m.readProblem(problem_path)
event_record = Default()
m.includeEventhdlr(event_record, "NodeFocusHandler", "")
m.optimize()
sol = m.getBestSol()
# x = m.readSolFile(solution_path)
# nodeCheckOptimal(m, node, sol)
print(len(event_record.feat))
m.freeProb()

