
from pyscipopt import Nodesel


class Default_Estimate(Nodesel):
    def nodeselect(self):
        node = self.model.executeNodeSel('estimate')
        return {"selnode": node}
    def nodecomp(self, node1, node2): 
        node = self.model.executeNodeComp('estimate', node1, node2)


class Default_BFS(Nodesel):
    def nodeselect(self):
        node = self.model.executeNodeSel('bfs')
        return {"selnode": node}

class Default_Breadth(Nodesel):
    def nodeselect(self):
        node = self.model.executeNodeSel('breadthfirst')
        return {"selnode": node}

class Default_DFS(Nodesel):
    def nodeselect(self):
        node = self.model.executeNodeSel('dfs')
        return {"selnode": node}

class Default_Hybrid(Nodesel):
    def nodeselect(self):
        node = self.model.executeNodeSel('hybridestim')
        return {"selnode": node}

class Default_UCT(Nodesel):
    def nodeselect(self):
        node = self.model.executeNodeSel('uct')
        return {"selnode": node}

class Default_Restart(Nodesel):
    def nodeselect(self):
        node = self.model.executeNodeSel('restartdfs')
        return {"selnode": node}


class Oracle(Nodesel):
    def __init__(self, solution_path): 
        self.opt_checked = {} 
        self.opt_nodes = {}
        self.feat = {'maxdepth':[]}
        self.optfeat = {'maxdepth':[]}
        self.optnodenumber = -1
        self.negate = True
        self.solution_path = solution_path
        
    def nodeinit(self): 
        self.feat['maxdepth'].append(self.model.getNBinVars() + self.model.getNintVars())
        self.optfeat['maxdepth'].append(self.model.getNBinVars() + self.model.getNintVars())
        self.opt_sol = self.model.readSolFile(self.solution_path)

    def nodeselect(self): 
        leaves, children, siblings = self.model.getOpenNodes()
        # print(type(leaves), type(children), type(siblings))
        self.optchild = -1
        for idx, child in enumerate(children):
            # print("loop activated")
            if child.getNumber() not in self.opt_checked: 
                parent = child.getParent()
                # root -> trivially optimal 
                if(parent.getDepth() > 0 and not self.opt_nodes[parent.getNumber()]): 
                    optimal = True   
                nbranchvars = child.getNParentBranchings()
                # print(nbranchvars)
                branchvars, branchbounds, boundtypes = child.getParentBranchings()
                optimal = True 
                for i in range(nbranchvars):
                    optval = self.model.getSolVal(self.opt_sol, branchvars[i])
                    # print(optval)
                    if boundtypes[i] == 0 and optval < branchbounds[i] or boundtypes[i] == 1 and optval > branchbounds[i]: 
                        optimal = False 
                self.opt_checked[child.getNumber()] = True
                self.opt_nodes[child.getNumber()] = optimal 
            
            if child.getNumber() in self.opt_nodes:
                self.optnodenumber = child.getNumber()
                self.optchild = idx          
                # print(self.optnodenumber)
                # print(self.optchild)

            #TODO: Lines 461 through to 503 of the nodesel_oracle.c file need to implemented here. 

        node = self.model.getBestNode()
        return {"selnode": node}
        
    def nodecomp(self, node1, node2): 
        isoptimal1 = self.opt_nodes[node1.getNumber()]
        isoptimal2 = self.opt_nodes[node2.getNumber()]
        if(isoptimal1):
            assert(not isoptimal2)
            return -1
        elif isoptimal2:
            return 1
        else:
            depth1 = node1.getDepth()
            depth2 = node2.getDepth()
            if (depth1 > depth2):
                return -1
            elif (depth1 < depth2):
                return 1
            else:
                lowerbound1 = node1.getLowerbound()
                lowerbound2 = node2.getLowerbound()
                if(self.model.isLT(lowerbound1, lowerbound2)):
                    return -1
                elif(self.model.isGT(lowerbound1, lowerbound2)):
                    return 1
                # Why check the same thing twice? What is the purpose/benefits?
                if lowerbound1 < lowerbound2:
                    return -1
                elif lowerbound1 > lowerbound2:
                    return 1
                else:
                    return 0


class OracleNodeSel(Nodesel):
    def __init__(self, solution_path): 
        self.opt_checked = {} 
        self.opt_nodes = {}
        self.feat = {'maxdepth':[]}
        self.optfeat = {'maxdepth':[]}
        self.optnodenumber = -1
        self.negate = True 
        self.solution_path = solution_path       

    def nodeinit(self): 
        self.feat['maxdepth'].append(self.model.getNBinVars() + self.model.getNintVars())
        self.optfeat['maxdepth'].append(self.model.getNBinVars() + self.model.getNintVars())
        self.opt_sol = self.model.readSolFile(self.solution_path)

    
    def nodeselect(self): 
        leaves, children, siblings = self.model.getOpenNodes()
        # print(type(leaves), type(children), type(siblings))
        self.optchild = -1
        for idx, child in enumerate(children):
            # print("loop activated")
            if child.getNumber() not in self.opt_checked: 
                parent = child.getParent()
                # root -> trivially optimal 
                if(parent.getDepth() > 0 and not self.opt_nodes[parent.getNumber()]): 
                    optimal = True   
                nbranchvars = child.getNParentBranchings()
                # print(nbranchvars)
                branchvars, branchbounds, boundtypes = child.getParentBranchings()
                optimal = True 
                for i in range(nbranchvars):
                    optval = self.model.getSolVal(self.opt_sol, branchvars[i])
                    # print(optval)
                    if boundtypes[i] == 0 and optval < branchbounds[i] or boundtypes[i] == 1 and optval > branchbounds[i]: 
                        optimal = False 
                self.opt_checked[child.getNumber()] = True
                self.opt_nodes[child.getNumber()] = optimal 
            
            if child.getNumber() in self.opt_nodes:
                self.optnodenumber = child.getNumber()
                self.optchild = idx

            #TODO: Lines 461 through to 503 of the nodesel_oracle.c file need to implemented here. 

        node = self.model.getBestNode()
        return {"selnode": node}



