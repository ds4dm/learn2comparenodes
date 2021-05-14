

from pyscipopt import  Nodesel


#Work with custom pyscipot implementing model.getRootNode and model.executeNodeSel
#Nodesel herited object is 

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



class InternalNodeSelector(Nodesel):
    def __init__(self, nodesel_name):
        self.nodesel_name = nodesel_name

    def nodeselect(self):
        selnode = self.model.executeNodeSel(self.nodesel_name)
        return {"selnode": selnode}

    def nodecomp(self, node1, node2): 
        return self.model.executeNodeComp(self.nodesel_name, node1, node2)
