from pyscipopt import  Nodesel



class FiFo(Nodesel):

  def nodeselect(self):
    '''first method called in each iteration in the main solving loop. '''

    leaves, children, siblings = self.model.getOpenNodes()
    nodes = leaves + children + siblings

    return {"selnode" : nodes[0]} if len(nodes) > 0 else {}

  def nodecomp(self, node1, node2):
    '''
    compare two leaves of the current branching tree
    It should return the following values:
      value < 0, if node 1 comes before (is better than) node 2
      value = 0, if both nodes are equally good
      value > 0, if node 1 comes after (is worse than) node 2.
    '''
    return 0



class BreadthFirstSearch(Nodesel):
    
    def nodeselect(self):
        
        selnode = self.model.getPrioSibling()
        if selnode == None: #no siblings to be visited (all have been LP-solved), since breath first, 
        #we take the heuristic of taking the best leaves among all leaves
            
            selnode = self.model.getBestLeaf() #DOESTN INCLUDE CURENT NODE CHILD !
            if selnode == None: 
                selnode = self.model.getPrioChild()
        
        return {"selnode": selnode}
        
        
    
    def nodecomp(self, node1, node2): 
        
        d1, d2 = node1.getDepth(), node2.getDepth()
        
        if d1 == d2:
            #choose the first created node
            return node1.getNumber() - node2.getNumber()
        
        #less deep node => better
        return d1 - d2
        
        


class DepthFirstSearch(Nodesel):
    
    def nodeselect(self):
        
        selnode = self.model.getPrioChild()  #aka best child of current node
        if selnode == None:
            
            selnode = self.model.getPrioSibling() #if current node is a leaf, get 
            # a sibling
            if selnode == None: #if no sibling, just get a leaf
                selnode = self.model.getBestLeaf()
                
        #Prio == unqueue depending of priority assigned by branching ,
        #Best == unqueue depending of priority defined only by node selection strategy, i.e nodecomp
            
        return {"selnode": selnode}
    
    def nodecomp(self, node1, node2):
        return -node1.getDepth() + node2.getDepth()
    

class Default_Estimate(Nodesel):
    def nodeselect(self):
        node = self.model.get
        return {"selnode": node}
    def nodecomp(self, node1, node2): 
        node = self.model.executeNodeComp('estimate', node1, node2)
       