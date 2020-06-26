from oracle_maxime import OracleNodeSelector
from feature_recorder_util import NodeFeatureRecorder 
from pyscipopt import Model, Nodesel

instance = "/Users/work/Desktop/LP/gisp/er_n=57_m=176_p=0.10_SET2_setparam=100.00_alpha=0.75_0.lp"
nodeselector = 'estimate'
print(f"instance {instance}")
sol_file = instance.strip(".") + "_solution.txt"
m = Model()
m.hideOutput()
m.readProblem(instance)
m.optimize()
m.writeBestSol(sol_file)
m.freeProb()
print(f"optimal solution written to {sol_file}")

m = Model()
m.hideOutput()
m.readProblem(instance)
solution = m.readSolFile(sol_file)
m.includeNodesel(OracleNodeSelector(solution, nodeselector), f"py_{nodeselector}", "", 666666, 666666)
m.optimize()
        
m.hideOutput()
m.optimize()
m.freeProb()

