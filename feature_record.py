from oracle_maxime import OracleNodeSelector
from feature_recorder_util import NodeFeatureRecorder 
from pyscipopt import Model, Nodesel
from pathlib import Path
instances = Path('LP/instances1/').glob("*.lp")
# instance = "/Users/work/Desktop/LP/instances/er_n=124_m=4524_p=0.60_SET2_setparam=100.00_alpha=0.50_0.lp"
for instance in instances: 
    nodeselector = 'estimate'
    # print(f"instance {instance}")
    sol_file = str(instance).strip(".") + "_solution.txt"
    try:
        f = open(sol_file)
        # Do something with the file
    except FileNotFoundError:
        m = Model()
        m.hideOutput()
        m.readProblem(str(instance))
        m.optimize()
        m.writeBestSol(str(instance) + "_solution.txt")
        sol_file = str(instance).strip(".") + "_solution.txt"

    m = Model()
    m.hideOutput()
    m.readProblem(instance)
    solution = m.readSolFile(sol_file)
    m.includeNodesel(OracleNodeSelector(solution, nodeselector, instance), f"py_{nodeselector}", "", 666666, 666666)
    m.optimize()
    m.freeProb()

