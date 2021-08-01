from pathlib import Path 
import pyscipopt.scip as sp
from node_selectors.classic_selectors import Default_DFS, Default_BFS
from node_selectors.oracle_selectors import OracleNodeSelector




# list_files_bounded = Path('/Users/work/Desktop/Learn2SelectNodes/mik.data/bounded/').rglob('*.mps.gz')    # 90 bounded files
list_files_setcover = Path('/Users/work/Desktop/learn2branch/data/instances/setcover/transfer_500r_1000c_0.05d/').glob("*.lp")
gisp_instance  = "data_generation/GISP/er_n=99_m=2899_p=0.60_SET2_setparam=100.00_alpha=0.50_5.lp"


model = sp.Model()


for node_sel, node_sel_name in [(Default_DFS, "dfss"), (Default_BFS, "bfss")]:
    
    model.readProblem(gisp_instance)
    
    model.includeNodesel(node_sel(), node_sel_name, "", 100,100)
    model.optimize()
    #print("Complexity of using " + " node selection method mesured by number of node created by b&b " + model.getNNodes())
    
    



