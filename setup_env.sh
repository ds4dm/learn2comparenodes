conda init bash
#conda update -n base -c defaults conda

conda env remove -n l2sn 

conda env create -f env.yml

conda install pyg -c pyg -c conda-forge -n l2sn

source activate l2sn
pip install ./pyscipopt

