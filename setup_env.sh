
conda deactivate 

conda env remove -n l2sn 

conda env create -f env.yml

conda activate l2sn

pip install ./pyscipopt
pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

#CPU

#pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cpu.html

#GPU
pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install --no-index torch-sparse -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install --no-index torch-cluster -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install --no-index torch-spline-conv -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install torch-geometric


#cuda=10.2

