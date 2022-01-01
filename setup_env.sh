
conda deactivate 

conda env remove -n l2sn 

conda env create -f env.yml

conda activate l2sn

pip install ./pyscipopt
pip install torch torchvision torchaudio

#CPU

#pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cpu.html

#GPU
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install torch-geometric


#cuda=10.2

