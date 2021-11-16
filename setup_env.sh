
conda deactivate 

conda env remove -n l2sn 

conda env create -f env.yml

conda activate l2sn

pip install ./pyscipopt
pip install torch torchvision torchaudio

#CPU

#pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cpu.html

#GPU
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu102.html



#cuda=10.2
