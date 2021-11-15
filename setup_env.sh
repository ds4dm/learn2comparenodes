conda init bash
#conda update -n base -c defaults conda

source deactivate 

conda env remove -n l2sn 

conda env create -f env.yml

source activate l2sn

pip install ./pyscipopt

#CPU
#pip install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
#pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cpu.html

#GPU
pip install torch torchvision torchaudio
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu102.html



#cuda=10.2

