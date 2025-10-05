GENERAL SETUP STEPS FOR PYTHON ENVIRONMENT:
============================================

1) Install PyENV: https://github.com/pyenv/pyenv

2) Download / Make sure that PyEnv has instance of Python 3.10.1 environment
  => pyenv update
  => pyenv install 3.10.1

5) Create instance of Python 3.10.1 
  5.1) creates .python-version file containing the desired version number
    => pyenv local 3.10.1
  5.2) create the virtual Python environment
    => python -m venv .venv

6) Activate created virtual env  
  => (on Windows) .venv\scripts\activate
  => (on Linux)   source .venv/bin/activate

7) Install required packages from requirements.txt into .venv
  => pip install -r requirements.txt


=====================================================================================

WORK WITH THE PYTHON NOTEBOOK: 
==============================

From now on you can use the created Python environment for CQL and IQL

1) Make sure Python environment is activated 
  => (on Windows) .venv\scripts\activate
  => (on Linux)   source .venv/bin/activate

2) Start the job, specifying the CUDA_VISIBLE_DEVICES via GPU_IDS and the number of parallel processes per GPU via PROCS_PER_GPU.
  => GPU_IDS=0 PROCS_PER_GPU=10 python cql_iql.py
