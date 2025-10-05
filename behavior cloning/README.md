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

## If numpy was installed in version >= 2.0 
9) Force numpy to be in version < 2.0
  => pip install numpy==1.26.4

=====================================================================================

WORK WITH THE PYTHON NOTEBOOK: 
==============================

From now on you can use the created Python environment for working on the ipynb. 

1) Make sure Python environment is activated 
  => (on Windows) .venv\scripts\activate
  => (on Linux)   source .venv/bin/activate

2) Start Python-Notebook
  => jupyter notebook
