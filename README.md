# The Gumbel-max trick
### Application to hard sampling problems

This repository contains python code to run experiments appearing in thesis.

#### Install
<!-- To install the required python version, install [pyenv](github.com/pyenv/pyenv) and then run:
```python
pyenv install
``` -->

To create a new environment and install the requirements used in the code, run:
```python
pip3 install -r requirements.txt
```

#### Compile
To compile the cython module `permanent` used in the code, run:
```python
python3 setup.py build_ext --inplace
```

#### Run experiments and plot results
To run the code to plot the graphs used in the thesis, run:
```python
source venv/bin/activate
python3 main.py
```