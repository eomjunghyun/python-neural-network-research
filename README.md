# python-neural-network-research
Research on the Mathematical Modeling and Internal Interpretation of Neural Network

## Windows Quick Start (Python 3.13 + venv)

Run all commands from the project root:

```powershell
# 1) Create and activate virtual environment
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1

# 2) Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Register Jupyter kernel
python -m ipykernel install --user --name nn-research --display-name "Python (nn-research)"

# 4) Start notebook
jupyter notebook
```

Open `notebooks/2026-03-11/0324_exp1.ipynb` and choose kernel `Python (nn-research)`.
