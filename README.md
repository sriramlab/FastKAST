<a href="https://zenodo.org/badge/latestdoi/429674106"><img src="https://zenodo.org/badge/429674106.svg" alt="DOI"></a>

<img src="FastKAST.png" alt="icon" width="100"/>

# Fast Model-X Kernel-based Set Testing Toolkits
https://pypi.org/project/fast-kernel-set-test/0.1.0/


This folder has been updated with both the [FastKAST](https://www.nature.com/articles/s41467-023-40346-2) and [QuadKAST](https://genome.cshlp.org/content/early/2024/08/29/gr.279140.124)

Please check sub-branch for detailed instruction on each specific method. 

# Table of contents:
1. [Installation](#Installation) 
2. [Basic usage](#Basic_usage) 
    1. [FastKAST](#FastKAST)
    2. [QuadKAST](#QuadKAST)
3. [Useful functions](#Functions)


## Installation <a name="Installation"></a>
1. You need python >= 3.60 in order to run the code (anaconda3 recommended)
2. `pip install fast-kernel-set-test` or install from source

You can either follow the standard pipeline `FastKAST_annot.py` and `QuadKAST_annot.py`, or import the neccessary function to build based on your own I/O.

## Basic usage <a name="Basic_usage"></a>

### FastKAST <a name="FastKAST"></a>
To run the demo FastKAST code with a customized window size, you can generate a annotation file with "start_index end_index" as a row, and run
```
python FastKAST_annot.py --bfile ./example/sim --phen ./example/sim.pheno --annot ./example/sim.new.annot
```
Or directly run
```
sh run_rbf_annot.sh
```

### QuadKAST <a name="QuadKAST"></a>
To run the demo QuadKAST code with a customized window size, you can generate a annotation file with "start_index end_index" as a row, and run
```
python QuadKAST_annot.py --bfile ./example/sim --phen ./example/sim.pheno --annot ./example/sim.new.annot
```
Or directly run
```
sh run_quad_annot.sh
```

## Useful functions <a name="Functions"></a>
* Single trait analysis
```python
## Given covariates c: (NxM), input Z: (NxD), and output y: (Nx1)
from FastKAST import getfullComponentPerm
results = getfullComponentPerm(c,Z,y,Perm=10)
## results: {'pval': [obs_pval, perm_pval1, ..., perm_pval10]}     
```
* Multi-traits analysis
```python
## Given covariates c: (NxM), input Z: (NxD), and output y: (NxK)
from FastKAST import getfullComponentMulti
results = getfullComponentMulti(c,Z,y)
## results: {'pval': [obs_pval1, obs_pval2, ..., obs_pvalK]}     
```

## Data availability
The detailed statistics used to generate the main table and the Venn diagram of the paper are provided in the `Data` folder

✅ Efficient multi-traits analysis (Sep 30, 2024)
