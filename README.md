<img src="FastKAST.png" alt="icon" width="100"/>

# **Fast Model-X Kernel-based Set Testing Toolkits**
[**PyPI Package**](https://pypi.org/project/fast-kernel-set-test)

This project integrates both [**FastKAST**](https://www.nature.com/articles/s41467-023-40346-2) and [**QuadKAST**](https://genome.cshlp.org/content/early/2024/08/29/gr.279140.124) to enhance software flexibility and usability. 

If the users intend to use a data format similar to ours, we further provide detailed instructions on each specific method pipeline in the sub-branches.

##  **Table of Contents**
1. [Installation](#Installation) 
2. [Basic usage](#Basic_usage) 
   -  [Hypothesis testing](#hypertest)
   -  [FastKAST](#FastKAST)
   - [QuadKAST](#QuadKAST)
   - [SKAT](#SKAT)
   - [Customized kernels](#custom)
3. [Data Availability](#data-availability)


## Installation <a name="Installation"></a>
1. You need python >= 3.60 in order to run the code (anaconda3 recommended)
2. `pip install fast-kernel-set-test` or install from source

You can either follow the standard pipeline `FastKAST_annot.py` and `QuadKAST_annot.py`, or import the neccessary function to build based on your own I/O.

## Basic usage <a name="Basic_usage"></a>

### Hypothesis Testing <a name="hypertest"></a>
The hypothesis testing functions offer flexible usage without extra data format or processing dependencies. 

* Single trait analysis
```
### Given covariates c: (NxM), input Z: (NxD), and output y: (Nx1)
from FastKAST.Compute.est import getfullComponentPerm
results = getfullComponentPerm(c,Z,y,Perm=10)
## results: {'pval': [obs_pval, perm_pval1, ..., perm_pval10]}     
```

* Multi-traits analysis
```
## Given covariates c: (NxM), input Z: (NxD), and output y: (NxK)
from FastKAST.Compute.est import getfullComponentMulti
results = getfullComponentMulti(c,Z,y)
## results: {'pval': [obs_pval1, obs_pval2, ..., obs_pvalK]}     
```


### FastKAST function <a name="FastKAST"></a>
FastKAST by default adapts the rbf kernel testing.

```
from FastKAST.methods.fastkast import FastKASTComponent
## Construct object
fastkast_component = FastKASTComponent(X, Z, y, MapFunc='rbf', D=50)
## Execution
results = fastkast_component.run()
```


### QuadKAST function <a name="QuadKAST"></a>
QuadKAST by default adapts the quadratic only kernel testing (e.g., in the absence of additive component).

```
from FastKAST.methods.fastkast import FastKASTComponent
## Construct object
fastkast_component = FastKASTComponent(X, Z, y, MapFunc='quadOnly') ## D is unused under explicit kernel construction
## Execution
results = fastkast_component.run()
```


### SKAT function <a name="SKAT"></a>
SKAT by default adapts the linear kernel testing. 

```
from FastKAST.methods.fastkast import FastKASTComponent
## Construct object
fastkast_component = FastKASTComponent(X, Z, y, MapFunc='linear') 
## Execution
results = fastkast_component.run()
```

### Customized kernel <a name="custom"></a>
The users are allowed to construct customized kernel based on their own need. The customized kernel is supposed to take the input data and transform to another matrix in the mapped dimension. 

```
### Test customized kernel
    def mapping(Z):
        mapping_func = PolynomialCountSketch(n_components=50)
        Z = mapping_func.fit_transform(Z)
        return Z
## Construct object
fastkast_component = FastKASTComponent(X, Z, y, mapping=mapping)
## Execution
results = fastkast_component.run()
```

## Data availability<a name="data-availability"></a>
The detailed statistics used to generate the main table and the Venn diagram of the paper are provided in the `Data` folder

✅ Efficient multi-traits analysis (Sep 30, 2024)

✅ Refactor the function into class (Dec 28, 2024)
