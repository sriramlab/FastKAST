# Fast Kernel-based Association Testing of non-linear genetic effectsfor Biobank-scale data

FastKAST is a highly scalable method for testing complex nonlinear relationships between a set of SNPs with a target trait. FastKAST uses a kernel function to model these non-linear relationships. Since direct computation using the kernel function is computationally prohibitive, FastKast uses insights from modern scalable machine learning to obtain an approximation that is efficient to compute. In particular, FastKAST is scalable to large sample size, and testing on 500K set with 30 snps can be done with a few minutes


## Requirements
1. You need python >= 3.60 in order to run the code (anaconda3 recommended)
2. Run ```pip install -r requirements.txt``` to install the required python package
3. (Optional) Install Julia and the FameSVD package [Li et al, 2019]. If FameSVD is not installed, SVD computation will reverse back to scipy package

## Exmaple
To run the demo code, you can run
```
python FastKAST.py --bfile ./example/sim --phen ./example/sim.pheno
```
or directly run
```
sh run.sh
```
