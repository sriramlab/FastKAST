<a href="https://zenodo.org/badge/latestdoi/429674106"><img src="https://zenodo.org/badge/429674106.svg" alt="DOI"></a>

This project aims to merge the FastKAST with QuadKAST


# Fast Kernel-based Association Testing of non-linear genetic effects for Biobank-scale data


FastKAST is a highly scalable method for testing complex nonlinear relationships between a set of SNPs with a target trait. FastKAST uses a kernel function to model these non-linear relationships. Since direct computation using the kernel function is computationally prohibitive, FastKast uses insights from modern scalable machine learning to obtain an approximation that is efficient to compute. In particular, FastKAST is scalable to large sample size, and testing on 500K set with 30 snps can be done with a few minutes


## Requirements
1. You need python >= 3.60 in order to run the code (anaconda3 recommended)
2. Run ```pip install -r requirements.txt``` to install the required python package
3. (Optional) Install Julia and the FameSVD package [Li et al, 2019]. If FameSVD is not installed, SVD computation will reverse back to scipy package


FastKAST receives the inpute genotype and phenotype, output p value.
```
* Input argument
 * ----------
 * bfile : prefix name of plink file (.bim, .bed, .fam)
 *     Genotype matrix
 * covar : (Optional) Linear covariates features to exclude. (order should be the same as genotype data)
 *     Chi-squared distributions.
 * phen : plink file, space delimiter, fifth column is the phenotype value
 *     Phenotype need to be standardized
 * map : int
 *     Appriximated dimension factor (default is 50, which means 50*snps)
 * window : int
 *     The physical window size for each set
 * thread: int
 *     Number of thread to be use (default is 1)
 * sw: int
 *     Number of neighboring window included to totally regress out linear effect (default is 2)
 * dir: output directory
 * output: output file name
* Output: a list of list with the following information(pkl file)
 * pval: best p-value at each set
 * p_perm: 10 permuted p-value at each set (useful to learn the distribution when testing multiple hyperparameters)
 * FastKAST_times: running time at each set
 * Index: snp physical index starting point for each set
 * N: number of effective samples being tested
 * d: number of snps tested
 * chrom: chromosome id for each set
 * bgamma: best hyperparamter gamma
 
error:
negative/0 p value: out of precision level (default is 1e-14)
p value = 2: p value calculation error 
```

## Exmaple
To run the demo code with a fixed window size, you can run
```
python FastKAST_annot.py --bfile ./example/sim --phen ./example/sim.pheno
```
or directly run
```
sh run.sh
```

To run the demo code with a customized window size, you can generate a annotation file with "start_index end_index" as a row, and run
```
python FastKAST_annot.py --bfile ./example/sim --phen ./example/sim.pheno --annot ./example/sim.annot
```
Or directly run
```
sh run_annot.sh
```

## Data availability
The detailed statistics used to generate the main table and the Venn diagram of the paper are provided in the `Data` folder
