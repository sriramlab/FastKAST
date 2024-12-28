# A Scalable Adaptive Quadratic Kernel Method for Interpretable Epistasis Analysis in Complex Traits

QuadKAST extends the FastKAST software by incorporating quadratic families, enhancing its functionality. Building on the scalability advantages of FastKAST, QuadKAST introduces robust support for quantifying epistasis strength in an interpretable manner.

## Requirements
1. You need python >= 3.60 in order to run the code (anaconda3 recommended)
2. Run ```pip install -r requirements.txt``` to install the required python package
3. (Optional) Install Julia and the FameSVD package [Li et al, 2019]. If FameSVD is not installed, SVD computation will reverse back to scipy package


QuadKAST receives the inpute genotype and phenotype, output p value.
```
* Input argument
 * ----------
 * bfile : prefix name of plink file (.bim, .bed, .fam)
 *     N x M Genotype matrix
 * covar : (Optional) Linear covariates features to exclude. (order should be the same as genotype data)
 *     Chi-squared distributions.
 * phen : plink file, space delimiter, fifth column is the phenotype value
 *     Phenotype need to be standardized
 * thread: int
 *     Number of thread to be use (default is 1)
 * sw: int
 *     Number of neighboring window included to totally regress out linear effect (default is 2)
 * annot: file
 *     Annotation file -- M x K boolean matrix. M: feature number; K: number of set tested. 1 indicates the inclusion of a feature.
 * filename: string
 *     Output file name
 * test: string
 *     Type of kernel test to use
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

## Example
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
python QuadKAST_annot.py --bfile ./example/sim --phen ./example/sim.pheno --annot ./example/sim.annot
```
Or directly run
```
sh run_annot.sh
```

## Data availability
The detailed statistics used to generate the main table and the Venn diagram of the paper are provided in the `Data` folder
