import numpy as np
import os, re
import argparse
import sys
import h5py
from sys import path as syspath
from os import path as ospath
import traceback
import time
import scipy
import pandas as pd
from tqdm import tqdm as tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
import multiprocessing
from bed_reader import open_bed
import pandas_plink
from pandas_plink import read_plink1_bin
from joblib import dump, load
sys.path.append('../')
from utils import *
from fastmle_res_jax import FastKASTRegression, getfullComponentPerm, Bayesian_Posterior



def LD_estimation(snp1, snps2, DEBUG=False):
    
    nanfilter1 = ~np.isnan(snp1).any(axis=1)
    nanfilter2 = ~np.isnan(snps2).any(axis=1)
    
    nanfilter = nanfilter1&nanfilter2
    snp1 = snp1[nanfilter]
    snps2 = snps2[nanfilter]
    
    p = np.mean(snp1) / 2
    q = np.mean(snps2,axis=0) / 2
    
    

    # Calculate observed frequency of both minor alleles
    # This requires identifying individuals with 1 or 2 for both SNPs and calculating the frequency
    observedXY = np.mean((snp1 / 2) * (snps2 / 2))

    # Calculate expected frequency under independence
    expectedXY = p * q

    # Calculate r^2
    r_squared = ((observedXY - expectedXY) ** 2) / (p * (1 - p) * q * (1 - q))
    if DEBUG:
        print(f'r square value is: {r_squared}')
    return r_squared


def paraCompute(args):
    start_index, end_index = args
    if end_index-start_index <=3:
        print(f'Warning: target window size too small, numerical issue may encounter')
    M = G.shape[1]
    
    if LDthresh > 0 and Test != 'general': # estimate the superwindow
        step_size=5
        max_iter=40
        snp1 = G.read(index=np.s_[Yeffect, start_index])
        left_index = start_index

        for i in range(1,max_iter+1): # a upper bound of 500 SNP in surrounding windows, with a step size of 10
            snps2 = G.read(index=np.s_[Yeffect, max(0,start_index-i*step_size):max(0,start_index-(i-1)*step_size)])
            left_index = max(0,start_index-i*step_size)
            if left_index == 0:
                break
            r_squares = LD_estimation(snp1,snps2,DEBUG=False)
            
            if np.max(r_squares) <=LDthresh:
                break
        
        snp1 = G.read(index=np.s_[Yeffect, end_index])
        right_index = end_index
        
        for i in range(1,max_iter+1): # a upper bound of 500 SNP in surrounding windows, with a step size of 10
            snps2 = G.read(index=np.s_[Yeffect, min(M,end_index+(i-1)*step_size):min(M,end_index+(i)*step_size)])
            right_index = min(M,end_index+(i)*step_size)
            if right_index == M:
                break
            r_squares = LD_estimation(snp1,snps2)
            if np.max(r_squares) <= LDthresh:
                break
        
    
    start = start_index
    end = end_index + 1
    wlen = end - start
    t0 = time.time()
    try:
        if Test == 'nonlinear':
            if LDthresh > 0:
                print(f'estimated ld block start: {left_index}; end at {right_index}')
                c = 2 - G.read(index=np.s_[Yeffect,left_index:right_index])
            else:
                c = 2 - G.read(index=np.s_[Yeffect,
                                        max(0, start - superWindow *
                                            wlen):min(G.shape[1], end +
                                                        superWindow * wlen)]
                           )  ## 2-G replicate the old conversion scheme
            # c = G[Yeffect, max(0,start-superWindow*wlen):min(G.shape[1],end+superWindow*wlen)].values
            c_test = None ## shouldn't have c_test at this scenario
        else:
            c = np.array([])
            if stage=='infer':
                c_test = np.array([])
        x = 2 - G.read(index=np.s_[Yeffect, start:end])
        if stage=='infer':
            x_test = 2 - G_test.read(index=np.s_[Yeffect_test, start:end])
        # x = G[Yeffect, start:end].values
    except Exception as e:
        traceback.print_exc()
        print(traceback.extract_tb(e.__traceback__)[0][1], e)
        # raise RuntimeError("An error occurred at line {}: {}".format(traceback.extract_tb(e.__traceback__)[0][1], e))
        print(f'start: {start}, end: {end}, Yeffect length: {len(Yeffect)}')
        return (None, None, None, None, None, None, None, None, None, None,
                None, None)

    # print(f'x shape is {x.shape}')
    t1 = time.time()

    # print(f'read value takes {t1-t0}')
    if covar:
        if Test != 'nonlinear':
            c = covarfile
            if stage=='infer':
                c_test = covarfile_test
        else:
            c = np.concatenate((c, covarfile), axis=1)
            if stage=='infer':
                c_test = np.concatenate((c_test, covarfile), axis=1)
        
    t0 = time.time()
    if Test == 'nonlinear':
        nanfilter = ~np.isnan(c).any(axis=1)
    else:
        nanfilter1 = ~np.isnan(x).any(axis=1)
        if c.size > 1:
            nanfilter2 = ~np.isnan(c).any(axis=1)
            nanfilter = nanfilter1 & nanfilter2
        else:
            nanfilter = nanfilter1
    t1 = time.time()
    # print(f'check nan takes {t1-t0}')
    if stage=='test':
        x = x[nanfilter]
        y = Y[nanfilter]
        c = c[nanfilter]
    else: # stage=='infer'
        impute = SimpleImputer()
        impute.fit(x)
        x = impute.transform(x)
        x_test = impute.transform(x_test)
        
   
    
    scaler = StandardScaler()
    x, dup_idx = np.unique(x, axis=1, return_index=True)
    scaler.fit(x)
    x = scaler.transform(x)
    if stage=='infer':
        x_test = x_test[:,dup_idx]
        x_test = scaler.transform(x_test)
    if c.size > 1:
        scaler = StandardScaler()
        c, dup_idx = np.unique(c, axis=1, return_index=True)
        impute = SimpleImputer()
        impute.fit(c)
        c = impute.transform(c)
        scaler.fit(c)
        c = scaler.transform(c)
        if stage=='infer':
            c_test = c_test.iloc[:,dup_idx]
            c_test = impute.transform(c_test)
            c_test = scaler.transform(c_test)
            
    
    t1 = time.time()
    N = x.shape[0]
    print(f'window shape is {x.shape}; covariate+sw shape is: {c.shape}')
    # y = y_new
    # for hyperparameter selection, simply replace the following list with a list of gamma values
    t0 = time.time()
    
    
    mapping = None
    Zs = []
    if Test=='nonlinear' or Test=='QuadOnly':
        mapping = PolynomialFeatures((2, 2),interaction_only=True,include_bias=False)
        Z = mapping.fit_transform(x)
        # Z = direct(x)
        
    elif Test=='linear':
        # mapping = PolynomialFeatures((2, 2),interaction_only=True,include_bias=False)
        # Z = mapping.fit_transform(x)
        # Z = x.copy()

        if stage == 'test':
            gammas = [0.001, 0.01, 0.1, 1.0, 10.0]
            for gamma in gammas:
                Z = np.sqrt(gamma) * x
                Zs.append(Z)

    elif Test=='general':
        mapping = PolynomialFeatures(degree=2,include_bias=False)

        if stage == 'test':
            gammas = [0.001, 0.01, 0.1, 1.0, 10.0]
            for gamma in gammas:
                x_gamma = np.sqrt(gamma) * x
                Z = mapping.fit_transform(x_gamma)
                Zs.append(Z)
        elif stage == 'infer':
            print(f'Use sig gamma: {sig_gamma}')
            x_gamma = np.sqrt(sig_gamma) * x
            x_gamma_test = np.sqrt(sig_gamma) * x_test
            Z = mapping.fit_transform(x_gamma)
            Zs.append(Z)
    
    else:
        raise Exception(f"The assigned test type {Test} is not supported")
    
    all_results = []
    for Z in Zs:
        # scaler=StandardScaler()
        # scaler.fit(Z)
        # Z = scaler.transform(Z)
        D = Z.shape[1]
        # Z = Z*1.0/np.sqrt(D)
        print(f'Mapping dimension D is: {D}') 
        if stage=='test':
            results = getfullComponentPerm(c,Z,y,VarCompEst=False,center=True)
            
            featImp = False
            if featImp:
                print(f'Compute the individual level Feature Importance')
                g, e = results['varcomp'][1], results['varcomp'][2]
                mu, cov = Bayesian_Posterior(c, Z, y, g, e) # Bayesian_Posterior(X,Z,y,g,e,center=True,full_cov=False):
                p_values = scipy.stats.norm.sf(abs(mu/cov))*2

                results['Bayesian_mean'] = mu
                results['Bayesian_std'] = cov
                results['Bayesian_pval'] = p_values
                
            all_results.append(results)
            # return results
        
        else:
            results = {}
            if mapping is None:
                Z_test = x_test.copy()
            else:
                Z_test = mapping.fit_transform(x_gamma_test)
            # Z_test = scaler.transform(Z_test)
            # Z_test = Z_test*1.0/np.sqrt(D)
            if c.size==0:
                c = None
                c_test=None

            # rbfs = RBFSampler(gamma=sig_gamma, n_components=x_gamma.shape[1]*50, random_state=1)
            # Z = rbfs.fit_transform(x_gamma)
            # Z_test = rbfs.fit_transform(x_gamma_test)
            
            reg, emb_train  = FastKASTRegression(c, x, Y)
            reg, emb_test = FastKASTRegression(c_test, x_test, Y_test,regs=reg)
            results['emb_train']=emb_train
            results['emb_test']=emb_test
            results['reg']=reg
            return results
            # def FastKASTRegression(X,
            #                Z,
            #                y,
            #                alphas=[1e-1,1e0,1e1],
            #                emb_return=True):
    return all_results



def parseargs():  # handle user arguments
    parser = argparse.ArgumentParser(description='main script.')
    parser.add_argument('--bfile',
                        required=True,
                        help='Plink bfile base name. Required.')
    parser.add_argument('--bfileTest',
                        help='Plink test bfile base name. Required.')
    parser.add_argument('--phen',
                        required=True,
                        help='Phenotype file. Required')
    parser.add_argument('--phenTest',
                        help='Phenotype test file. Required')
    parser.add_argument('--thread',
                        type=int,
                        default=1,
                        help='Default run on only one thread')
    parser.add_argument('--featImp',
                        action='store_true',
                        help="Compute the feature importance")
    parser.add_argument('--covar',
                        help="Covar train file.")
    parser.add_argument('--covarTest',
                        help="Covar test file.")
    parser.add_argument('--LDthresh',
                        type=float,
                        default=0,
                        help="Apply LD thresh to define superwindow")
    
    parser.add_argument(
        '--sw',
        type=int,
        default=2,
        help=
        'The superwindow is set to a multiple of the set dimension at both ends, default is 2'
    )
    
    parser.add_argument('--getPval',
                        default='regular',
                        choices=['regular', 'CCT'],
                        help="How to get p-value from mulitple testing")
    
    parser.add_argument('--output',
                        default='sim_results',
                        help='Prefix for output files.')
    parser.add_argument('--annot', default=None, help='Provided annotation file')
    parser.add_argument('--filename', default='sim', help='output file name')
    parser.add_argument('--stage',
                        default='test',
                        choices=['test', 'infer'],
                        help="Whether to perform testing or inference")
    parser.add_argument('--test',
                        default='nonlinear',
                        choices=['linear', 'nonlinear', 'general', 'QuadOnly'],
                        help='What type of kernel to test')
    parser.add_argument('--threshold',
                        default=5e-6,
                        type=float)
    parser.add_argument('--gammaFile', default=None, help='Significant gammas for inference')
    parser.add_argument('--tindex', required=True, type=int, help='Index used to submit job')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    CR = 1

    args = parseargs()
    Test = args.test
    tindex = args.tindex-1
    # set parameters
    superWindow = args.sw
    annot_path = args.annot
    featImp = args.featImp
    # read genotype data
    savepath = args.output
    LDthresh = args.LDthresh
    stage = args.stage
    bfile = args.bfile
    bfile_test = args.bfileTest
    phen = args.phen
    phenTest = args.phenTest
    covar = args.covar
    covarTest = args.covarTest
    threshold = args.threshold
    getPval = args.getPval

    if stage == 'infer':
        gamma_path = args.gammaFile
        with open(gamma_path, 'r') as file:
            lines = file.readlines()
        sig_gamma = float(lines[tindex].strip())
    
    if stage == 'infer':
        assert phenTest is not None and bfile_test is not None 
        bed_test = bfile_test + '.bed'
        G_test = open_bed(bed_test)
    
    bed = bfile + '.bed'
    fam = bfile + '.fam'
    bim = bfile + '.bim'
    
    
    fam_test = bfile_test + '.fam'
    
    # bimfile = pd.read_csv(bim, delim_whitespace=True, header=None)
    # bimfile.columns = ['chr', 'chrpos', 'MAF', 'pos', 'MAJ', 'MIN']

    gene_annot = pd.read_csv(annot_path, delimiter=' ', header=None)
    # gene_annot = np.loadtxt(annot_path,ndmin=2)
    print(f'Annotation file contains {gene_annot.shape[1]} sets to be tested')
    
    G = open_bed(bed)
    
    print('Finish lazy loading the genotype matrix')

    
    famfile = pd.read_csv(fam, delim_whitespace=True, header=None)
    famfile_test = pd.read_csv(fam_test, delim_whitespace=True, header=None)
    columns = ['FID', 'IID', 'Fa', 'Mo', 'Sex', 'Phi']
    famfile.columns = columns
    famfile_test.columns = columns
    # in total 22 indices, represent 22 chromosome
    # bimfile = bimfile[bimfile.chr==chrome]
    # Posits = bimfile.iloc[:, 3].values

    # prepare covariate
    if covar:
        covarfile = pd.read_csv(covar, delim_whitespace=True)
        assert covarfile.iloc[:, 0].equals(famfile.FID)
        covarfile = covarfile.iloc[:, 2:]
        
        if stage=='infer':
            covarfile_test = pd.read_csv(covarTest, delim_whitespace=True)
            assert covarfile_test.iloc[:, 0].equals(famfile_test.FID)
            covarfile_test = covarfile_test.iloc[:, 2:]

    print('Finish preparing the indices')
    start_index = int(tindex*CR)
    end_index = int((tindex+1)*CR)

    Y = pd.read_csv(f'{phen}.pheno', delim_whitespace=True).iloc[:, -1].values
    Yeffect = (Y != -9) & (~np.isnan(Y))
    Y = Y[Yeffect]
    if covar:
        covarfile = covarfile[Yeffect]
        
    if stage=='infer':
        Y_test = pd.read_csv(f'{phenTest}.pheno', delim_whitespace=True).iloc[:, -1].values
        Yeffect_test = (Y_test != -9) & (~np.isnan(Y_test))
        Y_test = Y_test[Yeffect_test]
        if covar:
            covarfile_test = covarfile_test[Yeffect_test]
        

        
    print('Finish loading the phenotype matrix')

    N = G.shape[0]
    AM = G.shape[1]
    results = []

    # filename = f'{args.phen}_w{wSize}_D{Map_Dim}.pkl'
    filename = args.filename
    
    
    if not os.path.isdir(f'{savepath}'):
        os.makedirs(f'{savepath}')
    sig_annot_rows = []
    if args.thread == 1:
        count = 0
        resumeFlag=True
        EMB_train = None
        if stage=='test':
            if len(glob.glob(f'{savepath}{filename}_*.pkl'))>0:
                for rcount, rownum in enumerate(range(start_index,end_index)):

                    count = rcount+1
                    annot_row=gene_annot.iloc[rownum]
                    
                    if ospath.exists(f'{savepath}{filename}_{count}.pkl'):
                            results = resumefile(savepath,filename+ '_' + str(count) + '.pkl')
                            resumeFlag=False
                            print(f'File {savepath}{filename}_{count}.pkl finished, continue')
                            continue
                    else:
                        if resumeFlag:
                            continue
                    
                    print(f'File {savepath}{filename}_{count}.pkl not exist, preceed')

                    if rownum >= gene_annot.shape[0]:
                        break
                    
                    results.append(paraCompute(annot_row))
                    dumpfile(results,
                            savepath,
                            filename + '_' + str(count) + '.pkl',
                            overwrite=True)
                    if count > 1:
                        os.remove(savepath + filename + '_' + str(count - 1) + '.pkl')
                        
                        
                    # print(results[-1]['pval'])
                    # pval = results[-1]['pval'][0][0]
                    # print(f'pval here is: {pval}')
                    # if pval <= threshold:
                    #     print(f'significant!')
                    #     sig_annot_rows.append([annot_row[0],annot_row[1]])

                    print(results[-1])
                    all_pvals = []
                    for result in results[-1]:
                        pval = result['pval'][0][0]
                        all_pvals.append(pval)

                    print(f'pvals are: {all_pvals}')
                    if getPval=='regular':
                        min_pval = min(all_pvals)
                    elif getPval=='CCT':
                        min_pval = CCT(np.array(all_pvals))
                    print(f'min pval is: {min_pval}')
                    if min_pval <= threshold:
                        print(f'significant!')
                        sig_annot_rows.append([annot_row[0],annot_row[1]])
                        
                    np.savetxt(f'{savepath}/{filename}_{count}.txt',np.array(sig_annot_rows))
                    if count > 1:
                        os.remove(savepath + filename + '_' + str(count - 1) + '.txt')
            else: ## start over
                for rcount, rownum in enumerate(range(start_index,end_index)):
                    print(rownum)
                    count = rcount+1
                    annot_row=gene_annot.iloc[rownum]
                    print(f'File {savepath}{filename}_{count}.pkl not exist, start from begining')
                    print(gene_annot.shape[0])
                    if rownum >= gene_annot.shape[0]:
                        break
                    
                    results.append(paraCompute(annot_row))
                    dumpfile(results,
                            savepath,
                            filename + '_' + str(count) + '.pkl',
                            overwrite=True)
                    if count > 1:
                        os.remove(savepath + filename + '_' + str(count - 1) + '.pkl')
                    
                    
                    # print(results[-1]['pval'])
                    # pval = results[-1]['pval'][0][0]
                    # print(f'pval here is: {pval}')
                    # if pval <= threshold:
                    #     print(f'significant!')
                    #     sig_annot_rows.append([annot_row[0],annot_row[1]])

                    print(results[-1])
                    all_pvals = []
                    for result in results[-1]:
                        pval = result['pval'][0][0]
                        all_pvals.append(pval)

                    print(f'pvals are: {all_pvals}')
                    
                    if getPval=='regular':
                        min_pval = min(all_pvals)
                    elif getPval=='CCT':
                        min_pval = CCT(np.array(all_pvals))
                        
                    print(f'min pval is: {min_pval}')
                    if min_pval <= threshold:
                        print(f'significant!')
                        sig_annot_rows.append([annot_row[0],annot_row[1]])
                        
                    np.savetxt(f'{savepath}/{filename}_{count}.txt',np.array(sig_annot_rows))
                    if count > 1:
                        os.remove(savepath + filename + '_' + str(count - 1) + '.txt')
                
        else: ## stage == infer
            for rcount, rownum in enumerate(range(start_index,end_index)):

                count = rcount+1
                annot_row=gene_annot.iloc[rownum]
                 
                result = paraCompute(annot_row)
                print(result)
                emb_train, emb_test = result['emb_train'], result['emb_test']
                emb_train = emb_train.astype(np.float16).reshape(-1,1)
                emb_test = emb_test.astype(np.float16).reshape(-1,1)
                if EMB_train is None:
                    EMB_train = emb_train
                    EMB_test = emb_test
                else:
                    EMB_train = np.concatenate((EMB_train,emb_train),axis=1)
                    EMB_test = np.concatenate((EMB_test,emb_test),axis=1)
                    
                with h5py.File(f'{savepath}/{filename}_train_{count}.h5', 'w') as hdf:
                    hdf.create_dataset('x', data=EMB_train)
                
                with h5py.File(f'{savepath}/{filename}_test_{count}.h5', 'w') as hdf:
                    hdf.create_dataset('x', data=EMB_test)
                    
                if count > 1:
                    os.remove(f'{savepath}/{filename}_train_{count-1}.h5')  
                    os.remove(f'{savepath}/{filename}_test_{count-1}.h5')  
                
