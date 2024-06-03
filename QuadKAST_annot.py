import numpy as np
import os, re
import argparse
import sys
from sys import path as syspath
from os import path as ospath
import time
import math
import pandas as pd
from tqdm import tqdm as tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
import multiprocessing
from bed_reader import open_bed
import pandas_plink
from pandas_plink import read_plink1_bin
from joblib import dump, load
from utils import *
from fastmle_res_jax import *



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
    SNP_annot = args
    indices_of_ones = np.where(SNP_annot==1)[0]
    M = SNP_annot.shape[0]
    
    if len(indices_of_ones) ==0:
        raise Exception("Annotation file cannot have columns with all zeros")
    else:
        start_index = indices_of_ones[0]
        end_index = indices_of_ones[-1]
    
    
    if end_index-start_index <=3:
        print(f'Warning: target window size too small, numerical issue may encounter')
    
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
        else:
            c = np.array([])
        x = 2 - G.read(index=np.s_[Yeffect, SNP_annot.astype(bool)])
        # x = G[Yeffect, start:end].values
    except Exception as e:
        print(f'Error: {e}')
        print(f'start: {start}, end: {end}, Yeffect: {Yeffect}')
        return (None, None, None, None, None, None, None, None, None, None,
                None, None)

    # print(f'x shape is {x.shape}')
    t1 = time.time()

    # print(f'read value takes {t1-t0}')
    if covar != None:
        if Test != 'nonlinear':
            c = covarfile
        else:
            c = np.concatenate((c, covarfile), axis=1)
    t0 = time.time()
    if Test == 'nonlinear':
        nanfilter = ~np.isnan(c).any(axis=1)
        c = c[nanfilter]
    else:
        nanfilter1 = ~np.isnan(x).any(axis=1)
        if c.size > 1:
            nanfilter2 = ~np.isnan(c).any(axis=1)
            nanfilter = nanfilter1 & nanfilter2
            c = c[nanfilter]
        else:
            nanfilter = nanfilter1
    t1 = time.time()
    # print(f'check nan takes {t1-t0}')
    x = x[nanfilter]
    y = Y[nanfilter]
    scaler = StandardScaler()
    x, x_idx = np.unique(x, axis=1, return_index=True)
    sort_x_idx = np.argsort(x_idx)
    x = x[:,sort_x_idx]
    if c.size > 1:
        c = np.unique(c, axis=1, return_index=False)
        c = scaler.fit_transform(c)
    x = scaler.fit_transform(x)
    N = x.shape[0]
    print(f'window shape is {x.shape}; covariate shape is {covarfile.shape}; covariate+sw shape is: {c.shape}')
    # y = y_new

    # for hyperparameter selection, simply replace the following list with a list of gamma values
    t0 = time.time()
    
    if Test=='nonlinear' or Test=='QuadOnly':
        mapping = PolynomialFeatures((2, 2),interaction_only=True,include_bias=False)
        Z = mapping.fit_transform(x)
        # Z = direct(x)
        
    elif Test=='linear':
        # mapping = PolynomialFeatures((2, 2),interaction_only=True,include_bias=False)
        # Z = mapping.fit_transform(x)
        Z = x.copy()

    elif Test=='general':
        mapping = PolynomialFeatures(degree=2,include_bias=False)
        Z = mapping.fit_transform(x)
    
    else:
        raise Exception(f"The assigned test type {Test} is not supported")
    
    scaler=StandardScaler()
    Z = scaler.fit_transform(Z)
    D = Z.shape[1]
    Z = Z*1.0/np.sqrt(D)
    print(f'Mapping dimension D is: {D}') 
    results = getfullComponentPerm(c,Z,y,VarCompEst=True,center=True)
    
    if featImp:
        print(f'Compute the individual level Feature Importance')
        g, e = results['varcomp'][1], results['varcomp'][2]
        mu, cov = Bayesian_Posterior(c, Z, y, g, e) # Bayesian_Posterior(X,Z,y,g,e,center=True,full_cov=False):
        p_values = scipy.stats.norm.sf(abs(mu/cov))*2

        results['Bayesian_mean'] = mu
        results['Bayesian_std'] = cov
        results['Bayesian_pval'] = p_values
        
    return results



def parseargs():  # handle user arguments
    parser = argparse.ArgumentParser(description='main script.')
    parser.add_argument('--bfile',
                        required=True,
                        help='Plink bfile base name. Required.')
    parser.add_argument('--covar',
                        required=False,
                        help='Covariate file. Not required')
    parser.add_argument('--phen',
                        required=True,
                        help='Phenotype file. Required')
    parser.add_argument('--thread',
                        type=int,
                        default=1,
                        help='Default run on only one thread')
    parser.add_argument('--featImp',
                        action='store_true',
                        help="Compute the feature importance")
    
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
    parser.add_argument('--output',
                        default='sim_results',
                        help='Prefix for output files.')
    parser.add_argument('--annot', default=None, help='Provided annotation file')
    parser.add_argument('--filename', default='sim', help='output file name')
    parser.add_argument('--test',
                        default='nonlinear',
                        choices=['linear', 'nonlinear', 'general', 'QuadOnly'],
                        help='What type of kernel to test')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseargs()
    Test = args.test
    # set parameters
    superWindow = args.sw
    annot_path = args.annot
    featImp = args.featImp
    # read genotype data
    savepath = args.output
    LDthresh = args.LDthresh
    bfile = args.bfile
    bed = bfile + '.bed'
    fam = bfile + '.fam'
    bim = bfile + '.bim'

    bimfile = pd.read_csv(bim, delim_whitespace=True, header=None)
    bimfile.columns = ['chr', 'chrpos', 'MAF', 'pos', 'MAJ', 'MIN']

   
    gene_annot = np.loadtxt(annot_path,ndmin=2)
    print(f'Annotation file contains {gene_annot.shape[1]} sets to be tested')
    
    G = open_bed(bed)
    print('Finish lazy loading the genotype matrix')

    
    famfile = pd.read_csv(fam, delim_whitespace=True, header=None)
    columns = ['FID', 'IID', 'Fa', 'Mo', 'Sex', 'Phi']
    famfile.columns = columns
    # in total 22 indices, represent 22 chromosome
    # bimfile = bimfile[bimfile.chr==chrome]
    Posits = bimfile.iloc[:, 3].values

    # prepare covariate
    covar = args.covar
    print(f'covar is {covar}')
    if covar != None:
        covarfile = pd.read_csv(covar, delim_whitespace=True)
        assert covarfile.iloc[:, 0].equals(famfile.FID)
        covarfile = covarfile.iloc[:, 2:]

    print('Finish preparing the indices')

    Y = pd.read_csv(args.phen, delim_whitespace=True).iloc[:, -1].values
    Yeffect = (Y != -9) & (~np.isnan(Y))
    Y = Y[Yeffect]
    if covar != None:
        covarfile = covarfile[Yeffect]
    print('Finish loading the phenotype matrix')

    N = G.shape[0]
    AM = G.shape[1]
    results = []

    # filename = f'{args.phen}_w{wSize}_D{Map_Dim}.pkl'
    filename = args.filename
    if args.thread == 1:
        count = 0
        for colnum in tqdm(range(0, gene_annot.shape[1])):
            count += 1
            annot_row=gene_annot[:,colnum]
            results.append(paraCompute(annot_row))
            dumpfile(results,
                     savepath,
                     filename + '_' + str(count) + '.pkl',
                     overwrite=True)
            if count > 1:
                os.remove(savepath + filename + '_' + str(count - 1) + '.pkl')
