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
import multiprocessing
import pandas_plink
from pandas_plink import read_plink1_bin
from joblib import dump, load
from estimators import *
from utils import *


def flatten_perm(pairs):
    N = len(pairs) # N is the number of hyperparamter
    M = len(pairs[0]) # M is the number of windows teste
    p_all = []
    for n in range(N):
        p_row = []
        for m in range(1,M):
            p_row.append(pairs[n][m][0])
        p_all.append(p_row)
    p_all = np.array(p_all)
    p_min = np.amin(p_all,axis=0)
    return list(p_min[:10])

def flatten_p(pairs,debug=False):
    N = len(pairs) # N is the number of hyperparamter
    M = len(pairs[0]) # M is the number of permutations
    p_all = []
    for n in range(N):
        p_row = []
        for m in range(1):
            p_row.append(pairs[n][m][0])
        p_all.append(p_row)
    p_all = np.array(p_all)
    p_min = np.amin(p_all)
    bindex = np.argmin(p_all)
    return (p_min,bindex)


def paraCompute(args):
    chrome = args[0]
    Index = args[1]
    indices = bimfile[(bimfile.chr==chrome)&(bimfile.pos>=Index)&(bimfile.pos<Index+wSize)].index
    if indices.size <= 2:
        return (None, None, None, Index, 0, 0, chrome)
    start = indices[0]
    end = indices[-1]
    wlen = end - start
    c = G[:, max(0,start-2*wlen):min(G.shape[1],end+2*wlen)].values
    x = G[:, indices].values
    if covar != None:
        c = np.concatenate((c,covarfile),axis=1)
    c = np.concatenate((covarfile,c),axis=1)
    nanfilter=~np.isnan(c).any(axis=1)
    c = c[nanfilter]
    x = x[nanfilter]
    y = Y[nanfilter]
    # c =  c[:,~np.all(c[1:] == c[:-1], axis=0)]
    scaler = StandardScaler()
    c = scaler.fit_transform(c)
    x = scaler.fit_transform(x)
    print(f'x shape is {x.shape}')
    # print(f'covar shape is {c.shape}') 
    N = x.shape[0]
    # print(f'window shape is {x.shape}')
    # y = y_new
    d = x.shape[1]
    D = d*Map_Dim

    SKATs= []
    FastKAST_times = []
    # for hyperparameter selection, simply replace the following list with a list of gamma values
    t0 = time.time()
    for gamma in gammas:
        params = dict(gamma=gamma, kernel_metric='rbf', D=D, center=True, hutReps=250)
        # print('start rbf')
        SKAT,FastKAST_time = estimateSigmasGeneral(y, c, x, how='fast_mle', params=params, method='Perm')
        if len(SKAT) == 0:
            continue
            # return (None,None,None,None,Index,0,0,chrome)
        SKATs.append(SKAT)
        FastKAST_times.append(FastKAST_time)
    t1 = time.time()
    (pval,bindex), p_perm = flatten_p(SKATs),flatten_perm(SKATs)
    bgamma = gammas[bindex]
    print('#######################')
    print(f'hyperparameter gamma is {bgamma}')
    print(f'pval is {pval}')
                 
    return (pval, p_perm, FastKAST_times, Index, N, d, chrome, bgamma)



def parseargs():    # handle user arguments
    parser = argparse.ArgumentParser(description='isoQTL main script.')
    parser.add_argument('--bfile', required=True, help='Plink bfile base name. Required.')
    parser.add_argument('--covar', required=False, help='Covariate file. Not required')
    parser.add_argument('--phen',  required=True, help='Phenotype file. Required')
    parser.add_argument('--map', type=int, default=50, help='The mapping dimension divided by m')
    parser.add_argument('--window', type=int, default=100000, help='The physical length of window')
    parser.add_argument('--thread', type=int, default=1, help='Default run on only one thread')
    parser.add_argument('--sw', type=float, default=2, help='The superwindow is set to a multiple of the set dimension at both ends, default is 2')
    parser.add_argument('--dir', default='./example/', help='Path to the output')
    parser.add_argument('--output', default='sim_results', help='Prefix for output files.')
    parser.add_argument('--region', default='partial', help='region to test, default is to test only on chromosome 1. To test the whole data, change to "all" ')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parseargs()
    # set parameters
    Map_Dim = args.map
    wSize = args.window
    superWindow= args.sw

    # read genotype data
    savepath = args.output
    bfile = args.bfile
    bed = bfile+'.bed'
    fam = bfile+'.fam'
    bim = bfile+'.bim'
    G = read_plink1_bin(bed, bim, fam, verbose=False)
    print('Finish lazy loading the genotype matrix')

    bimfile = pd.read_csv(bim,delim_whitespace=True,header=None)
    bimfile.columns = ['chr', 'chrpos', 'MAF', 'pos','MAJ','MIN']
    famfile = pd.read_csv(fam,delim_whitespace=True,header=None)
    columns = ['FID','IID','Fa','Mo','Sex','Phi']
    famfile.columns = columns
    # in total 22 indices, represent 22 chromosome
    # bimfile = bimfile[bimfile.chr==chrome]
    Windows = [] 
    Posits = bimfile.iloc[:,3].values

    # prepare covariate
    covar = args.covar
    if covar != None:
        covarfile = pd.read_csv(covar,delim_whitespace=True)
        assert covarfile.iloc[:,0].equals(famfile.FID)
        covarfile = covarfile.iloc[:,2:]

    # prepare index information
    chromes = np.unique(bimfile.chr)
    for chrome in chromes:
        start = np.min(Posits[bimfile.chr==chrome])
        end = np.max(Posits[bimfile.chr==chrome])
        Windows.append([(chrome, w) for w in range(start, end+wSize, wSize)])
    
    if args.region == 'partial':
        print(f'partial region')
        # only test the first chromosome
        Windows = [Windows[0]]
        # chunks = math.ceil(M/1000)
    print('Finish preparing the indices')
    
    # read phenotype 
    
    Y = pd.read_csv(args.phen,delimiter=' ',header=None).iloc[:,5].values
    print('Finish loading the phenotype matrix')

    N = G.shape[0]
    AM = G.shape[1]
    results = []
    gammas = [0.1] # to perform testing with multiple hyperparameter gamma, simply put the candidates here

    filename = f'{args.phen}_w{wSize}_D{Map_Dim}.pkl'
    filename = '/'+filename.split('/')[-1]
    if args.thread == 1:
        for p, chrome in enumerate(Windows):
            print(f'In chromesome: {p+1}')
            for w in tqdm(chrome, desc='set'):
                results.append(paraCompute(w))
    else:
        print('start parallel')
        pool = multiprocessing.Pool(processes=32)
        results = pool.map(paraCompute,Windows[chr_index][int(start_index):int(end_index)])
        pool.close()
        pool.join()
    dumpfile(results,savepath,filename,overwrite=True)
