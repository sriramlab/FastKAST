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
from estimators import *
from utils import *


def flatten_perm(pairs):
    N = len(pairs) # N is the number of hyperparamter
    M = len(pairs[0]) # M is the number of windows tested
    # print(pairs)
    p_all = []
    for n in range(N):
        p_row = []
        for m in range(1,M):
            p_row.append(pairs[n][m])
        p_all.append(p_row)
    p_all = np.array(p_all)
    p_min = np.amin(p_all,axis=0)
    return list(p_min)

def flatten_p(pairs,complete=False):
    N = len(pairs) # N is the number of hyperparamter
    M = len(pairs[0]) # M is the number of permutations
    p_all = []
    for n in range(N):
        p_row = []
        for m in range(1):
            p_row.append(pairs[n][m])
        p_all.append(p_row)
    p_all = np.array(p_all).flatten()
    p_min = np.amin(p_all)
    bindex = np.argmin(p_all)
    if complete:
        return (p_all,bindex)
    else:
        return (p_min,bindex)


def CCT(pvals, ws=None):
    N = len(pvals)
    if not ws:
        ws = np.array([1/N for i in range(N)])
    T = np.sum(ws*np.tan(0.5-pvals)*np.pi)
    pval = 0.5 - np.arctan(T)/np.pi
    return pval

def MCM(pvals, ws = None):
    pval = CCT(pvals,ws)
    pval = 2*np.min([0.5,min(pvals),pval])
    return pval

def CMC(pvals, ws = None):
    pval = CCT(pvals,ws)
    pvals = np.array([pval, np.min(pvals)])
    pval = CCT(pvals)
    return pval
    

def paraCompute(args):
    start_index,end_index = args
    start = start_index
    end = end_index
    wlen = end - start
    t0 = time.time()
    try:
        if Test!='general':
            c = 2-G.read(index=np.s_[Yeffect,max(0,start-superWindow*wlen):min(G.shape[1],end+superWindow*wlen)]) ## 2-G replicate the old conversion scheme
            # c = G[Yeffect, max(0,start-superWindow*wlen):min(G.shape[1],end+superWindow*wlen)].values
        else:
            c = np.array([])
        x = 2-G.read(index=np.s_[Yeffect,start:end])
        # x = G[Yeffect, start:end].values
    except Exception as e:
        print(f'Error: {e}')
        print(f'start: {start}, end: {end}, Yeffect: {Yeffect}')
        return (None, None, None, None, None, None, None, None, None, None, None, None)
    
    # c = G[Yeffect, max(0,start-superWindow*wlen):min(G.shape[1],end+superWindow*wlen)].values
    # x = G[Yeffect, start_index:end_index].values
    print(f'x shape is {x.shape}')
    t1 = time.time()
    
    print(f'read value takes {t1-t0}')
    if covar != None:
        if Test=='general':
            c = covarfile
        else:
            c = np.concatenate((c,covarfile),axis=1)
    t0 = time.time()
    if Test!='general':
        nanfilter=~np.isnan(c).any(axis=1)
        c = c[nanfilter]
    else:
        nanfilter1=~np.isnan(x).any(axis=1)
        if c.size>1:
            nanfilter2=~np.isnan(c).any(axis=1)
            nanfilter=nanfilter1&nanfilter2
            c = c[nanfilter]
        else:
            nanfilter = nanfilter1
    t1 = time.time()
    print(f'check nan takes {t1-t0}')
    x = x[nanfilter]
    y = Y[nanfilter]  
    if Test=='higher':
        poly = PolynomialFeatures(2,interaction_only=True, include_bias=False)
        quad = poly.fit_transform(x)[:,(x.shape[1]):]
        c = np.concatenate((c,quad),axis=1)
    # c =  c[:,~np.all(c[1:] == c[:-1], axis=0)]
    scaler = StandardScaler()
    t0 = time.time()
    x = np.unique(x, axis=1, return_index=False)
    if c.size>1:
        c = np.unique(c, axis=1, return_index=False)
        c = scaler.fit_transform(c)
    x = scaler.fit_transform(x)
    t1 = time.time()
    # print(f'standardized takes {t1-t0}')
    # print(f'x shape is {x.shape}')
    # print(f'c shape is {c.shape}')
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
        params = dict(gamma=gamma, kernel_metric='rbf', D=D, center=True, hutReps=250, version=QMC)
        # params = dict(gamma=gamma, kernel_metric='rbf', D=D, center=True, hutReps=250)
        # print('start rbf')
        print(f'Start computing FastKAST')
        SKAT,FastKAST_time, states = estimateSigmasGeneral(y, c, x, how='fast_mle', params=params, method=HP,Test=Test)
        print(f'Start computing Exact kernel')
        # SKAT_MLE,Exact_time, mle_states  = estimateSigmasGeneral(y, c, x, how='mle', params=params)
        SKAT_MLE = None
        if len(SKAT) == 0:
            continue
            # return (None,None,None,None,Index,0,0,chrome)
        SKATs.append(SKAT)
        FastKAST_times.append(FastKAST_time)
    t1 = time.time()

    ### SKATs: N*M matrix, with N as number of hyperprameter tested, M as number of 

    if len(gammas) == 1:
        pval = SKATs[0][0]
        print(f'pval is {pval}')   
        return (pval, None, FastKAST_times, start_index, end_index, N, d, None, states, SKAT_MLE,count)
    elif HP =='Perm':
        (pval,bindex), p_perm = flatten_p(SKATs),flatten_perm(SKATs)
        bgamma = gammas[bindex]
        print('#######################')
        print(f'hyperparameter gamma is {bgamma}')
        # print(f'pval is {pval}')     
        return (pval, p_perm, FastKAST_times, start_index, end_index, N, d, bgamma, states, count)
    elif HP=='CCT':
        pvals,bindex = flatten_p(SKATs,complete=True)
        pval = CCT(pvals)
        print(f'CCT pval is {pval}')
        return (pval, None, FastKAST_times, start_index, end_index, N, d, None, pvals, states, count)
    elif HP=='vary': # various statistics computation
        pvals,bindex = flatten_p(SKATs,complete=True)
        pval_cct = CCT(pvals)
        pval_mcm = MCM(pvals)
        pval_cmc = CMC(pvals)
        print(f'CCT pval is {pval_cct}, MCM pval is {pval_mcm}, CMC pval is {pval_cmc}')
        return ((pval_cct,pval_mcm,pval_cmc), None, FastKAST_times, start_index, end_index, N, d, None, pvals, states)
    else:
        raise ValueError(f"{HP} is currently not supported")



def parseargs():    # handle user arguments
    parser = argparse.ArgumentParser(description='isoQTL main script.')
    parser.add_argument('--bfile', required=True, help='Plink bfile base name. Required.')
    parser.add_argument('--covar', required=False, help='Covariate file. Not required')
    parser.add_argument('--phen',  required=True, help='Phenotype file. Required')
    parser.add_argument('--HP', default='Perm', choices=['Perm', 'CCT','vary'],help='The p-value calculation scheme when grid search is implemented')
    parser.add_argument('--map', type=int, default=50, help='The mapping dimension divided by m')
    parser.add_argument('--window', type=int, default=100000, help='The physical length of window')
    parser.add_argument('--thread', type=int, default=1, help='Default run on only one thread')
    parser.add_argument('--sw', type=int, default=2, help='The superwindow is set to a multiple of the set dimension at both ends, default is 2')
    parser.add_argument('--mc', default='Vanilla', choices=['Vanilla', 'Halton','Sobol'], help='sampling method for RFF')
    parser.add_argument('--output', default='sim_results', help='Prefix for output files.')
    parser.add_argument('--annot', help='Provided annotation file')
    parser.add_argument('--filename', default='sim', help='output file name')
    parser.add_argument('--test', default='nonlinear', choices=['nonlinear', 'higher','general'], help='What type of kernel to test')
    parser.add_argument('--gammas',default=[0.01,0.1,1], nargs='+',type=float)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseargs()
    Test = args.test
    HP = args.HP
    # set parameters
    Map_Dim = args.map
    wSize = args.window
    superWindow= args.sw
    annot_path = args.annot
    gammas = args.gammas
    print(f'gammas is {gammas}')
    QMC = args.mc
    # read genotype data
    savepath = args.output
    bfile = args.bfile
    bed = bfile+'.bed'
    fam = bfile+'.fam'
    bim = bfile+'.bim'
   
    gene_annot = pd.read_csv(annot_path,delimiter=' ')
    G = open_bed(bed)
    # G = read_plink1_bin(bed, bim, fam, verbose=False)
    print('Finish lazy loading the genotype matrix')

    bimfile = pd.read_csv(bim,delim_whitespace=True,header=None)
    bimfile.columns = ['chr', 'chrpos', 'MAF', 'pos','MAJ','MIN']
    famfile = pd.read_csv(fam,delim_whitespace=True,header=None)
    columns = ['FID','IID','Fa','Mo','Sex','Phi']
    famfile.columns = columns
    # in total 22 indices, represent 22 chromosome
    # bimfile = bimfile[bimfile.chr==chrome]
    Posits = bimfile.iloc[:,3].values

    # prepare covariate
    covar = args.covar
    print(f'covar is {covar}')
    if covar != None:
        covarfile = pd.read_csv(covar,delim_whitespace=True)
        assert covarfile.iloc[:,0].equals(famfile.FID)
        covarfile = covarfile.iloc[:,2:]

    # prepare index information
#     chromes = np.unique(bimfile.chr).astype(int)
#     for chrome in chromes:
#         start = 0
# #        start = np.min(Posits[bimfile.chr==chrome])
#         end = np.max(Posits[bimfile.chr==chrome])
#         Windows.append([(chrome, w) for w in range(start, end+wSize, wSize)])

    
    print('Finish preparing the indices')

    
    Y = pd.read_csv(args.phen,delim_whitespace=True,header=None).iloc[:,-1].values
    Yeffect = (Y!=-9)&(~np.isnan(Y))
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
        for rownum in tqdm(range(0,gene_annot.shape[0])):
            count += 1
            if rownum < gene_annot.shape[0]:
                w = gene_annot.iloc[rownum]
            else:
                w = None
                break
            results.append(paraCompute(w))
            dumpfile(results,savepath,filename+ '_' + str(count) + '.pkl',overwrite=True)
            if count > 1:
                os.remove(savepath+filename+'_' + str(count-1) + '.pkl')
