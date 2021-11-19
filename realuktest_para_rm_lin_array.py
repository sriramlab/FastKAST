import numpy as np
import os, re
from sklearn import linear_model
from scipy.stats import norm, wishart
import sys
from sys import path as syspath
from os import path as ospath
import time
import math
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm import tqdm as tqdm
from numpy.linalg import svd
from scipy.sparse import csr_matrix
from sklearn import preprocessing
import multiprocessing
import pandas_plink
from pathlib import Path
from pandas_plink import read_plink1_bin
from functools import partial
from sklearn.linear_model import LinearRegression
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import additive_chi2_kernel
from joblib import dump, load
from sklearn.model_selection import KFold
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
    return p_min


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
    print(f'shape of c is {c.shape}')
    x = G[:, indices].values
    x = x[sort,:]
    c = c[sort,:]
    m = x.shape[1]
    rmNA = ~np.isnan(c).any(axis=1)
    print(f'removed index are {rmNA[0:10]}')
    print(f'before remove na Y mean is {np.mean(Y)}')
    y = Y[rmNA]
    print(f'after remove na y mean is {np.mean(y)}')
    xc = covariate[rmNA]
    print(f'covar snps shape is {c.shape}') 
#    col_mean = np.nanmean(c, axis=0)
#    inds = np.where(np.isnan(c))
#    c[inds] = np.take(col_mean, inds[1])
    x = x[rmNA]
    # x,index =np.unique(x, axis=0, return_index=True)
    c = c[rmNA]
    x = (np.unique(x, axis=1, return_index=False))
    c = (np.unique(c, axis=1, return_index=False))
    # y = y[index]
    x =  np.float32(preprocessing.scale(x))
    xc = np.concatenate((xc,c),axis=1)
    xc =  np.float32(preprocessing.scale(xc))
    print(f'y mean after reg is {np.mean(y)}') 
    N = x.shape[0]
    print(f'window shape is {x.shape}')
    # y = y_new
    for N in tqdm([N], desc='N'):
        for d in tqdm([m], leave=False, desc='d:'):
            for D in tqdm([d*Map_Dim], leave=False, desc='D'):
                # generate data
                print('Generating data...')
                SKATs= []
                SKAT_times = []
                gammas = np.logspace(-3,1,5)
                t0 = time.time()
                for gamma in gammas:
                    params = dict(gamma=gamma, kernel_metric='rbf', D=D, center=True, hutReps=250)
                    print('start rbf')
                    SKAT,SKAT_time = estimateSigmasGeneral(y, y, c, x, how='fast_mle', params=params, method='Perm')
                    if len(SKAT) == 0:
                        continue
                        # return (None,None,None,None,Index,0,0,chrome)
                    SKATs.append(SKAT)
                    SKAT_times.append(SKAT_time)
                t1 = time.time()
                pval, p_perm = flatten_p(SKATs),flatten_perm(SKATs)
                print(pval)
                print(p_perm)
                print(f'gamma test takes {t1-t0}')
                 
    return (pval, p_perm, SKAT_times, Index, N, m, chrome, gammas)




if __name__ == "__main__":
    tindex = int(sys.argv[1])-1
    debug=True
    single_index=False
    if len(sys.argv) == 3:
        single_index=True
        file_index = int(sys.argv[2])-1
    superWindow=2
    Map_Dim = 50
    savepath = '/u/flashscratch/b/boyang19/MoM2/traitsResults/'
    wSize = 100000 # window size 100Kb
    CR = 100 # need 291 slots
    orgpath = '/u/project/sriram/ukbiobank/33127/ukb21970_plink/phenos_mapped_33297_to_33127/pheno_files/'
    path = '/u/project/sgss/UKBB/data/cal/'
    # path2 = '/u/project/sgss/UKBB/data/imp/Qced/All/'
    bed = path+'filter4.bed'
    fam = path+'filter4.fam'
    bim = path+'filter4.bim'
    bimfile = pd.read_csv(bim,sep='\t',header=None)
    bimfile.columns = ['chr', 'chrpos', 'MAF', 'pos','MAJ','MIN']
    famfile = pd.read_csv(fam,sep=' ',header=None)
    columns = ['FID','IID','Fa','Mo','Sex','Phi']
    famfile.columns = columns
    # in total 22 indices, represent 22 chromosome
    # bimfile = bimfile[bimfile.chr==chrome]
    Windows = [] 
    Posits = bimfile.iloc[:,3].values
    for chrome in range(1,23):
        start = np.min(Posits[bimfile.chr==chrome])
        end = np.max(Posits[bimfile.chr==chrome])
        Windows.append([(chrome, w) for w in range(start, end+wSize, wSize)])
    
        # chunks = math.ceil(M/1000)
    print('start')
    numIterations = 1
    hmean = 0
    results = []

    traits = []
    regex=re.compile('(.*pheno)')
    for root,dirs,files in os.walk(orgpath):
        for file in files:
            if regex.match(file):
                trait = str(file).split('.')[0]
                try:
                    np.load(f'./newtraits/{trait}_regress_pheno.npy')
                    traits.append(trait)
                except:
                    continue
    G = read_plink1_bin(bed, bim, fam, verbose=False)
    famfile = pd.read_csv(fam,sep=' ',header=None)
    famfile.columns = columns
    Index = pd.DataFrame({'Index': np.arange(famfile.shape[0])})
    famfile = famfile.join(Index)
    famfile = famfile.set_index('IID')
    traits = ['blood_mpv', 'blood_mch', 'blood_mscv', 'cystatin_c', 'ldl', 'shbg', 'urate']
    if single_index:
        traits = traits[file_index:(file_index+1)]
    else:
        traits = traits[3:]
    for trait in traits:
        print(f'in trait {traits}')
        covariate = pd.read_csv(orgpath+f'{trait}.covar',delimiter=' ').iloc[:,2:].to_numpy()
        newfam = savepath+f'{trait}.fam'
        newfamfile = pd.read_csv(newfam,sep=' ',header=None)
        print(f'new fam file dim is {newfamfile.shape}')
        newfamfile.columns = columns
        NIndex = pd.DataFrame({'Index': np.arange(newfamfile.shape[0])})
        newfamfile = newfamfile.join(NIndex)
        newfamfile = newfamfile.set_index('IID')
        sort = famfile.loc[newfamfile.index].Index.values

        
        Y = pd.read_csv(orgpath+f'/{trait}.pheno',delimiter=' ').pheno.values
        covariate = (covariate[~(Y == -9)])
        Y = np.float32(Y[~(Y == -9)])
#        covariate = preprocessing.scale(covariate)
#        G = read_plink1_bin(bed, bim, newfam, verbose=False)
        print('done reading genotype')
        N = G.shape[0]
        AM = G.shape[1]
        results = []
        # y = np.load('./processphen o/y-25-25.npy')
        # print(y.shape)
        start_index = int(tindex*CR)
        end_index = int((tindex+1)*CR)
        chr_index = 0
        for i in range(1,22):
            if start_index >= np.ceil(len(Windows[i-1])/CR)*CR:
                start_index -= np.ceil(len(Windows[i-1])/CR)*CR
                end_index -= np.ceil(len(Windows[i-1])/CR)*CR
                chr_index = i
            else:
                break
        path = f'./array_trait/FameSVD/{Map_Dim}_gamma/'
        filename = f'{trait}_w{wSize}_D{Map_Dim}_{chr_index}_part_{tindex}_gamma.pkl'
        if fileExist(path,filename):
            continue
        if debug:
            for w in Windows[chr_index][int(start_index):int(end_index)]:
                results.append(paraCompute(w))
        else:
            print('start parallel')
            pool = multiprocessing.Pool(processes=32)
            results = pool.map(paraCompute,Windows[chr_index][int(start_index):int(end_index)])
            pool.close()
            pool.join()
        dumpfile(results,path,filename,overwrite=True)
#        np.save(f'./traitsResults/{trait}_scale_impute_whole_part_{tindex}', results)
#        dump(results, f'./imputeBug/{trait}_w100_largewindow_array_morecovar_cv0_{chr_index}_part_{tindex}.pkl')
