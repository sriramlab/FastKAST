import numpy as np
# from numba_stats import norm
from sklearn.impute import SimpleImputer
import os
import argparse
import time
import pandas as pd
from tqdm import tqdm as tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from bed_reader import open_bed


def impute_def(x):
    col_mean = np.nanmean(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(col_mean, inds[1])
    return x


def impute(x):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    x = imp.fit_transform(x)
    return x


def index_match(start, chrome, wSize, bimfile):
    '''
    match bim file position to relative index
    '''
    indices = bimfile[(bimfile.chr == chrome) & (bimfile.pos >= start) &
                      (bimfile.pos < start + wSize)].index
    if len(indices) == 0:
        return (np.nan, np.nan)
    rstart = indices[0]
    rend = indices[-1]
    return (rstart, rend)


def flatten_perm(pairs):
    '''
    Flatten the permuted p-value
    '''
    N = len(pairs)  # N is the number of hyperparamter
    M = len(pairs[0])  # M is the number of windows tested
    # print(pairs)
    p_all = []
    for n in range(N):
        p_row = []
        for m in range(1, M):
            p_row.append(pairs[n][m])
        p_all.append(p_row)
    p_all = np.array(p_all)
    p_min = np.amin(p_all, axis=0)
    return list(p_min)


def flatten_p(pairs, complete=False):
    '''
    Flatten the original p-value
    '''
    N = len(pairs)  # N is the number of hyperparamter
    M = len(pairs[0])  # M is the number of permutations
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
        return (p_all, bindex)
    else:
        return (p_min, bindex)


def geno_phen_processing_contig(args):
    '''
    Define the annotation file as a contiguous set of features (SNPs), with start and end index position
    '''
    Test = args.test
    HP = args.HP
    # set parameters
    Map_Dim = args.map
    wSize = args.window
    superWindow = args.sw
    annot_path = args.annot
    gammas = args.gammas
    covar = args.covar
    print(f'gammas is {gammas}')
    QMC = args.mc
    # read genotype data
    savepath = args.output
    bfile = args.bfile
    bed = bfile + '.bed'
    fam = bfile + '.fam'
    bim = bfile + '.bim'

    bimfile = pd.read_csv(bim, delim_whitespace=True, header=None)
    bimfile.columns = ['chr', 'chrpos', 'MAF', 'pos', 'MAJ', 'MIN']

    gene_annot = np.loadtxt(annot_path, ndmin=2)
    print(f'Annotation file contains {gene_annot.shape[1]} sets to be tested')
    G = open_bed(bed)
    # G = read_plink1_bin(bed, bim, fam, verbose=False)
    print('Finish lazy loading the genotype matrix')

    famfile = pd.read_csv(fam, delim_whitespace=True, header=None)
    columns = ['FID', 'IID', 'Fa', 'Mo', 'Sex', 'Phi']
    famfile.columns = columns
    # in total 22 indices, represent 22 chromosome
    # bimfile = bimfile[bimfile.chr==chrome]
    Posits = bimfile.iloc[:, 3].values

    # prepare covariate

    print(f'covar is {covar}')
    if covar != None:
        covarfile = pd.read_csv(covar, delim_whitespace=True)
        assert covarfile.iloc[:, 0].equals(famfile.FID)
        covarfile = covarfile.iloc[:, 2:]

    print('Finish preparing the indices')

    Y = pd.read_csv(args.phen, delim_whitespace=True,
                    header=None).iloc[:, -1].values
    Yeffect = (Y != -9) & (~np.isnan(Y))
    Y = Y[Yeffect]
    if covar != None:
        covarfile = covarfile[Yeffect]
    print('Finish loading the phenotype matrix')

    return G, Y, covarfile, Yeffect, gene_annot,

    # filename = f'{args.phen}_w{wSize}_D{Map_Dim}.pkl'
    filename = args.filename
    if args.thread == 1:
        count = 0
        for colnum in tqdm(range(0, gene_annot.shape[1])):
            count += 1
            annot_row = gene_annot[:, colnum]
            results.append(paraCompute(annot_row))
            dumpfile(results,
                     savepath,
                     filename + '_' + str(count) + '.pkl',
                     overwrite=True)
            if count > 1:
                os.remove(savepath + filename + '_' + str(count - 1) + '.pkl')
