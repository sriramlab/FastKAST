import numpy as np
import os
import argparse
import time
import pandas as pd
from tqdm import tqdm as tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from bed_reader import open_bed
from estimators import *
from utils import *








def parseargs():  # handle user arguments
    parser = argparse.ArgumentParser(description='isoQTL main script.')
    parser.add_argument('--bfile',
                        required=True,
                        help='Plink bfile base name. Required.')
    parser.add_argument('--covar',
                        required=False,
                        help='Covariate file. Not required')
    parser.add_argument('--phen',
                        required=True,
                        help='Phenotype file. Required')
    parser.add_argument(
        '--HP',
        default='Perm',
        choices=['Perm', 'CCT', 'vary'],
        help='The p-value calculation scheme when grid search is implemented')
    parser.add_argument('--map',
                        type=int,
                        default=50,
                        help='The mapping dimension divided by m')
    parser.add_argument('--window',
                        type=int,
                        default=100000,
                        help='The physical length of window')
    parser.add_argument('--thread',
                        type=int,
                        default=1,
                        help='Default run on only one thread')
    parser.add_argument(
        '--sw',
        type=int,
        default=2,
        help=
        'The superwindow is set to a multiple of the set dimension at both ends, default is 2'
    )
    parser.add_argument('--mc',
                        default='Vanilla',
                        choices=['Vanilla', 'Halton', 'Sobol'],
                        help='sampling method for RFF')
    parser.add_argument('--output',
                        default='sim_results',
                        help='Prefix for output files.')
    parser.add_argument('--annot', default=None, help='Provided annotation file')
    parser.add_argument('--filename', default='sim', help='output file name')
    parser.add_argument('--test',
                        default='nonlinear',
                        choices=['nonlinear', 'higher', 'general'],
                        help='What type of kernel to test')
    parser.add_argument('--gammas',
                        default=[0.01, 0.1, 1],
                        nargs='+',
                        type=float)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseargs()
    Test = args.test
    HP = args.HP
    # set parameters
    Map_Dim = args.map
    wSize = args.window
    superWindow = args.sw
    annot_path = args.annot
    gammas = args.gammas
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

    # if annot_path is None:
    #     Windows = []
    #     print(f'No annotation file is provided, scan all windows with a fixed window size')
    #     chromes = np.unique(bimfile.chr)
    #     Posits = bimfile.pos.values
    #     for chrome in chromes:
    #         start = np.min(Posits[bimfile.chr == chrome])
    #         end = np.max(Posits[bimfile.chr == chrome])
    #         Windows.extend([index_match(w,chrome,wSize,bimfile) for w in range(start, end + wSize, wSize)])
    #     gene_annot = pd.DataFrame(data=Windows)
    #     gene_annot = gene_annot.dropna().astype(int)
    #     print(gene_annot)
    # else:
    #     gene_annot = pd.read_csv(annot_path, delimiter=' ')
    
    gene_annot = np.loadtxt(annot_path,ndmin=2)
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
    covar = args.covar
    print(f'covar is {covar}')
    if covar != None:
        covarfile = pd.read_csv(covar, delim_whitespace=True)
        assert covarfile.iloc[:, 0].equals(famfile.FID)
        covarfile = covarfile.iloc[:, 2:]

    # prepare index information
#     chromes = np.unique(bimfile.chr).astype(int)
#     for chrome in chromes:
#         start = 0
# #        start = np.min(Posits[bimfile.chr==chrome])
#         end = np.max(Posits[bimfile.chr==chrome])
#         Windows.append([(chrome, w) for w in range(start, end+wSize, wSize)])

    print('Finish preparing the indices')

    Y = pd.read_csv(args.phen, delim_whitespace=True,
                    header=None).iloc[:, -1].values
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
