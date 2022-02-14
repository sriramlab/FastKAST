#!/bin/sh


gen=./example/sim
phe=./example/sim.pheno

python FastKAST.py --bfile ${gen} --phen ${phe} --covar ${gen}.covar --region all
