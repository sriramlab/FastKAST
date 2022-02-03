#!/bin/sh


gen=./example/all
phe=./example/all.pheno

python FastKAST.py --bfile ${gen} --phen ${phe}
