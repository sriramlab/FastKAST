#!/bin/sh


gen=./example/sim
phe=./example/sim.pheno

python ./FastKAST_annot.py \
--bfile ${gen} --phen ${phe} --mc Vanilla  --map 50 \
--output ./sim_results/ \
--gammas 0.1  \
--HP Perm --sw 2 --filename sim_M_50_known_kernel_100_beta_scan_all
