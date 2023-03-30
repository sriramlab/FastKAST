#!/bin/sh


gen=./example/sim
phe=./example/sim.phen0.pheno

python ./FastKAST_annot.py \
--bfile ${gen} --phen ${phe} --mc Vanilla  --map 50 \
--annot ./example/sim.overlap.annot --output ./sim_results/ \
--HP Perm --sw 2 --filename sim_M_50_unknown_kernel_overlap_beta_more
