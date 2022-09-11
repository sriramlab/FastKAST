#!/bin/sh


gen=./example/sim
phe=./example/sim.pheno

python /home/boyang1995/research/FastKAST_revise/FastKAST/FastKAST_annot.py \
--bfile ${gen} --phen ${phe} --mc Vanilla  --map 50 \
--annot ./example/sim.annot --output ./sim_results/ \
--HP Perm --sw 2 --filename sim_M_50
