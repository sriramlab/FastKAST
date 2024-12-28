#!/bin/sh


gen=./example/sim
phe=./example/sim.pheno

python ./QuadKAST_annot.py \
--bfile ${gen} --phen ${phe} \
--annot ./example/sim.new.annot --output ./sim_results/ \
--sw 2 --filename sim_M_50_quadKAST --featImp --LDthresh 0.5
