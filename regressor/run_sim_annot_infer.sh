#!/bin/sh


gen=train_test/sim_train
phe=train_test/sim.train.phen0

genTest=train_test/sim_test
pheTest=train_test/sim.test.phen0

python QuadKAST_annot.py \
--bfile ${gen} --bfileTest ${genTest} --phen ${phe} --phenTest ${pheTest} \
--annot ../example/sim.overlap.annot --output sim_emb_results/ \
--test general --stage infer --filename sim_M_50_unknown_kernel_infer
