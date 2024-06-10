#!/bin/sh
#$ -cwd
#$ -l h_data=16G,h_rt=4:00:00
#$ -e /u/scratch/p/panand2/joblogs/urea_hyper_5e-8_embed
#$ -o /u/scratch/p/panand2/joblogs/urea_hyper_5e-8_embed
#$ -N infer
#$ -t 1-51:1

# echo job info on joblog:
echo "Job $JOB_ID started on:   " `hostname -s`
echo "Job $JOB_ID started on:   " `date `
echo " "

# load the job environment:
# . /u/home/class/big22/big2202/Symbolic-Pursuit/__init__.py
. /u/local/Modules/default/init/modules.sh
## Edit the line below as needed:
module load anaconda3
module load plink
python --version

# basePath=/u/project/sriram/boyang19/Epi/UKBB/train_test_split
basePath=/u/project/sriram/boyang19/Epi/UKBB/unrelWB

trait=urea
trait2=${trait}_hyper_5e-8
ofile=inferStage$((SGE_TASK_ID-1))

# inference annot path:
# test_annot=/u/scratch/p/panand2/genes_info_array_full_50_inds
test_annot=/u/scratch/p/panand2/genes_info_array_50_inds
inf_annot=/u/scratch/p/panand2/FastKAST_regressor/sig_genes/${trait2}.txt

python QuadKAST_annot_CR.py \
--bfile ${basePath}/train/filter4_train --bfileTest ${basePath}/test/filter4_test \
--phen ${basePath}/train/pheno_ivrt/${trait} --phenTest ${basePath}/test/pheno_ivrt/${trait} \
--covar ${basePath}/train/pheno_ivrt/${trait}.covar --covarTest ${basePath}/test/pheno_ivrt/${trait}.covar \
--annot ${inf_annot} --output /u/scratch/p/panand2/FastKAST_regressor/inference/${trait2}/ \
--test general --stage infer --filename ${ofile} --gammaFile /u/scratch/p/panand2/FastKAST_regressor/sig_gammas/${trait2}.txt --tindex ${SGE_TASK_ID}
