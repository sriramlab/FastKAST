#!/bin/sh
#$ -cwd
#$ -l h_data=16G,h_rt=4:00:00
#$ -e /u/scratch/p/panand2/joblogs/lipo_a
#$ -o /u/scratch/p/panand2/joblogs/lipo_a
#$ -N test
#$ -t 1-191:1

# echo job info on joblog:
echo "Job $JOB_ID started on:   " `hostname -s`
echo "Job $JOB_ID started on:   " `date `
echo " "

# load the job environment:
. /u/local/Modules/default/init/modules.sh
## Edit the line below as needed:
module load anaconda3
module load plink
python --version

basePath=/u/project/sriram/boyang19/Epi/UKBB/unrelWB

trait=lipo_a
ofile=testStage$((SGE_TASK_ID-1))

annot=/u/scratch/p/panand2/genes_info_array_50_inds
threshold=5e-6

python QuadKAST_annot_CR.py \
--bfile ${basePath}/train/filter4_train --bfileTest ${basePath}/test/filter4_test \
--phen ${basePath}/train/pheno_ivrt/${trait} --phenTest ${basePath}/test/pheno_ivrt/${trait} \
--covar ${basePath}/train/pheno_ivrt/${trait}.covar --covarTest ${basePath}/test/pheno_ivrt/${trait}.covar \
--getPval 'CCT' \
--annot ${annot} --output /u/scratch/p/panand2/FastKAST_regressor/results/${trait}_${threshold}/ \
--test general --threshold ${threshold} --stage test --filename ${ofile} --tindex ${SGE_TASK_ID}
