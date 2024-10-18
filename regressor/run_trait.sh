#!/bin/sh
#$ -cwd
#$ -l h_data=32G,h_rt=10:00:00
#$ -e /u/scratch/p/panand2/joblogs/embed_rbf
#$ -o /u/scratch/p/panand2/joblogs/embed_rbf
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

trait=$1
# stage=$2
# idx=$3

# ofile=testStage${stage}
ofile=testStage$((SGE_TASK_ID-1))

annot=/u/scratch/p/panand2/genes_info_array_50_inds
threshold=5e-6

outdir=/u/scratch/p/panand2/FastKAST_regressor/results_rbf/${trait}_${threshold}/
embeddir=/u/scratch/p/panand2/FastKAST_regressor/results_rbf/${trait}_${threshold}/embeddings/

mkdir -p ${outdir}
mkdir -p ${embeddir}

# python QuadKAST_annot_CR.py --bfile /u/project/sriram/boyang19/Epi/UKBB/unrelWB/train/filter4_train --bfileTest /u/project/sriram/boyang19/Epi/UKBB/unrelWB/test/filter4_test --phen /u/project/sriram/boyang19/Epi/UKBB/unrelWB/train/pheno_ivrt/lipo_a --phenTest /u/project/sriram/boyang19/Epi/UKBB/unrelWB/test/pheno_ivrt/lipo_a --covar /u/project/sriram/boyang19/Epi/UKBB/unrelWB/train/pheno_ivrt/lipo_a.covar --covarTest /u/project/sriram/boyang19/Epi/UKBB/unrelWB/test/pheno_ivrt/lipo_a.covar --getPval 'CCT' --annot /u/scratch/p/panand2/genes_info_array_50_inds --output /u/scratch/p/panand2/FastKAST_regressor/results_cct/testing/ --test general --threshold 5e-6 --stage test --filename test --CR 1 --tindex 15

python QuadKAST_annot_CR.py \
--bfile ${basePath}/train/filter4_train --bfileTest ${basePath}/test/filter4_test \
--phen ${basePath}/train/pheno_ivrt/${trait} --phenTest ${basePath}/test/pheno_ivrt/${trait} \
--covar ${basePath}/train/pheno_ivrt/${trait}.covar --covarTest ${basePath}/test/pheno_ivrt/${trait}.covar \
--getPval 'CCT' --gammas 0.01 0.1 1.0 \
--annot ${annot} --output ${outdir} \
--test RBF --threshold ${threshold} --stage test --filename ${ofile} --tindex ${SGE_TASK_ID}

