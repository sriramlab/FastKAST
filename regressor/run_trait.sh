#!/bin/sh
#$ -cwd
#$ -l h_data=16G,h_rt=2:00:00
#$ -e /u/scratch/p/panand2/joblogs/joblog.$JOB_ID
#$ -o /u/scratch/p/panand2/joblogs/joblog.$JOB_ID
#$ -N embeddings
#$ -t 1-10078:1

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

SGE_TASK_ID=1

basePath=/u/project/sriram/boyang19/Epi/UKBB/train_test_split

trait=Eosinophill_count_30150-0.0
ofile=testStage_$((SGE_TASK_ID-1))

python QuadKAST_annot_CR.py \
--bfile ${basePath}/train/full_excld_sub_train --bfileTest ${basePath}/test/full_excld_sub_test \
--phen ${basePath}/train/pheno/${trait} --phenTest ${basePath}/test/pheno/${trait} \
--covar ${basePath}/train/pheno/covar.txt.pheno --covarTest ${basePath}/test/pheno/covar.txt.pheno \
--getPval 'CCT' \
--annot /u/scratch/p/panand2/genes.annot --output /u/scratch/b/boyang19/tmp/u/flashscratch/b/boyang19/QuadKAST_emb/real_trait/results/${trait}/ \
--test general --stage test --filename ${ofile} --tindex ${SGE_TASK_ID}
