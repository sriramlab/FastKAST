gen=../sim

module load plink
plink --bfile ${gen} --keep train_individuals.txt --make-bed --out sim_train

plink --bfile ${gen} --keep test_individuals.txt --make-bed --out sim_test