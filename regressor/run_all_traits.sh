#!/bin/sh

# Path to the original shell script
SCRIPT_PATH="/u/home/a/aanand2/FastKAST_regressor/regressor/run_trait.sh"

# Path to the file containing the list of traits
TRAITS_FILE="/u/home/a/aanand2/FastKAST_regressor/regressor/trait_list.txt"

# Loop over each trait in the file and submit the job
while IFS= read -r TRAIT; do
  qsub ${SCRIPT_PATH} ${TRAIT}
done < "${TRAITS_FILE}"