#!/bin/sh

# Path to the original shell script
SCRIPT_PATH="run_trait.sh"

# Path to the file containing the list of traits
TRAITS_FILE="traits.txt"

# Loop over each trait in the file and submit the job
while IFS= read -r TRAIT; do
  qsub ${SCRIPT_PATH} ${TRAIT}
done < "${TRAITS_FILE}"
