#!/bin/bash

for file in $(ls -1); do
    if [ "$file" == "enroll_all_sbatch.sh" ]; then
        continue
    fi
    if [[ "$file" == * ]]; then
        sbatch $file
    fi
done