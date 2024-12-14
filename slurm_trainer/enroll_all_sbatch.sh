for file in $(ls -1); do
    if file == "enroll_all_sbatch.sh"; then
        continue
    fi
    sabtch $file
done
