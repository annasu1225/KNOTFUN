##!/bin/bash
##SBATCH --job-name=h2_job
##SBATCH --output=h2_job_%j.out
##SBATCH --error=h2_job_%j.err
##SBATCH --time=2-00:00:00
##SBATCH --mem-per-cpu 10G 
##SBATCH -p gpu
##SBATCH --gpus=1
##SBATCH --mail-type=ALL
##SBATCH -c 2
##SBATCH -C "a100|rtx5000|rtx3090"

# Activate the conda environment
# module --force purge
# module restore cuda11
# conda activate ph

nvidia-smi

# Directory containing the input files
input_dir="/gpfs/gibbs/pi/gerstein/as4272/KnotFun/dataset/human_proteins/human_protein_data_files_intersection"

# Directory to save the output files
output_dir="/gpfs/gibbs/pi/gerstein/as4272/KnotFun/dataset/human_proteins/human_protein_ph_h2_files"

counter=0

# For each directory in the input directory
for dir in $(ls -d ${input_dir}/*/)
do
    # Extract the id from the directory name
    id=$(basename ${dir})

    # Create the output directory if it doesn't exist
    mkdir -p "${output_dir}/${id}"

    # Construct the paths to the input and output files
    input_file="${dir}${id}_rec_bb.txt"
    threshold=30
    intermediate_file="${output_dir}/${id}/${id}_result_dict.json"
    output_file="${output_dir}/${id}/${id}_ph_vec.npy"
    diagrams_png="${output_dir}/${id}/${id}_pd.png"
    landscape_png="${output_dir}/${id}/${id}_pl.png"

    # Check if the output_file already exists, if yes, skip to the next iteration
    if [ -f "$output_file" ]; then
        echo "Skipping ${id}, ${output_file} already exists"
        continue
    fi

    # Call the Python script with the constructed file paths
    python /gpfs/gibbs/pi/gerstein/as4272/KnotFun/calculate_ph/ph_functions_h2.py ${input_file} ${threshold} ${intermediate_file} ${output_file} --diagrams_png ${diagrams_png} --landscape_png ${landscape_png}
    
    ((counter++))
    echo "${id} ph vector generated, counter: $counter"
done

echo "Total PDBs processed: $counter"
