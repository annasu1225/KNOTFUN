#!/bin/bash
#SBATCH --job-name=calculate_features
#SBATCH --output=calculate_features_%j.out
#SBATCH --error=calculate_features_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

source /gpfs/gibbs/project/gerstein/as4272/conda_envs/python_env/bin/activate

# Directory where the raw PDB files are stored
pdb_dir="PDBs"

# Directory where the output files will be stored
output_dir="data_files"

for pdb_file in $pdb_dir/*.pdb; do
    # Get the PDB ID from the file name
    pdb_id=$(basename $pdb_file .pdb)

    # Create a directory for this PDB ID
    mkdir -p $output_dir/$pdb_id

    # Run the scripts and save the output files in the PDB ID directory
    srun python calculate_features/reconstruct.py $pdb_file $output_dir/$pdb_id/${pdb_id}_rec.pdb
    srun python calculate_features/extract.py $output_dir/$pdb_id/${pdb_id}_rec.pdb $output_dir/$pdb_id/${pdb_id}_rec_bb.txt
    srun python calculate_features/distances.angles.py $output_dir/$pdb_id/${pdb_id}_rec_bb.txt $output_dir/$pdb_id/${pdb_id}_bl.txt $output_dir/$pdb_id/${pdb_id}_ba.txt $output_dir/$pdb_id/${pdb_id}_da.txt
done
