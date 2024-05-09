def modify_molecular_functions(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Find the start of the PDB ID field and extract the PDB ID
            pdb_id_index = line.find("PDB ID: ") + 8
            pdb_id_end_index = line.find(",", pdb_id_index)
            pdb_id = line[pdb_id_index:pdb_id_end_index].strip()

            # Find the molecular function dictionary and replace the 'id'
            mf_start_index = line.find("MF: {")
            if mf_start_index != -1:
                # Reconstruct the line with the new 'id'
                before_mf = line[:mf_start_index + len("MF: {'id': '")]
                after_mf = line[mf_start_index + len("MF: {'id': '") + len(pdb_id):]
                
                # Update the 'id' in the molecular functions dictionary
                new_line = before_mf + pdb_id + after_mf
                outfile.write(new_line)
            else:
                # If no MF section is found, write the line as is
                outfile.write(line)

# Usage example:
input_path = "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/dataset/human_proteins/human_protein_mf_final.txt"
output_path = "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/dataset/human_proteins/human_protein_mf_final_unified.txt"
modify_molecular_functions(input_path, output_path)
