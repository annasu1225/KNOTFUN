def make_knotprot_labels(input_file, output_file):
    # Open the input file and the output file
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # Read the first line to skip the header
        next(infile)
        
        # Process each subsequent line in the input file
        for line in infile:
            # Split the line into fields based on tab separation
            fields = line.strip().split('\t')
            if len(fields) != 4:
                continue  # Skip lines that don't have exactly 4 fields
            
            # Extract fields
            pdb_code, chain, knot_type, molecular_function = fields
            
            # Split the molecular functions by commas and strip extra spaces
            molecular_functions = [func.strip() for func in molecular_function.split(',')]
            
            # Prepare the molecular function dictionary
            mf_dict = {
                'id': pdb_code,
                'molecular_functions': molecular_functions
            }
            
            # Write the formatted string to the output file
            output_line = f"PDB ID: {pdb_code}, Chain: {chain}, Uniprot ID: None, MF: {mf_dict}\n"
            outfile.write(output_line)

# Example usage
input_filename = 'KnotFun/dataset/knotted_proteins/knotprot_mf_processed_final.txt'  # Path to the input file
output_filename = 'KnotFun/dataset/knotted_proteins/knotprot_final.txt'  # Path to the output file

make_knotprot_labels(input_filename, output_filename)
