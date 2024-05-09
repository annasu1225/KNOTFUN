def make_alphaknot_labels(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # Skip the header line
        next(infile)
        
        # Process each subsequent line
        for line in infile:
            # Strip line to remove any extra whitespace and newline characters
            line = line.strip()
            
            # Split line into components based on tab separator
            components = line.split('\t')
            
            if len(components) == 3:
                knot_type, uniprot_id, molecular_functions = components
                
                # Split molecular functions by comma and strip whitespace
                functions_list = [func.strip() for func in molecular_functions.split(',')]
                
                # Format output as specified
                output_line = (
                    f"PDB ID: None, Chain: None, Uniprot ID: {uniprot_id}, "
                    f"MF: {{'id': '{uniprot_id}', 'molecular_functions': {functions_list}}}\n"
                )
                
                # Write formatted string to output file
                outfile.write(output_line)

input_filename = 'KnotFun/dataset/knotted_proteins/alphaknot_mf_processed.txt'  # Path to the input file
output_filename = 'KnotFun/dataset/knotted_proteins/alphaknot_mf_processed_final.txt'  # Path to the output file

make_alphaknot_labels(input_filename, output_filename)