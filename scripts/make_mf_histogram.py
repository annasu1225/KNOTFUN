# Make MF histogram from .txt file like
# PDB ID: None, Chain: None, Uniprot ID: Q54T48, MF: {'id': 'Q54T48', 'molecular_functions': ['Hydrolase', 'Protease', 'Thiol protease']}
# PDB ID: None, Chain: None, Uniprot ID: Q9VYQ3, MF: {'id': 'Q9VYQ3', 'molecular_functions': ['Chromatin regulator', 'Hydrolase', 'Protease', 'Thiol protease']}
# PDB ID: None, Chain: None, Uniprot ID: Q9XZ61, MF: {'id': 'Q9XZ61', 'molecular_functions': ['Chromatin regulator', 'Hydrolase', 'Protease', 'Thiol protease']}
# PDB ID: None, Chain: None, Uniprot ID: I1MV53, MF: {'id': 'I1MV53', 'molecular_functions': ['Hydrolase', 'Protease', 'Thiol protease']}

import matplotlib.pyplot as plt
from collections import Counter
import re
import json

def read_file_and_extract_data(input_file):
    # This function reads the file and extracts molecular functions
    molecular_functions = []
    
    with open(input_file, 'r') as file:
        for line in file:
            # Find the MF JSON-like part
            match = re.search(r"MF: (\{.*\})", line)
            if match:
                # Convert the JSON-like string to a dictionary
                mf_data = eval(match.group(1))
                # Append all molecular functions found in the line to the list
                molecular_functions.extend(mf_data['molecular_functions'])
    
    return molecular_functions

def plot_histogram(molecular_functions, output_path, tag):
    # Count the occurrences of each molecular function
    counts = dict(Counter(molecular_functions).most_common(20))

    # Create a histogram
    plt.figure(figsize=(10, 6))
    plt.bar(counts.keys(), counts.values(), color='blue')
    plt.xlabel('Molecular Function')
    plt.ylabel('Count')
    plt.title(f'Histogram of Molecular Function Counts {tag}')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



def main():
    input_file = '/gpfs/gibbs/pi/gerstein/as4272/KnotFun/dataset/human_proteins/human_protein_mf_final.txt'
    output_path = '/gpfs/gibbs/pi/gerstein/as4272/KnotFun/plots/human_histogram.png'
    molecular_functions = read_file_and_extract_data(input_file)
    # I subtracted 1 here b/c it was originally showing 124. I think one of the functions does not have a matching PDB file when you run the pipeline.
    plot_histogram(molecular_functions, output_path, f"(Human, Top 20 of {len(Counter(molecular_functions))-1}, n={len(molecular_functions)})")
    

    input_file = '/gpfs/gibbs/pi/gerstein/as4272/KnotFun/dataset/knotted_proteins/knotprot_mf_processed_final.txt'
    output_path = '/gpfs/gibbs/pi/gerstein/as4272/KnotFun/plots/knotted_histogram.png'
    molecular_functions = read_file_and_extract_data(input_file)
    plot_histogram(molecular_functions, output_path, f"(Knotted, Top 20 of {len(Counter(molecular_functions))}, n={len(molecular_functions)})")

if __name__ == "__main__":
    main()
