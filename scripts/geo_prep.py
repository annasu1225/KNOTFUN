from utils import make_data
import os
import json

def save_master_file(database, processed_catalog_path, folder_path, output_path):
    """
    database: either alphaknot or knotprot
    processed_catalog_path: path to the processed catalog file
    folder_path: path to the folder containing the processed data
    output_path: path to save the master file
    """
    # Process data to get Y (assuming your make_data function is designed to handle the data structure from load_data)
    data, Y = make_data(processed_catalog_path, database)

    # Initialize the JSON object
    pdb_json = {}

    # Iterate over each row in the data
    for row in data:
        if database == 'knotprot':
            pdb_id = row['pdb_code']
        elif database == 'alphaknot':
            pdb_id = row['uniprot_id']
        else:
            raise ValueError("Database must be either 'alphaknot' or 'knotprot'")
        
        subdir = os.path.join(folder_path, pdb_id)

        # Construct the paths for the pdb, atoms, bond length, and bond angle files
        pdb_file_path = os.path.join(subdir, f'{pdb_id}_rec.pdb')
        atoms_file_path = os.path.join(subdir, f'{pdb_id}_rec_bb.txt')
        bond_length_file_path = os.path.join(subdir, f'{pdb_id}_bl.txt')
        bond_angle_file_path = os.path.join(subdir, f'{pdb_id}_ba.txt')

        # Check if the directory and all files exist
        if os.path.isdir(subdir) and all(os.path.isfile(path) for path in [pdb_file_path, atoms_file_path, bond_length_file_path, bond_angle_file_path]):
            # Get the absolute paths for the files
            absolute_pdb_path = os.path.abspath(pdb_file_path)
            absolute_atoms_path = os.path.abspath(atoms_file_path)
            absolute_bond_length_path = os.path.abspath(bond_length_file_path)
            absolute_bond_angle_path = os.path.abspath(bond_angle_file_path)

            # Assuming the index of the current row matches the index in Y
            labels = Y[data.index(row)]

            # Add the entry to the JSON object
            pdb_json[pdb_id] = {
                'pdb_path': absolute_pdb_path,
                'atoms_path': absolute_atoms_path,
                'bond_length_path': absolute_bond_length_path,
                'bond_angle_path': absolute_bond_angle_path,
                'labels': labels.tolist()  # Convert to list if Y is a numpy array
            }

    # Convert the JSON object to a string and save it to a file
    with open(output_path, 'w') as json_file:
        json.dump(pdb_json, json_file, indent=4)

# Path to the knot_file.txt and folder_path
database = 'knotprot' # Changes the format in which we expect data
knot_file_path = 'catalogs/knotprot_mf_processed.txt'
folder_path = 'data_files'
output_path = 'output.json'

if __name__ == '__main__':
    save_master_file(database, knot_file_path, folder_path, output_path)