# Output: a Python dictionary of numpy arrays of persistence pairs; the dictionary is indexed by the dimension of the array of persistence pairs.
# Source: https://github.com/simonzhang00/ripser-plusplus

import json
import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import ripserplusplus as rpp_py
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import gudhi.representations as gdr
import gudhi.tensorflow.perslay as gdtf
from ph_functions import plot_pers_landscape

# Function to compute the largest distance between points
def largest_distance(points):
    n = points.shape[0]
    max_distance = 0
    for i in range(n):
        for j in range(i+1, n):
            distance = np.linalg.norm(points[i] - points[j])
            if distance > max_distance:
                max_distance = distance
    return max_distance

def load_dataset(input_file_path):
    dataset = np.loadtxt(input_file_path, skiprows=1, usecols=(1, 2, 3))
    return dataset

def run_ripserplusplus(threshold, dataset):
    # Call ripserplusplus with the given threshold
    command = f"--format point-cloud --sparse --dim 2 --threshold {threshold}"
    result_dict = rpp_py.run(command, dataset)
    print("Persistence pairs:", result_dict)
    
    return result_dict

def plot_persistence_diagrams(result_dict, id):
    # Plot persistence diagrams
    plt.figure()
    colors = ['blue', 'red', 'green', 'yellow']  # Add more colors if needed
    for dim, pairs in result_dict.items():
        for pair in pairs:
            birth, death = pair
            plt.scatter(birth, death, color=colors[int(dim)], label=f'H{dim}', s=10)
    plt.xlabel('Birth')
    plt.ylabel('Death')
    plt.title('Persistence Diagrams')

    # Create custom handles for the legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in range(len(result_dict))]
    plt.legend(handles, [f'H{i}' for i in range(len(result_dict))])

    return plt

def get_perslayer(diagrams):
    # Convert the dictionary of diagrams to a list of lists
    diagrams_list = [pairs for dim, pairs in sorted(diagrams.items(), key=lambda x: int(x[0]))]

    # Filter out infinite points and rescale coordinates
    diagrams_filtered = []
    for diagram in diagrams_list:
        filtered_diagram = []
        for point in diagram:
            if not np.isinf(point[1]):  # Check if the second value is not infinite
                filtered_diagram.append(point)
        diagrams_filtered.append(filtered_diagram)
    
    print(diagrams_filtered)

    # Reshape the filtered diagrams to 2D array
    diagrams_reshaped = [np.array(diagram) for diagram in diagrams_filtered]

    print(diagrams_reshaped)

    # Apply DiagramScaler to rescale the coordinates of the points in the persistence diagrams in the unit square
    scaler = gdr.DiagramScaler(use=True, scalers=[([0, 1], MinMaxScaler())])
    diagrams_scaled = scaler.fit_transform(diagrams_reshaped)

    # Convert into ragged tensors
    diagrams_scaled = tf.concat([
        tf.RaggedTensor.from_tensor(tf.constant(diagrams_scaled[0][None,:], dtype=tf.float32)),
        tf.RaggedTensor.from_tensor(tf.constant(diagrams_scaled[1][None,:], dtype=tf.float32)),
        tf.RaggedTensor.from_tensor(tf.constant(diagrams_scaled[2][None,:], dtype=tf.float32)),
    ], axis=0)

    with tf.GradientTape() as tape:
        rho = tf.identity 
        phi = gdtf.TentPerslayPhi(np.array(np.arange(-1.,2.,.001), dtype=np.float32))
        weight = gdtf.PowerPerslayWeight(1.,0.)
        perm_op = 'top3'
        
        perslay = gdtf.Perslay(phi=phi, weight=weight, perm_op=perm_op, rho=rho)
        vectors = perslay(diagrams_scaled)
    
    print('Gradient is ', tape.gradient(vectors, phi.samples))

    return vectors


def process_file(input_file, threshold, output_dir):
    # Load dataset
    dataset = load_dataset(input_file)
    # Run Ripser++
    result_dict = run_ripserplusplus(threshold, dataset)
    result_dict_serializable = {dim: pairs.tolist() for dim, pairs in result_dict.items()}
    id = os.path.basename(input_file).replace('_rec_bb.txt', '')
    
    intermediate_file = os.path.join(output_dir, f"{id}_result_dict.json")
    output_file = os.path.join(output_dir, f"{id}_ph_vec.npy")
    diagrams_png = os.path.join(output_dir, f"{id}_pd.png")
    landscape_png = os.path.join(output_dir, f"{id}_pl.png")

    # Save the result dictionary to an intermediate JSON file
    with open(intermediate_file, "w") as outfile:
        json.dump(result_dict_serializable, outfile)
    print("Result dictionary stored for", id)
    
    # Call get_perslayer to compute vectors
    vectors = get_perslayer(result_dict_serializable)
    
    # Save the computed vectors
    np.save(output_file, vectors.numpy())  # Make sure to convert tensors to numpy arrays if necessary
    print("Vectors stored for", id)
    
    # Plot and save persistence diagrams
    plt_diagrams = plot_persistence_diagrams(result_dict_serializable, id)
    plt_diagrams.savefig(diagrams_png)
    plt_diagrams.close()
    print(f"Persistence diagrams saved for {id}")
    
    # Plot and save persistence landscapes
    # Assuming plot_pers_landscape modifies the global plt instance
    plot_pers_landscape(vectors, id)  # This may need adaptation based on how plot_pers_landscape is implemented
    plt.savefig(landscape_png)
    plt.close()
    print(f"Persistence landscapes saved for {id}")


def main():
    parser = argparse.ArgumentParser(description='Apply persistent homology to protein backbones.')
    parser.add_argument('input_dir', help='Directory containing input files')
    parser.add_argument('output_dir', help='Directory to save output files')
    parser.add_argument('threshold', help='Persistent homology filtration threshold', type=int)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    threshold = args.threshold

    counter = 0

     # Iterate through each directory in the input directory
    for dir_path in glob.glob(os.path.join(input_dir, '*/')):
        id = os.path.basename(os.path.dirname(dir_path))
        output_subdir = os.path.join(output_dir, id)
        
        # Check if the output directory exists and is not empty
        if os.path.exists(output_subdir) and os.listdir(output_subdir):
            output_file = os.path.join(output_subdir, f"{id}_ph_vec.npy")
            # Check if the .npy file exists and is non-empty
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                print(f"Skipping {id}, output file already exists and is non-empty.")
                continue  # Skip this ID
        
        # If the output directory doesn't exist, create it
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        input_file = os.path.join(dir_path, f"{id}_rec_bb.txt")
        if os.path.exists(input_file):
            process_file(input_file, threshold, output_subdir)
            counter += 1  # Increment counter
            print(f"Processed {id}, total processed: {counter}")
    
    print(f"Total IDs processed: {counter}")

if __name__ == '__main__':
    main()

'''
Example command line usage
'''
# python calculate_ph/ph_functions_h2_v2.py 'dataset/human_proteins/human_protein_data_files_intersection' 'dataset/human_proteins/human_protein_ph_h2_files' 30