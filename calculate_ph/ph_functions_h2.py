# Output: a Python dictionary of numpy arrays of persistence pairs; the dictionary is indexed by the dimension of the array of persistence pairs.
# Source: https://github.com/simonzhang00/ripser-plusplus

import json
import argparse
import os
import ripserplusplus as rpp_py
import numpy as np
import matplotlib.pyplot as plt

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

# Compute the largest distance
# max_distance = largest_distance(human_dataset)
# print("Largest distance between any two points:", max_distance)

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


def main():
    parser = argparse.ArgumentParser(description='Apply persistent homology to protein backbones.')
    parser.add_argument('input_file', help='Protein backbone file')
    parser.add_argument('threshold', help='Persistent homology filtration threshold')
    parser.add_argument('intermediate_file', help='PH dictionary JSON file')
    parser.add_argument('output_file', help='PersLayer vectors file')
    parser.add_argument('--diagrams_png', help='Persistence diagrams PNG file')
    parser.add_argument('--landscape_png', help='Persistence landscape PNG file')

    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset(args.input_file)

    # Run Ripser++
    result_dict = run_ripserplusplus(args.threshold, dataset)
    result_dict_serializable = {dim: pairs.tolist() for dim, pairs in result_dict.items()}

    if args.intermediate_file:
        with open((args.intermediate_file).replace('_rec_bb.txt', ''), "w") as outfile:
            json.dump(result_dict_serializable, outfile)
        print("Result dictionary stored.")
    
    # Call get_perslayer
    vectors = get_perslayer(result_dict_serializable)

    if args.output_file:
        np.save(args.output_file, vectors)
        print("Vectors stored.")

    # Plot persistence diagrams
    plt_diagrams = plot_persistence_diagrams(result_dict, os.path.basename(args.input_file).replace('_rec_bb.txt', ''))
    if args.diagrams_png:
        plt_diagrams.savefig(args.diagrams_png)
    plt_diagrams.close()

    # Plot persistence landscapes
    plot_pers_landscape(vectors, os.path.basename(args.input_file).replace('_rec_bb.txt', ''))
    if args.landscape_png:
        plt.savefig(args.landscape_png)
    plt.close()
    


if __name__ == '__main__':
    main()

'''
Example command line usage
'''
# python calculate_ph/ph_functions_h2.py 'dataset/human_proteins/human_protein_data_files/6AKR/6AKR_rec_bb.txt' 30 'dataset/human_proteins/human_protein_ph_h2_files/6AKR/6AKR_result_dict.json' 'dataset/human_proteins/human_protein_ph_h2_files/6AKR/6AKR_ph_vec.npy' --diagrams_png 'dataset/human_proteins/human_protein_ph_h2_files/6AKR/6AKR_pd.png' --landscape_png 'dataset/human_proteins/human_protein_ph_h2_files/6AKR/6AKR_pl.png'
