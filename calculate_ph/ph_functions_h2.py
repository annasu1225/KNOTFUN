# Output: a Python dictionary of numpy arrays of persistence pairs; the dictionary is indexed by the dimension of the array of persistence pairs.
# Source: https://github.com/simonzhang00/ripser-plusplus


import ripserplusplus as rpp_py
import numpy as np
import json

example_dataset = "ripser-plusplus/ripserplusplus/examples/o3_4096.point_cloud"
human_dataset = np.loadtxt('human_protein_data_files/5BK8/5BK8_rec_bb.txt', skiprows=1, usecols=(1, 2, 3))

# Function to compute pairwise distances between points
def pairwise_distances(points):
    n = points.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            distances[i, j] = np.linalg.norm(points[i] - points[j])
            distances[j, i] = distances[i, j]  # Symmetric matrix
    return distances

# # Read point cloud data
# with open(example_dataset, "r") as file:
#     lines = file.readlines()

# # Parse point cloud data
# points = []
# for line in lines:
#     coordinates = line.strip().split("\t")
#     point = [float(coord) for coord in coordinates]
#     points.append(point)

# points = np.array(points)

# Compute pairwise distances
distances = pairwise_distances(human_dataset)

# Find the largest distance
max_distance = np.max(distances)

print("Largest distance between any two points:", max_distance)


# data = np.loadtxt('human_protein_data_files/5BK8/5BK8_rec_bb.txt', skiprows=1, usecols=(1, 2, 3))
# print("data=", data)

# dict = rpp_py.run("--format point-cloud --sparse --dim 2 --threshold 100", data)

# Call ripserplusplus with the threshold set to max_distance
command = f"--format point-cloud --sparse --dim 2 --threshold 30"
result_dict = rpp_py.run(command, human_dataset)
print("Persistence pairs:", result_dict)

# Convert NumPy arrays to lists in the result_dict
result_dict_serializable = {dim: pairs.tolist() for dim, pairs in result_dict.items()}

# Store the result_dict to a file
with open("result_dict.json", "w") as outfile:
    json.dump(result_dict_serializable, outfile)
print("Result dictionary stored in 'result_dict.json'.")