import ripserplusplus as rpp_py
import numpy as np
import json

example_dataset = "ripser-plusplus/ripserplusplus/examples/o3_4096.point_cloud"

# Read point cloud data
with open(example_dataset, "r") as file:
    lines = file.readlines()

# Parse point cloud data
points = []
for line in lines:
    coordinates = line.strip().split("\t")
    point = [float(coord) for coord in coordinates]
    points.append(point)

points = np.array(points)

# Compute pairwise distances
distances = pairwise_distances(human_dataset)

# Find the largest distance
max_distance = np.max(distances)

print("Largest distance between any two points:", max_distance)


rpp_py.run("--format point-cloud --sparse --dim 2 --threshold 1.4", example_dataset)