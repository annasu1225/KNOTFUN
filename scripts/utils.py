import numpy as np

def load_data(path, database):
    """
    Path is the path to one of the downloaded catalogs
    Database expects either 'alphaknot' or 'knotprot'

    Returns data, which is a dictionary with {"knotType": [knot_type], "molecular_functions": [molecular_functions]}
    """
    if database == 'knotprot':
        data = []
        with open(path, 'r') as file:
            for i, line in enumerate(file):
                if i == 0: 
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                pdb_code, chain, knot_type, molecular_functions = parts
                data.append({"pdb_code": pdb_code, "knotType": [knot_type], "molecular_functions": molecular_functions.split(", ")})
        return data
    elif database == 'alphaknot':
        data = []
        with open(path, 'r') as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                knot_type = parts[0]
                uniprot_id = parts[1]
                molecular_functions = parts[2].split(", ")
                data.append({"uniprot_id": uniprot_id, "knotType": [knot_type], "molecular_functions": molecular_functions})
        return data
    else:
        raise ValueError("Database must be either 'alphaknot' or 'knotprot'")

def get_unique_categories(data, key):
    unique_values = set()
    for item in data:
        values = item.get(key, [])
        unique_values.update(values)
    return sorted(unique_values)

def encode_features(data, features):
    feature_indices = {feature: i for i, feature in enumerate(features)}
    encoded_data = []

    for item in data:
        encoded_item = [0] * len(features)
        for value in item.get("knotType", []):
            if value in feature_indices:
                encoded_item[feature_indices[value]] = 1
        encoded_data.append(encoded_item)

    return np.array(encoded_data)

def encode_labels(data, labels):
    label_indices = {label: i for i, label in enumerate(labels)}
    encoded_data = []

    for item in data:
        encoded_item = [0] * len(labels)
        for value in item.get("molecular_functions", []):
            if value in label_indices:
                encoded_item[label_indices[value]] = 1
        encoded_data.append(encoded_item)

    return np.array(encoded_data)

def make_data(path, database):
    # Load data
    data = load_data(path, database)

    # Get unique knot types and molecular functions
    knot_types = get_unique_categories(data, "knotType")
    molecular_functions = get_unique_categories(data, "molecular_functions")

    # One-hot encode knot types
    X = encode_features(data, knot_types)

    # Multi-hot encode molecular functions
    Y = encode_labels(data, molecular_functions)

    return data, Y
