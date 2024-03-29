import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import numpy as np
import pandas as pd
import os
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import json
from itertools import cycle


def plot_label_bar_chart(labels, prefix=""):
    """
    Plot the sums of the labels
    """

    label_sums = [np.sum(label) for label in labels]

    # Get counts of each label sum
    label_sum_counts = {}
    for label_sum in label_sums:
        if label_sum in label_sum_counts:
            label_sum_counts[label_sum] += 1
        else:
            label_sum_counts[label_sum] = 1

    # Get counts of each label id
    label_id_counts = {}

    for label in labels:
        # The labels are multi-hot encoded, get the indices that are 1
        label_ids = np.where(label == 1)[0]
        for label_id in label_ids:
            if label_id in label_id_counts:
                label_id_counts[label_id] += 1
            else:
                label_id_counts[label_id] = 1
    
    os.makedirs(f'./{prefix}', exist_ok=True)

    # Plot the bar chart
    plt.bar(label_id_counts.keys(), label_id_counts.values())
    plt.xlabel('Label ID')
    plt.ylabel('Frequency')
    plt.title('Frequency of Labels')
    plt.savefig(os.path.join(prefix, 'label_id_histogram.jpg'))

    plt.bar(label_sum_counts.keys(), label_sum_counts.values(), tick_label=list(map(int, label_sum_counts.keys())))
    plt.xlabel('Number of Labels for a PDB ID')
    plt.ylabel('Frequency')
    plt.title('Frequency of Number of Labels for a PDB ID')
    plt.savefig(os.path.join(prefix,'./label_histogram.jpg'))

def create_dataset(csv_path, labels_path, embeddings_path, ph_files_path, ph_only=False):
    
    # Read the PDB ids
    pdb_ids = pd.read_csv(csv_path, header=None).iloc[:, 0]

    # Load labels and graph embeddings
    labels = np.load(labels_path)

    print("Length of labels", labels.shape)

    graph_embeddings = np.load(embeddings_path)

    # Initialize X and Y
    X = []
    Y = []

    # Process each PDB id
    for pdb_id, label in zip(pdb_ids, labels):
        ph_file_path = os.path.join(ph_files_path, pdb_id, f"{pdb_id}_ph_vec.npy")
        if os.path.exists(ph_file_path):
            try:
                ph_vec = np.load(ph_file_path).flatten()
            except Exception as e:
                print(f"Error for PDB id: {pdb_id}", e)
            
            if ph_only:
                feature_vec = ph_vec
            else:
                graph_emb = graph_embeddings[pdb_ids.tolist().index(pdb_id)]
                feature_vec = np.concatenate((ph_vec, graph_emb))
            X.append(feature_vec)
            Y.append(label)

        else:
            print(f"Error for PDB id: {pdb_id}", "File not found")

    return np.array(X), np.array(Y)

def create_dataset_from_scratch(label_path, ph_folder):
    """
    Create the PH only dataset (X, Y)

    Input:
    - label_path: Path to the labels.txt file.
    - ph_folder: Path to the folder containing PH vectors.

    Returns:
    - (X, Y) where X is a numpy array of PH vectors, and Y is a numpy array of associated molecular functions, multi-hot encoded.
    """

    functions_dict = {}
    with open(label_path, 'r') as f:
        for line in f.readlines():
            pdb_id = line.split(',')[0].split(': ')[1]
            functions = eval(line.split(',', 3)[-1].split(': ', 1)[-1])['molecular_functions']
            functions_dict[pdb_id] = functions

    # Initialize lists to hold our PH vectors (X) and labels (Y)
    X, Y = [], []
    all_functions = set()

    # # Load PH vectors and associate them with molecular functions
    # for pdb_id, functions in functions_dict.items():
    #     ph_vec_path = os.path.join(ph_folder, pdb_id, pdb_id + "_ph_vec.npy")
    #     if os.path.exists(ph_vec_path):
    #         ph_vec = np.load(ph_vec_path).flatten()
    #         X.append(ph_vec)
    #         Y.append(functions)  # Temporarily store the list of functions
    #         all_functions.update(functions)
    #     else:
    #         print("Missing path for", pdb_id)

    # Load PH vectors and associate them with molecular functions
    for pdb_id, functions in functions_dict.items():
        ph_vec_path = os.path.join(ph_folder, pdb_id, pdb_id + "_ph_vec.npy")
        if os.path.exists(ph_vec_path):
            ph_vec = np.load(ph_vec_path)
            if ph_vec.size > 0:  # Check if the array is non-empty
                ph_vec = ph_vec.flatten()
                X.append(ph_vec)
                Y.append(functions)  # Temporarily store the list of functions
                all_functions.update(functions)
            else:
                print(f"{pdb_id} vector is empty")  # Indicate that the array is empty
        else:
            print(f"Missing path for {pdb_id}")


    # Convert all_functions to a list and map each function to an index
    all_functions = list(all_functions)
    function_to_index = {function: i for i, function in enumerate(all_functions)}

    # Initialize Y as a matrix of zeros
    Y_encoded = np.zeros((len(Y), len(all_functions)))

    # Encode Y using multi-hot encoding
    for i, functions in enumerate(Y):
        for function in functions:
            if function in function_to_index:  # Safety check
                Y_encoded[i, function_to_index[function]] = 1

    # Convert X to a numpy array for consistency
    X = np.array(X)

    return X, Y_encoded

def train(model, X_train, Y_train, learning_rate=0.005):
    """
    Train the TF model on the dataset. Evaluate on ROC-AUC
    """

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Accuracy(name='accuracy'), tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'), exact_match_accuracy])

    # Train the model
    history = model.fit(X_train, Y_train, epochs=500, batch_size=32, validation_split=0.2)

    return model, history

def evaluate(model, X_test, Y_test):
    """
    Evaluate the model on the dataset. Evaluate on the metric the model was compiled for.
    """

    # Evaluate the model on the AUC metric
    loss, auc, accuracy, binary_accuracy, exact_accuracy = model.evaluate(X_test, Y_test)

    return loss, auc, accuracy, binary_accuracy, exact_accuracy

def exact_match_accuracy(y_true, y_pred):
    """
    Compute the exact match accuracy for multi-label classification.
    
    Parameters:
    - y_true: TensorFlow tensor, the true labels.
    - y_pred: TensorFlow tensor, the predicted labels (as probabilities).
    
    Returns:
    - accuracy: A tensor representing the exact match accuracy.
    """
    # Convert predictions to binary by thresholding at 0.5
    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
    
    # Calculate if all labels match for each example
    matches = tf.reduce_all(tf.equal(y_true, y_pred_binary), axis=-1)
    
    # Compute accuracy as mean of matches
    accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))
    
    return accuracy

def get_model(input_size, output_size, hidden_size=64, hidden_layers=1):
    """
    Get the MLP model used.
    """
    
    layers = [Dense(64, activation='relu', input_shape=(input_size,)), ]

    for layer in range(hidden_layers):
        layers.append(Dense(64, activation='relu'),)
    
    layers.append(Dense(output_size, activation='sigmoid'))

    model = Sequential(layers)

    return model

def remove_dominant_label(X, Y):
    """
    Remove all samples that have the dominant label from the dataset
    """
    # Get counts of each label id
    label_id_counts = {}

    for label in Y:
        # The labels are multi-hot encoded, get the indices that are 1
        label_ids = np.where(label == 1)[0]
        for label_id in label_ids:
            if label_id in label_id_counts:
                label_id_counts[label_id] += 1
            else:
                label_id_counts[label_id] = 1
    
    # Get the index of the dominant label
    dominant_label_index = max(label_id_counts, key=label_id_counts.get)

    X_wo = X[Y[:, dominant_label_index] == 0]
    Y_wo = Y[Y[:, dominant_label_index] == 0]

    return X_wo, Y_wo

def reduce_dataset_k_labels(X, Y, k):
    """
    Reduces the dataset X and multi-hot labels Y to only include the k most frequent labels.
    
    Parameters:
    - X: NumPy array, the dataset with shape (n_samples, n_features)
    - Y: NumPy array, the multi-hot encoded labels with shape (n_samples, n_labels)
    - k: int, the number of most frequent labels to keep
    
    Returns:
    - X_reduced: NumPy array, the reduced dataset
    - Y_reduced: NumPy array, the reduced multi-hot encoded labels
    """
    # Count the frequency of each label
    label_frequencies = np.sum(Y, axis=0)
    
    # Identify the indices of the k most frequent labels
    most_frequent_labels = np.argsort(label_frequencies)[-k:]
    
    # Filter Y to only include columns for the k most frequent labels
    Y_reduced = Y[:, most_frequent_labels]
    
    # Identify samples that are associated with at least one of the k most frequent labels
    samples_to_keep = np.where(np.sum(Y_reduced, axis=1) > 0)[0]
    
    # Filter X to include only those samples
    X_reduced = X[samples_to_keep]
    Y_reduced = Y_reduced[samples_to_keep]
    
    return X_reduced, Y_reduced

def remove_ph_group(X, Y, group_index):
    """
    Removes group_index (0, 1, 2) PH group from the dataset

    Returns the modified X and the original Y
    """

    # Original vector for each example is x_i \in 3x9000. So the flattened is 1x27000
    if group_index == 0:
        return X[:, 9000:], Y # Ablate the first group
    elif group_index == 1:
        return X[:, :9000], Y # Ablate the second group
    elif group_index == 2:
        return X[:, 18000:], Y # Ablate the third group
    else:
        raise ValueError("Invalid group index provided for removing PH group. Must be 0, 1, 2.")


# def run():
#     # Path to CSV file containing PDB ids used for GeoGNN Training. Was used to ensure that we only ran PH on PDB ids we had GeoGNN embeddings for.
#     ids_path = "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/PaddleHelix/apps/pretrained_compound/ChemRL/GEM/finetune_models/knot/ids_in_order.csv"
#     # Path to the labels file for the dataset (Y)
#     labels_path = "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/PaddleHelix/apps/pretrained_compound/ChemRL/GEM/finetune_models/knot/labels.npy"
#     # Path to the PH embeddings for the dataset (X_ph)
#     x_ph_path = "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/human_protein_ph_h2_files"
#     # Path to the GeoGNN embeddings for the dataset (X_geo)
#     x_geo_path = "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/PaddleHelix/apps/pretrained_compound/ChemRL/GEM/finetune_models/knot/graph_encodings.npy"
#     # Path to the new labels file
#     labels_path = "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/catalogs/human_protein_mf_processed.txt"
    
#     X, Y = create_dataset_from_scratch(labels_path, x_ph_path)

#     # Training w/ all data
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#     model, _ = train(get_model(X.shape[1], Y.shape[1]), X_train, Y_train)
#     full_loss, full_auc, full_accuracy, full_binary_accuracy, full_exact_accuracy = evaluate(model, X_test, Y_test)

#     label_counts = [20, 15, 10, 5]

#     # Loop through each label count, train, and evaluate models
#     for k in label_counts:
#         # Reduce dataset to top k labels
#         X, Y = reduce_dataset_k_labels(X, Y, k)
#         # Split the dataset
#         X_train_k, X_test_k, Y_train_k, Y_test_k = train_test_split(X, Y, test_size=0.2, random_state=42)
#         # Train the model
#         model_k, _ = train(get_model(X.shape[1], Y.shape[1]), X_train_k, Y_train_k)
#         # Evaluate the model
#         loss_k, auc_k, accuracY, binary_accuracY, exact_accuracY = evaluate(model_k, X_test_k, Y_test_k)
        
#         # Print the results
#         print(f"Results for Top {k} Labels Model:")
#         print(f"Loss: {loss_k}")
#         print(f"AUC: {auc_k}")
#         print(f"Accuracy: {accuracY}")
#         print(f"Binary Accuracy: {binary_accuracY}")
#         print(f"Exact Accuracy: {exact_accuracY}\n")

    # # Ablation w/o Dominant Label
    # X_wo_dom, Y_wo_dom = remove_dominant_label(X, Y)
    # X_train_wo_dom, X_test_wo_dom, Y_train_wo_dom, Y_test_wo_dom = train_test_split(X_wo_dom, Y_wo_dom, test_size=0.2, random_state=42)
    # model_wo_dom, _ = train(get_model(X.shape[1], Y.shape[1]), X_train_wo_dom, Y_train_wo_dom)
    # loss_wo_dom, auc_wo_dom, accuracy_wo_dom, binary_accuracy_wo_dom, exact_accuracy_wo_dom = evaluate(model_wo_dom, X_test_wo_dom, Y_test_wo_dom)

    # # Ablation w/o PH Group 0
    # X_wo_ph0, Y_wo_ph0 = remove_ph_group(X, Y, 0)
    # X_train_wo_ph0, X_test_wo_ph0, Y_train_wo_ph0, Y_test_wo_ph0 = train_test_split(X_wo_ph0, Y_wo_ph0, test_size=0.2, random_state=42)
    # model_wo_ph0, _ = train(get_model(X_wo_ph0.shape[1], Y.shape[1]), X_train_wo_ph0, Y_train_wo_ph0)
    # loss_wo_ph0, auc_wo_ph0, accuracy_wo_ph0, binary_accuracy_wo_ph0, exact_accuracy_wo_ph0 = evaluate(model_wo_ph0, X_test_wo_ph0, Y_test_wo_ph0)

    # # Ablation w/o PH Group 1
    # X_wo_ph1, Y_wo_ph1 = remove_ph_group(X, Y, 1)
    # X_train_wo_ph1, X_test_wo_ph1, Y_train_wo_ph1, Y_test_wo_ph1 = train_test_split(X_wo_ph1, Y_wo_ph1, test_size=0.2, random_state=42)
    # model_wo_ph1, _ = train(get_model(X_wo_ph1.shape[1], Y.shape[1]), X_train_wo_ph1, Y_train_wo_ph1)
    # loss_wo_ph1, auc_wo_ph1, accuracy_wo_ph1, binary_accuracy_wo_ph1, exact_accuracy_wo_ph1 = evaluate(model_wo_ph1, X_test_wo_ph1, Y_test_wo_ph1)

    # # Sanity check -- only one column of PH
    # X_wo_everything, Y_wo_everything = X[:, 0:1], Y
    # X_train_wo_everything, X_test_wo_everything, Y_train_wo_everything, Y_test_wo_everything = train_test_split(X_wo_everything, Y_wo_everything, test_size=0.2, random_state=42)
    # model_wo_everything, _ = train(get_model(X_wo_everything.shape[1], Y.shape[1]), X_train_wo_everything, Y_train_wo_everything)
    # loss_wo_everything, auc_wo_everything, accuracy_wo_everything, binary_accuracy_wo_everything, exact_accuracy_wo_everything = evaluate(model_wo_everything, X_test_wo_everything, Y_test_wo_everything)

    # # Now print out all results including exact_accuracy for each scenario
    # print("Full Evaluation Results: Loss, AUC, Accuracy, Binary Accuracy, Exact Accuracy")
    # print(full_loss, full_auc, full_accuracy, full_binary_accuracy, full_exact_accuracy)
    
    # # Full model evaluation results
    # print("Full Model Evaluation:")
    # print(f"Loss: {full_loss}, AUC: {full_auc}, Accuracy: {full_accuracy}, Binary Accuracy: {full_binary_accuracy}, Exact Accuracy: {full_exact_accuracy}\n")

    # # Top k labels model evaluation results
    # print("Top 20 Labels Model Evaluation:")
    # print(f"Loss: {loss_15}, AUC: {auc_15}, Accuracy: {accuracy_15}, Binary Accuracy: {binary_accuracy_15}, Exact Accuracy: {exact_accuracy_15}\n")

    # # Ablation without Dominant Label model evaluation results
    # print("Ablation without Dominant Label Model Evaluation:")
    # print(f"Loss: {loss_wo_dom}, AUC: {auc_wo_dom}, Accuracy: {accuracy_wo_dom}, Binary Accuracy: {binary_accuracy_wo_dom}, Exact Accuracy: {exact_accuracy_wo_dom}\n")

    # # Ablation without PH Group 0 model evaluation results
    # print("Ablation without PH Group 0 Model Evaluation:")
    # print(f"Loss: {loss_wo_ph0}, AUC: {auc_wo_ph0}, Accuracy: {accuracy_wo_ph0}, Binary Accuracy: {binary_accuracy_wo_ph0}, Exact Accuracy: {exact_accuracy_wo_ph0}\n")

    # # Ablation without PH Group 1 model evaluation results
    # print("Ablation without PH Group 1 Model Evaluation:")
    # print(f"Loss: {loss_wo_ph1}, AUC: {auc_wo_ph1}, Accuracy: {accuracy_wo_ph1}, Binary Accuracy: {binary_accuracy_wo_ph1}, Exact Accuracy: {exact_accuracy_wo_ph1}\n")

    # # Sanity check (only one column of PH) model evaluation results
    # print("Sanity Check (Only One Column of PH) Model Evaluation:")
    # print(f"Loss: {loss_wo_everything}, AUC: {auc_wo_everything}, Accuracy: {accuracy_wo_everything}, Binary Accuracy: {binary_accuracy_wo_everything}, Exact Accuracy: {exact_accuracy_wo_everything}\n")



def run_hyperparameter_tuning(X, Y, tag=''):
    """
    Run hyperparameter tuning for a given dataset.

    Parameters:
    X (numpy.ndarray): The input features.
    Y (numpy.ndarray): The target labels.

    Returns:
    None

    Saves best model to path `best_model_{tag}` and best hyperparameters to `best_hyperparameters_{tag}`
    """
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Define the hyperparameters to tune
    hyperparameters = {
        "hidden_size": [32, 64, 128],
        "hidden_layers": [1, 2, 3],
        "learning_rate": [0.001, 0.005, 0.01]
    }

    # Initialize the best hyperparameters
    best_hyperparameters = {}
    best_auc = 0
    best_exact_acc = 0
    best_binary_acc = 0

    # Make the training dir for the results if needed
    os.makedirs("training", exist_ok=True)

    # Loop through all hyperparameter combinations
    for hidden_size in hyperparameters["hidden_size"]:
        for hidden_layers in hyperparameters["hidden_layers"]:
            for learning_rate in hyperparameters["learning_rate"]:
                # Create the model
                model = get_model(X.shape[1], Y.shape[1], hidden_size, hidden_layers)
                # Compile the model
                model, history = train(model, X_train, Y_train, learning_rate)
                # Evaluate the model
                _, auc, _, binary_accuracy, exact_accuracy = model.evaluate(X_test, Y_test)

                # Items to save
                out_dict = {
                            "auc": float(auc),
                            "exact_accuracy": float(exact_accuracy),
                            "binary_accuracy": float(binary_accuracy),
                            "hidden_size": hidden_size,
                            "hidden_layers": hidden_layers,
                            "learning_rate": learning_rate,
                            "history": str(history.history)
                        }
                with open(f"training/metrics_{hidden_size}_{hidden_layers}_{learning_rate}_{tag}.json", "w") as f:
                    json.dump(out_dict, f)
                
                # Update the best hyperparameters if necessary
                if auc > best_auc:
                    best_auc = auc
                    best_exact_acc = exact_accuracy
                    best_binary_acc = binary_accuracy

                    best_hyperparameters = {
                        "hidden_size": hidden_size,
                        "hidden_layers": hidden_layers,
                        "learning_rate": learning_rate
                    }

                    # Save the best model
                    model.save(f"training/best_model_{tag}")

                    # Save the best hyperparmeters and the metrics
                    with open(f"training/best_hyperparameters_{tag}.json", "w") as f:
                        json.dump(out_dict, f)

    print("Best Hyperparameters:", best_hyperparameters)
    print("Best AUC:", best_auc)
    print("Best Exact Accuracy:", best_exact_acc)
    print("Best Binary Accuracy:", best_binary_acc)

    print("Completed hyperparameter tuning")

def run_ablation_evaluation(X, Y, X_k_wo_dom, Y_k_wo_dom, tag=''):
    """
    Train all ablations and evaluate them based on the best hyperparameters (saved as json in the folder `training/best_hyperparmameters_{tag}.json`).
    """

    # Load the best hyperparameters
    with open(f"training/best_hyperparameters_{tag}.json", "r") as f:
        best_hyperparameters = json.load(f)

    # Ablation w/o Dominant Label
    X_wo_dom, Y_wo_dom = X_k_wo_dom, Y_k_wo_dom
    X_train_wo_dom, X_test_wo_dom, Y_train_wo_dom, Y_test_wo_dom = train_test_split(X_wo_dom, Y_wo_dom, test_size=0.2, random_state=42)
    model_wo_dom, _ = train(get_model(X.shape[1], Y.shape[1], best_hyperparameters["hidden_size"], best_hyperparameters["hidden_layers"]),
                            X_train_wo_dom,
                            Y_train_wo_dom,
                            best_hyperparameters["learning_rate"])

    loss_wo_dom, auc_wo_dom, accuracy_wo_dom, binary_accuracy_wo_dom, exact_accuracy_wo_dom = evaluate(model_wo_dom, X_test_wo_dom, Y_test_wo_dom)

    # Ablation w/o PH Group 0
    X_wo_ph0, Y_wo_ph0 = remove_ph_group(X, Y, 0)
    X_train_wo_ph0, X_test_wo_ph0, Y_train_wo_ph0, Y_test_wo_ph0 = train_test_split(X_wo_ph0, Y_wo_ph0, test_size=0.2, random_state=42)
    model_wo_ph0, _ = train(get_model(X_wo_ph0.shape[1], Y_wo_ph0.shape[1], best_hyperparameters["hidden_size"], best_hyperparameters["hidden_layers"]),
                            X_train_wo_ph0, 
                            Y_train_wo_ph0,
                            best_hyperparameters["learning_rate"])
    loss_wo_ph0, auc_wo_ph0, accuracy_wo_ph0, binary_accuracy_wo_ph0, exact_accuracy_wo_ph0 = evaluate(model_wo_ph0, X_test_wo_ph0, Y_test_wo_ph0)

    # Ablation w/o PH Group 1
    X_wo_ph1, Y_wo_ph1 = remove_ph_group(X, Y, 1)
    X_train_wo_ph1, X_test_wo_ph1, Y_train_wo_ph1, Y_test_wo_ph1 = train_test_split(X_wo_ph1, Y_wo_ph1, test_size=0.2, random_state=42)
    model_wo_ph1, _ = train(get_model(X_wo_ph1.shape[1], Y_wo_ph1.shape[1], best_hyperparameters["hidden_size"], best_hyperparameters["hidden_layers"]),
                            X_train_wo_ph1, 
                            Y_train_wo_ph1,
                            best_hyperparameters["learning_rate"])
    loss_wo_ph1, auc_wo_ph1, accuracy_wo_ph1, binary_accuracy_wo_ph1, exact_accuracy_wo_ph1 = evaluate(model_wo_ph1, X_test_wo_ph1, Y_test_wo_ph1)

    # Ablation w/o PH Group 2
    X_wo_ph2, Y_wo_ph2 = remove_ph_group(X, Y, 2)
    X_train_wo_ph2, X_test_wo_ph2, Y_train_wo_ph2, Y_test_wo_ph2 = train_test_split(X_wo_ph2, Y_wo_ph2, test_size=0.2, random_state=42)
    model_wo_ph2, _ = train(get_model(X_wo_ph2.shape[1], Y_wo_ph2.shape[1], best_hyperparameters["hidden_size"], best_hyperparameters["hidden_layers"]),
                            X_train_wo_ph2, 
                            Y_train_wo_ph2,
                            best_hyperparameters["learning_rate"])
    loss_wo_ph2, auc_wo_ph2, accuracy_wo_ph2, binary_accuracy_wo_ph2, exact_accuracy_wo_ph2 = evaluate(model_wo_ph2, X_test_wo_ph2, Y_test_wo_ph2)


    # Now print out all results including exact_accuracy for each scenario
    print("Results for w/o Dominant Label")
    print(f"Loss: {loss_wo_dom}, AUC: {auc_wo_dom}, Accuracy: {accuracy_wo_dom}, Binary Accuracy: {binary_accuracy_wo_dom}, Exact Accuracy: {exact_accuracy_wo_dom}\n")

    print("Results for w/o PH Group 0")
    print(f"Loss: {loss_wo_ph0}, AUC: {auc_wo_ph0}, Accuracy: {accuracy_wo_ph0}, Binary Accuracy: {binary_accuracy_wo_ph0}, Exact Accuracy: {exact_accuracy_wo_ph0}\n")

    print("Results for w/o PH Group 1")
    print(f"Loss: {loss_wo_ph1}, AUC: {auc_wo_ph1}, Accuracy: {accuracy_wo_ph1}, Binary Accuracy: {binary_accuracy_wo_ph1}, Exact Accuracy: {exact_accuracy_wo_ph1}\n")

    print("Results for w/o PH Group 2")
    print(f"Loss: {loss_wo_ph2}, AUC: {auc_wo_ph2}, Accuracy: {accuracy_wo_ph2}, Binary Accuracy: {binary_accuracy_wo_ph2}, Exact Accuracy: {exact_accuracy_wo_ph2}\n")

    # Save ablation results to json
    out_dict = {
        "w/o Dominant Label": {
            "loss": float(loss_wo_dom),
            "auc": float(auc_wo_dom),
            "accuracy": float(accuracy_wo_dom),
            "binary_accuracy": float(binary_accuracy_wo_dom),
            "exact_accuracy": float(exact_accuracy_wo_dom)
        },
        "w/o PH Group 0": {
            "loss": float(loss_wo_ph0),
            "auc": float(auc_wo_ph0),
            "accuracy": float(accuracy_wo_ph0),
            "binary_accuracy": float(binary_accuracy_wo_ph0),
            "exact_accuracy": float(exact_accuracy_wo_ph0)
        },
        "w/o PH Group 1": {
            "loss": float(loss_wo_ph1),
            "auc": float(auc_wo_ph1),
            "accuracy": float(accuracy_wo_ph1),
            "binary_accuracy": float(binary_accuracy_wo_ph1),
            "exact_accuracy": float(exact_accuracy_wo_ph1)
        },
        "w/o PH Group 2": {
            "loss": float(loss_wo_ph2),
            "auc": float(auc_wo_ph2),
            "accuracy": float(accuracy_wo_ph2),
            "binary_accuracy": float(binary_accuracy_wo_ph2),
            "exact_accuracy": float(exact_accuracy_wo_ph2)
        }
    }

    with open(f"training/ablation_results_{tag}.json", "w") as f:
        json.dump(out_dict, f)

    print("Completed ablation evaluation")

if __name__ == "__main__":
    # Path to CSV file containing PDB ids used for GeoGNN Training. Was used to ensure that we only ran PH on PDB ids we had GeoGNN embeddings for.
    # ids_path = "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/PaddleHelix/apps/pretrained_compound/ChemRL/GEM/finetune_models/knot/ids_in_order.csv"
    # Path to the labels file for the dataset (Y)
    # labels_path = "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/PaddleHelix/apps/pretrained_compound/ChemRL/GEM/finetune_models/knot/labels.npy"
    # Path to the PH embeddings for the dataset (X_ph)
    x_ph_path = "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/dataset/human_proteins/human_protein_ph_h2_files"
    # Path to the GeoGNN embeddings for the dataset (X_geo)
    # x_geo_path = "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/PaddleHelix/apps/pretrained_compound/ChemRL/GEM/finetune_models/knot/graph_encodings.npy"
    # Path to the new labels file
    labels_path = "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/dataset/human_proteins/human_protein_mf_processed_intersection.txt"
    
    X, Y = create_dataset_from_scratch(labels_path, x_ph_path)
    X_k, Y_k = reduce_dataset_k_labels(X, Y, 5)
    X_k_wo_dom, Y_k_wo_dom = reduce_dataset_k_labels(*remove_dominant_label(X, Y), 5)

    run_hyperparameter_tuning(X_k, Y_k, tag='5_labels')
    run_ablation_evaluation(X_k, Y_k, X_k_wo_dom, Y_k_wo_dom, tag='5_labels')