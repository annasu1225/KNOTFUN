import os
import json
import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, auc, roc_curve

tf.compat.v1.enable_eager_execution()

def convert_to_python_types(obj):
    if isinstance(obj, np.generic):
        return np.asscalar(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert arrays to list
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)  # Convert numpy floats to Python float
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)  # Convert numpy ints to Python int
    else:
        return obj


def create_dataset_from_scratch(label_path, ph_folder, pb_file):
    """
    Create the PH only dataset (X, Y)

    Input:
    - label_path: Path to the labels.txt file.
    - ph_folder: Path to the folder containing PH vectors.
    - pb_file: Path to the file containing pb vectors

    Returns:
    - (X, Y) where X is a numpy array of PH vectors, and Y is a numpy array of associated molecular functions, multi-hot encoded.
    """

    functions_dict = {}
    with open(label_path, 'r') as f:
        for line in f.readlines():
            #id_ = eval(line.split(',', 3)[-1].split(': ', 1)[-1])['id']
            
            pdb_id_index = line.find("PDB ID: ") + 8
            pdb_id_end_index = line.find(",", pdb_id_index)
            id_ = line[pdb_id_index:pdb_id_end_index].strip()

            functions = eval(line.split(',', 3)[-1].split(': ', 1)[-1])['molecular_functions']
            functions_dict[id_] = functions

    # Initialize lists to hold our PH vectors (X) and labels (Y)
    X, Y = [], []
    all_functions = set()

    if pb_file:
        print("Also making from pb")
        fastas = pd.read_csv(pb_file)
    
    # Load PH vectors and associate them with molecular functions
    for id_, functions in functions_dict.items():
        ph_vec_path = os.path.join(ph_folder, id_, id_ + "_ph_vec.npy")
        pb_vec_path = fastas[fastas['PDB ID'] == id_].iloc[0]['pb_path'] # Get the first b/c dont care ab duplicates

        if os.path.exists(ph_vec_path) and (not pb_file or os.path.exists(pb_vec_path)):
            ph_vec = np.load(ph_vec_path)
            
            if pb_file:
                pb_vec = np.load(pb_vec_path)

            if ph_vec.size > 0 and (not pb_file or pb_vec.size > 0):  # Check if the array is non-empty
                ph_vec = ph_vec.flatten()
                if pb_file:
                    pb_vec = pb_vec.flatten()
                    X.append(np.concatenate([ph_vec, pb_vec]))
                else:
                    X.append(ph_vec)

                Y.append(functions)  # Temporarily store the list of functions
                all_functions.update(functions)
            else:
                print(f"{id_} vector is empty")  # Indicate that the array is empty
        else:
            print(f"Missing path for {id_}")

    functions_dict = {k: functions_dict[k] for k in sorted(functions_dict)}

    # Convert all_functions to a list and map each function to an index
    all_functions = sorted(list(all_functions))
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

    print("Completed create_dataset_from_scratch")

    return X, Y_encoded, all_functions

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

def train(model, X_train, Y_train, hyperparameters):
    """
    Train the TF model on the dataset. Evaluate on ROC-AUC
    """
    
    learning_rate = hyperparameters['learning_rate']
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='binary_crossentropy', 
                  metrics=[tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Accuracy(name='accuracy'), tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'), exact_match_accuracy])

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

def detailed_evaluate(model, X_test, Y_test, labels):
    """
    Detailed evaluation including per-label accuracy.
    """
    loss, auc, accuracy, binary_accuracy, exact_accuracy = model.evaluate(X_test, Y_test)
    y_pred = model.predict(X_test)
    per_label_acc_dict = per_label_accuracy(Y_test, y_pred, labels)
    return loss, auc, accuracy, binary_accuracy, exact_accuracy, per_label_acc_dict

def per_label_accuracy(y_true, y_pred, labels):
    """
    Compute the accuracy for each label in a multi-label classification and include label names.
    
    Parameters:
    - y_true: TensorFlow tensor, the true labels.
    - y_pred: TensorFlow tensor, the predicted labels (as probabilities).
    - labels: List of strings, the names corresponding to each label index.
    
    Returns:
    - accuracies: Dictionary with label names as keys and tuple (accuracy, total_count, correct_count) as values.
    """

    # Ensure both tensors are the same type before comparison
    y_true = tf.cast(y_true, tf.float32)
    # Convert predictions to binary by thresholding at 0.5
    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
    
    # Calculate the correct predictions for each label
    correct_predictions = tf.cast(tf.equal(y_true, y_pred_binary), tf.float32)
    # Calculate total counts and correct counts for each label
    total_counts = tf.reduce_sum(y_true, axis=0)
    correct_counts = tf.reduce_sum(correct_predictions, axis=0)
    # Calculate accuracies for each label
    accuracies = tf.where(
        total_counts > 0,
        correct_counts / total_counts,
        tf.zeros_like(total_counts)  # Handle division by zero by setting accuracies to zero where total_counts is zero
    )
    
    # Convert tensors to numpy for easier manipulation and returning
    accuracies_np = accuracies.numpy()
    total_counts_np = total_counts.numpy()
    correct_counts_np = correct_counts.numpy()
    
    # Prepare dictionary to return
    results_dict = {}
    for i, label in enumerate(labels):
        results_dict[label] = (accuracies_np[i], int(total_counts_np[i]), int(correct_counts_np[i]))
        print(f"{label}: Accuracies: {results_dict[label][0]}, Total counts: {results_dict[label][1]}, Correct counts: {results_dict[label][2]}")
    
    return results_dict

def get_model(input_size, output_size, hyperparameters):
    """
    Get the MLP model used.
    """

    hidden_size = hyperparameters['hidden_size']
    hidden_layers = hyperparameters['hidden_layers']
    
    layers = [Dense(hidden_size, activation='relu', input_shape=(input_size,)), ]

    for layer in range(hidden_layers):
        layers.append(Dense(hidden_size, activation='relu'),)
    
    layers.append(Dense(output_size, activation='sigmoid'))

    model = Sequential(layers)

    return model

def get_dominant_label_index(X, Y, labels):
    # Count the frequency of each label
    label_frequencies = np.sum(Y, axis=0)
    
    # Identify the index of the most dominant label
    dominant_label_index = np.argmax(label_frequencies)
    print("Dominant label index:", dominant_label_index)
    print("Dominant label:", labels[dominant_label_index])
    
    return dominant_label_index

def remove_dominant_label(X, Y, dominant_label_index):
    """
    Filter the dataset to generate a subset only including samples unassociated with the dominant label
    """
    # Remove all samples that have the dominant label from the dataset
    X_wo = X[Y[:, dominant_label_index] == 0]
    Y_wo = Y[Y[:, dominant_label_index] == 0]

    print("Completed remove_dominant_label")

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
    
    print("Completed reduce_dataset_k_labels")
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


def run_hyperparameter_tuning(X, Y, tag, date):
    """
    Run hyperparameter tuning for a given dataset.
    Saves best model to path `best_model_{tag}` and best hyperparameters to `best_hyperparameters_{tag}`
    """
    print("Start hyperparameter tuning")

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
    os.makedirs(f"training_{date}_overnight", exist_ok=True)

    # Loop through all hyperparameter combinations
    for hidden_size in hyperparameters["hidden_size"]:
        for hidden_layers in hyperparameters["hidden_layers"]:
            for learning_rate in hyperparameters["learning_rate"]:
                cur_hypers = {'hidden_size': hidden_size, 'hidden_layers': hidden_layers, 'learning_rate': learning_rate}
                # Create the model
                model = get_model(X.shape[1], Y.shape[1], cur_hypers)
                # Compile the model
                model, history = train(model, X_train, Y_train, cur_hypers)
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
                with open(f"training_{date}_overnight/metrics_{hidden_size}_{hidden_layers}_{learning_rate}_{tag}.json", "w") as f:
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
                    model.save(f"training_{date}_overnight/best_model_{tag}",)

                    # Save the best hyperparmeters and the metrics
                    with open(f"training_{date}_overnight/best_hyperparameters_{tag}.json", "w") as f:
                        json.dump(out_dict, f)

    print("Best Hyperparameters:", best_hyperparameters)
    print("Best AUC:", best_auc)
    print("Best Exact Accuracy:", best_exact_acc)
    print("Best Binary Accuracy:", best_binary_acc)

    print("Completed hyperparameter tuning")

    return best_hyperparameters

def load_hyperparameters(json_file_path):
    with open(json_file_path, 'r') as file:
        hyperparameters = json.load(file)
    return hyperparameters

def run_per_label_evaluation(X, Y, labels, tag, date, model):
    '''
    Outputs detailed per label accuracy (saved as json in the folder `training_{date}_overnight/detailed_evaluation.json`)
    Parameters:
        X: Features dataset.
        Y: Labels dataset.
        labels: List of label names.
        tag: Tag used for saving models and results.
        date: Date string used for folder naming.
        model: Pre-loaded or pre-trained TensorFlow/Keras model.
    '''

    print("Start per label evaluation")

    # Assuming hyperparameters and best_model are available from tuning
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Perform detailed evaluation
    loss, auc, accuracy, binary_accuracy, exact_accuracy, per_label_acc_dict = detailed_evaluate(model, X_test, Y_test, labels)

    # Prepare results for saving
    results = {
        "loss": convert_to_python_types(loss),
        "auc": convert_to_python_types(auc),
        "accuracy": convert_to_python_types(accuracy),
        "binary_accuracy": convert_to_python_types(binary_accuracy),
        "exact_accuracy": convert_to_python_types(exact_accuracy),
        "per_label_accuracy": {k: tuple(convert_to_python_types(v) for v in values) for k, values in per_label_acc_dict.items()}
    }

    # Save to JSON file
    with open(f'training_{date}_overnight/detailed_evaluation.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("Completed per label evaluation")

def run_ablation_evaluation(X, Y, X_k_wo_dom, Y_k_wo_dom, tag, date, hyperparameters):
    """
    Train all ablations and evaluate them based on the best hyperparameters (saved as json in the folder `training/best_hyperparmameters_{tag}.json`).
    """

    print("Start ablation evaluation")

    # Ablation w/o Dominant Label
    print("Start Ablation w/o Dominant Label")
    X_wo_dom, Y_wo_dom = X_k_wo_dom, Y_k_wo_dom
    X_train_wo_dom, X_test_wo_dom, Y_train_wo_dom, Y_test_wo_dom = train_test_split(X_wo_dom, Y_wo_dom, test_size=0.2, random_state=42)
    model_wo_dom, _ = train(get_model(X.shape[1], Y.shape[1], hyperparameters),
                            X_train_wo_dom,
                            Y_train_wo_dom,
                            hyperparameters)
    loss_wo_dom, auc_wo_dom, accuracy_wo_dom, binary_accuracy_wo_dom, exact_accuracy_wo_dom = evaluate(model_wo_dom, X_test_wo_dom, Y_test_wo_dom)
    print("Completed Ablation w/o Dominant Label")

    # Ablation w/o PH Group 0
    print("Start Ablation w/o PH Group 0")
    X_wo_ph0, Y_wo_ph0 = remove_ph_group(X, Y, 0)
    X_train_wo_ph0, X_test_wo_ph0, Y_train_wo_ph0, Y_test_wo_ph0 = train_test_split(X_wo_ph0, Y_wo_ph0, test_size=0.2, random_state=42)
    model_wo_ph0, _ = train(get_model(X_wo_ph0.shape[1], Y_wo_ph0.shape[1], hyperparameters),
                            X_train_wo_ph0, 
                            Y_train_wo_ph0,
                            hyperparameters)
    loss_wo_ph0, auc_wo_ph0, accuracy_wo_ph0, binary_accuracy_wo_ph0, exact_accuracy_wo_ph0 = evaluate(model_wo_ph0, X_test_wo_ph0, Y_test_wo_ph0)
    print("Completed Ablation w/o PH Group 0")

    # Ablation w/o PH Group 1
    print("Start Ablation w/o PH Group 1")
    X_wo_ph1, Y_wo_ph1 = remove_ph_group(X, Y, 1)
    X_train_wo_ph1, X_test_wo_ph1, Y_train_wo_ph1, Y_test_wo_ph1 = train_test_split(X_wo_ph1, Y_wo_ph1, test_size=0.2, random_state=42)
    model_wo_ph1, _ = train(get_model(X_wo_ph1.shape[1], Y_wo_ph1.shape[1], hyperparameters),
                            X_train_wo_ph1, 
                            Y_train_wo_ph1,
                            hyperparameters)
    loss_wo_ph1, auc_wo_ph1, accuracy_wo_ph1, binary_accuracy_wo_ph1, exact_accuracy_wo_ph1 = evaluate(model_wo_ph1, X_test_wo_ph1, Y_test_wo_ph1)
    print("Completed Ablation w/o PH Group 1")

    # Ablation w/o PH Group 2
    print("Start Ablation w/o PH Group 2")
    X_wo_ph2, Y_wo_ph2 = remove_ph_group(X, Y, 2)
    X_train_wo_ph2, X_test_wo_ph2, Y_train_wo_ph2, Y_test_wo_ph2 = train_test_split(X_wo_ph2, Y_wo_ph2, test_size=0.2, random_state=42)
    model_wo_ph2, _ = train(get_model(X_wo_ph2.shape[1], Y_wo_ph2.shape[1], hyperparameters),
                            X_train_wo_ph2, 
                            Y_train_wo_ph2,
                            hyperparameters)
    loss_wo_ph2, auc_wo_ph2, accuracy_wo_ph2, binary_accuracy_wo_ph2, exact_accuracy_wo_ph2 = evaluate(model_wo_ph2, X_test_wo_ph2, Y_test_wo_ph2)
    print("Completed Ablation w/o PH Group 2")

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

    with open(f"training_{date}_overnight/ablation_results_{tag}.json", "w") as f:
        json.dump(out_dict, f)

    print("Completed ablation evaluation")

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate MLP_persistent homology models')
    parser.add_argument('--ph_path', default=None, help='Persistent homology files')
    parser.add_argument('--labels_path', default=None, help='Molecular function labels file')
    parser.add_argument('--pb_file', default=None, help='Protein Bert Embeddings File')
    parser.add_argument('--top_k', type=int, help='Top k most frequent labels')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    parser.add_argument('--model_path', help='Path to the model', default=None)
    parser.add_argument('--hyperparameters_path', help='Path to the JSON file with best hyperparameters', default=None)
    parser.add_argument('--per_label_eval', action='store_true', help='Run per label evaluation')
    parser.add_argument('--ablation', action='store_true', help='Run ablation study')
    parser.add_argument('--from_scratch', help='Create dataset from scratch', default=False)
    parser.add_argument('--x_file', help='File to X data (or to save them to depending on --from_scratch)', default=None)
    parser.add_argument('--y_file', help='File to Y labels (or to save them to depending on --from_scratch)', default=None)
    parser.add_argument('--labels_file', help='File to Y label names (or to save them to depending on --from_scratch)', default=None)
    parser.add_argument('--pb_only', action='store_true', help='Whether or not to train on ProteinBERT embeddings only')
    parser.add_argument('--ph_only', action='store_true', help='Whether or not to train on PersLay embeddings only')

    args = parser.parse_args()

    # Create dataset from labels and PH vectors
    if args.from_scratch:
        X, Y, labels = create_dataset_from_scratch(args.labels_path, args.ph_path, args.pb_file)
        pd.DataFrame(labels).to_csv(args.labels_file, sep=',', index=False, header=False)
        np.save(args.x_file, X)
        np.save(args.y_file, Y)
    else:
        X, Y = np.load(args.x_file), np.load(args.y_file)
        labels = pd.read_csv(args.labels_file).iloc[:,0].to_list()
    
    if args.pb_only:
        PH_FEATURE_SIZE = 27000
        X = X[:,PH_FEATURE_SIZE:] # cut off the PH features so we are only left w/ PB ones
        print("X Shape", X.shape)
    
    # Automatically set the date to today's date in YYYY_MM_DD format
    today_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # Make the training dir for the results if needed
    os.makedirs(f"training_{today_date}", exist_ok=True)
    # Format tag with top_k value
    tag = f"{args.top_k}_labels"

    # Load hyperparameters if path is provided and valid
    if args.hyperparameters_path and os.path.exists(args.hyperparameters_path):
        hyperparameters = load_hyperparameters(args.hyperparameters_path)
    else:
        print("Running hyperparameter tuning")
        hyperparameters = run_hyperparameter_tuning(X, Y, tag, today_date)
    
    # Load model if path is provided and valid
    if args.model_path and os.path.exists(args.model_path):
        custom_objects = {'exact_match_accuracy': exact_match_accuracy}
        model = load_model(args.model_path, custom_objects=custom_objects)
        print("Model loaded")
    else:
        print("Training model...")
        model = get_model(X.shape[1], Y.shape[1], hyperparameters)
        model, _ = train(model, X, Y, hyperparameters)
        model.save(f"training_{today_date}/best_model_{tag}")
        print("Finished training. Model saved.")

    if args.per_label_eval:
        run_per_label_evaluation(X, Y, labels, tag, today_date, model)
    
    if args.ablation:
        # Get the dominant label index
        dominant_label_index = get_dominant_label_index(X, Y, labels)
        # Dataset 1: k most frequent labels
        X_k, Y_k = reduce_dataset_k_labels(X, Y, args.top_k)
        # Dataset 2: k most frequent labels after excluding the dominant label
        X_k_wo_dom, Y_k_wo_dom = reduce_dataset_k_labels(*(remove_dominant_label(X, Y, dominant_label_index)), args.top_k)
        run_ablation_evaluation(X_k, Y_k, X_k_wo_dom, Y_k_wo_dom, tag, today_date, hyperparameters)

if __name__ == '__main__':
    main()

# python train_tf.py "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/dataset/human_proteins/human_protein_ph_h2_files" "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/dataset/human_proteins/human_protein_mf_final.txt" 123 --model_path "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/training_2024_04_17/best_model_123_labels" --hyperparameters_path "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/training_2024_04_13/best_hyperparameters_5_labels.json" --ablation 2>&1 | tee train_tf_output.txt
# python train_tf.py "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/dataset/knotted_proteins/knotted_protein_ph_h2_files" "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/dataset/knotted_proteins/knotted_mf_combined_final.txt" 123 --model_path "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/training_2024_04_18/best_model_123_labels" --hyperparameters_path "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/training_2024_04_18/best_hyperparameters_123_labels.json" --ablation 2>&1 | tee train_tf_knotted_output.txt
'''
Human Protein Dataset Paths
# Path to the PH embeddings for the dataset (X_ph)
x_ph_path = "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/dataset/human_proteins/human_protein_ph_h2_files"
# Path to the labels file
labels_path = "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/dataset/human_proteins/human_protein_mf_final.txt"
# Path to best model 
model_path = "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/training_2024_04_17/best_model_123_labels"
# Path to best hyperparameters
hyperparameters_path = "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/training_2024_04_13/best_hyperparameters_5_labels.json"
'''

'''
Knotted Protein Dataset Paths
# Path to the PH embeddings for the dataset (X_ph)
x_ph_path = "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/dataset/knotted_proteins/knotted_protein_ph_h2_files"
# Path to the labels file
labels_path = "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/dataset/knotted_proteins/knotted_mf_combined_final.txt"
'''

# x_ph_path = "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/dataset/knotted_proteins/knotted_protein_ph_h2_files"
# labels_path = "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/dataset/knotted_proteins/knotted_mf_combined_final.txt"
# hyperparameters_path = "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/training_2024_04_13/best_hyperparameters_5_labels.json"

# X, Y, labels = create_dataset_from_scratch(labels_path, x_ph_path)

# hyperparameters = load_hyperparameters(hyperparameters_path)

# # custom_objects = {'exact_match_accuracy': exact_match_accuracy}
# # model = load_model(model_path, custom_objects=custom_objects)
# # print("Model loaded")

# dominant_label_index = get_dominant_label_index(X, Y, labels)

# # Dataset 1: 5 most frequent labels
# #X_k, Y_k = reduce_dataset_k_labels(X, Y, 5)
# X_k, Y_k = X, Y

# # Dataset 2: 5 most frequent labels after excluding the dominant label
# #X_k_wo_dom, Y_k_wo_dom = reduce_dataset_k_labels(*(remove_dominant_label(X, Y, dominant_label_index)), 5)
# X_k_wo_dom, Y_k_wo_dom = remove_dominant_label(X, Y, dominant_label_index)

# # run_hyperparameter_tuning(X_k, Y_k, tag='5_labels', date='2024_04_13')
# # run_per_label_evaluation(X, Y, labels, '123_labels', '2024_04_17', model)
# run_ablation_evaluation(X_k, Y_k, X_k_wo_dom, Y_k_wo_dom, '123_labels', '2024_04_17', hyperparameters)