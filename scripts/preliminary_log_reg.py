"""
Preliminary attempt at logistic regression classification of molecular function from knot type.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from utils import make_data

# Make the dataset
X, Y, metadata = make_data('./catalogs/knotprot_mf_processed.txt')

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model for multi-label classification
logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
ovr = OneVsRestClassifier(logreg)

# Train the model
ovr.fit(X_train, Y_train)

# Predict on the test set
Y_pred = ovr.predict(X_test)

# Evaluate the model
import matplotlib.pyplot as plt
import numpy as np

def get_results(true_labels, pred_labels, label_names):
    results = {}
    for i, name in enumerate(label_names):
        correct = 0
        incorrect = 0
        for j in range(len(true_labels)):
            if true_labels[j][i] == pred_labels[j][i]:
                correct += 1
            else:
                incorrect += 1
        results[name] = {'correct': correct, 'incorrect': incorrect}
    return results

def get_knot_type_results(X, true_labels, pred_labels, knot_types):
    results = {knot_type: {'correct': 0, 'incorrect': 0} for knot_type in knot_types}
    for i in range(len(X)):
        knot_type_index = np.argmax(X[i])  # Assuming one-hot encoding
        knot_type = knot_types[knot_type_index]
        if np.array_equal(true_labels[i], pred_labels[i]):
            results[knot_type]['correct'] += 1
        else:
            results[knot_type]['incorrect'] += 1
    return results


knot_types, molecular_functions = metadata
mf_results = get_results(Y_test, Y_pred, molecular_functions)
knot_results = get_knot_type_results(X_test, Y_test, Y_pred, knot_types)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.bar(knot_results.keys(), [r['correct'] for r in knot_results.values()], label='Correct')
ax1.bar(knot_results.keys(), [r['incorrect'] for r in knot_results.values()])
ax1.set_title('Knot Types')
ax1.set_xlabel('Knot Type')
ax1.set_ylabel('Number of Instances')
ax1.legend()

ax2.bar(mf_results.keys(), [r['correct'] for r in mf_results.values()], label='Correct')
ax2.bar(mf_results.keys(), [r['incorrect'] for r in mf_results.values()])
ax2.set_title('Molecular Functions')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.set_xlabel('Molecular Function')
ax2.set_ylabel('Number of Instances')
ax2.legend()



plt.savefig('knotfun_results.png')

print(classification_report(Y_test, Y_pred))