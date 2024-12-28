"""
This is a boilerplate pipeline 'quantum_n_n'
generated using Kedro 0.19.10
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml
import pennylane.numpy as pnp
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)
from pennylane.optimize import AdamOptimizer
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Spacer


def define_quantum_network_1(n_qubits):
    # Assignment of a quantum device
    dev = qml.device("default.qubit", wires=n_qubits)

    # Quantum layer
    def quantum_layer(weights):
        for i in range(n_qubits):
            qml.Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])

        for i in range(n_qubits):
            qml.RY(weights[i, 0], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])

    # Quantum model
    @qml.qnode(dev)
    def quantum_neural_network(inputs, weights):
        for i in range(n_qubits):
            qml.RX(inputs[i % len(inputs)], wires=i)
        quantum_layer(weights)
        return qml.expval(qml.PauliZ(0))

    return quantum_neural_network


def train_quantum_model_1(
    X_train,
    y_train,
    quantum_neural_network=define_quantum_network_1(4),
    n_qubits=4,
    steps=200,
    stepsize=0.1,
):
    # Cost function
    def cost(weights, X, y):
        predictions = [quantum_neural_network(X[i], weights) for i in range(len(X))]
        return pnp.mean((pnp.array(predictions) - pnp.array(y)) ** 2)

    # Reduction of the input dimension
    def preprocess(X):
        return np.tanh(np.mean(X, axis=1)).reshape(-1, 1)

    # Data preprocessing
    X_train = X_train.to_numpy()
    X_train_reduced = preprocess(X_train)

    # Parameter initialization
    weights = pnp.random.randn(n_qubits, 3, requires_grad=True)

    # Training the model
    opt = AdamOptimizer(stepsize=stepsize)
    for step in range(steps):
        weights, cost_val = opt.step_and_cost(
            lambda w: cost(w, X_train_reduced, y_train), weights
        )
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}, Cost = {cost_val:.4f}")

    return weights


def validate_quantum_model_1(X_val, y_val, quantum_neural_network, weights):
    # Reduction of the input dimension
    def preprocess(X):
        return np.tanh(np.mean(X, axis=1)).reshape(-1, 1)

    X_val = X_val.to_numpy()
    X_val_reduced = preprocess(X_val)

    predictions = [
        quantum_neural_network(X_val_reduced[i], weights)
        for i in range(len(X_val_reduced))
    ]
    mse = np.mean((np.array(predictions) - np.array(y_val)) ** 2)
    print(f"Validation MSE: {mse:.4f}")
    return mse


def define_quantum_network_2(n_qubits, n_layers):
    # Assignment of a quantum device
    dev = qml.device("default.qubit", wires=n_qubits)

    # Quantum layer using AngleEmbedding and BasicEntanglerLayers
    def quantum_layer(weights, inputs):
        # Encoding the input data using AngleEmbedding
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")

        # Quantum layers with BasicEntanglerLayers
        qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))

    # Quantum model
    @qml.qnode(dev)
    def quantum_neural_network(inputs, weights):
        quantum_layer(weights, inputs)
        return qml.expval(qml.PauliZ(0))

    return quantum_neural_network


def train_quantum_model_2(
    X_train,
    y_train,
    quantum_neural_network=define_quantum_network_2(4, 2),
    n_qubits=4,
    n_layers=2,
    steps=200,
    stepsize=0.1,
):
    # Cost function
    def cost(weights, X, y):
        predictions = [quantum_neural_network(X[i], weights) for i in range(len(X))]
        return pnp.mean((pnp.array(predictions) - pnp.array(y)) ** 2)

    # Reduction of the input dimension
    def preprocess(X):
        return np.tanh(np.mean(X, axis=1)).reshape(-1, 1)

    # Data preprocessing
    X_train = X_train.to_numpy()
    X_train_reduced = preprocess(X_train)

    # Parameter initialization for BasicEntanglerLayers
    weights_shape = (n_layers, n_qubits)
    weights = pnp.random.randn(*weights_shape, requires_grad=True)

    # Training the model
    opt = AdamOptimizer(stepsize=stepsize)
    for step in range(steps):
        weights, cost_val = opt.step_and_cost(
            lambda w: cost(w, X_train_reduced, y_train), weights
        )
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}, Cost = {cost_val:.4f}")

    return weights


def validate_quantum_model_2(X_val, y_val, quantum_neural_network, weights):
    # Reduction of the input dimension
    def preprocess(X):
        return np.tanh(np.mean(X, axis=1)).reshape(-1, 1)

    X_val = X_val.to_numpy()
    X_val_reduced = preprocess(X_val)

    predictions = [
        quantum_neural_network(X_val_reduced[i], weights)
        for i in range(len(X_val_reduced))
    ]
    mse = np.mean((np.array(predictions) - np.array(y_val)) ** 2)
    print(f"Validation MSE: {mse:.4f}")
    return mse


def define_quantum_network_3(n_qubits, n_layers):
    # Assignment of a quantum device
    dev = qml.device("default.qubit", wires=n_qubits)

    # Quantum layer using AmplitudeEmbedding and StronglyEntanglingLayers
    def quantum_layer(weights, inputs):
        # Encoding the input data using AmplitudeEmbedding
        qml.templates.AmplitudeEmbedding(
            inputs, wires=range(n_qubits), normalize=True, pad_with=0.0
        )

        # Quantum layers with StronglyEntanglingLayers
        qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    # Quantum model
    @qml.qnode(dev)
    def quantum_neural_network(inputs, weights):
        quantum_layer(weights, inputs)
        return qml.expval(qml.PauliZ(0))

    return quantum_neural_network


def train_quantum_model_3(
    X_train,
    y_train,
    quantum_neural_network=define_quantum_network_3(4, 2),
    n_qubits=4,
    n_layers=2,
    steps=200,
    stepsize=0.1,
):
    # Cost function
    def cost(weights, X, y):
        predictions = [quantum_neural_network(X[i], weights) for i in range(len(X))]
        return np.mean((np.array(predictions) - np.array(y)) ** 2)

    # Reduction of the input dimension
    def preprocess(X):
        return np.tanh(np.mean(X, axis=1)).reshape(-1, 1)

    # Data preprocessing
    X_train = X_train.to_numpy()
    X_train_reduced = preprocess(X_train)

    # Parameter initialization for StronglyEntanglingLayers
    weights_shape = (
        n_layers,
        n_qubits,
        3,
    )  # Number of parameters in layers (n_layers, n_qubits, 3)
    weights = np.random.randn(*weights_shape)

    # Training the model
    opt = AdamOptimizer(stepsize=stepsize)
    for step in range(steps):
        weights, cost_val = opt.step_and_cost(
            lambda w: cost(w, X_train_reduced, y_train), weights
        )
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}, Cost = {cost_val:.4f}")

    return weights


def validate_quantum_model_3(X_val, y_val, quantum_neural_network, weights):
    # Reduction of the input dimension
    def preprocess(X):
        return np.tanh(np.mean(X, axis=1)).reshape(-1, 1)

    X_val = X_val.to_numpy()
    X_val_reduced = preprocess(X_val)

    predictions = [
        quantum_neural_network(X_val_reduced[i], weights)
        for i in range(len(X_val_reduced))
    ]
    mse = np.mean((np.array(predictions) - np.array(y_val)) ** 2)
    print(f"Validation MSE: {mse:.4f}")
    return mse


def validate_quantum_models(
    X_val,
    y_val,
    weights_1,
    weights_2,
    weights_3,
    quantum_neural_networks=[
        define_quantum_network_1(4),
        define_quantum_network_2(4, 2),
        define_quantum_network_3(4, 2),
    ],
):

    weights_ = [weights_1, weights_2, weights_3]
    results = []

    # Reduction of the input dimension
    def preprocess(X):
        return np.tanh(np.mean(X, axis=1)).reshape(-1, 1)

    X_val = X_val.to_numpy()  # Assuming that X_val is in DataFrame format
    X_val_reduced = preprocess(X_val)

    # Iterating through the models
    for quantum_neural_network, weights in zip(quantum_neural_networks, weights_):
        # Calculating the prediction
        predictions = [
            quantum_neural_network(X_val_reduced[i], weights)
            for i in range(len(X_val_reduced))
        ]

        # If the problem is a classifier, change the predicates to 0 or 1
        predicted_labels = (np.array(predictions) > 0.5).astype(int)

        # Metrics
        mse = np.mean((np.array(predictions) - np.array(y_val)) ** 2)
        accuracy = accuracy_score(y_val, predicted_labels)
        f1 = f1_score(y_val, predicted_labels)
        auc = roc_auc_score(y_val, predictions)
        error_rate = 1 - accuracy

        conf_matrix = confusion_matrix(y_val, predicted_labels)
        fpr, tpr, _ = roc_curve(y_val, predictions)

        # Generating a report for a particular model
        model_report = {
            "Model": quantum_neural_network.__name__,  # Assuming that the model function is named
            "MSE": mse,
            "Accuracy": accuracy,
            "F1-score": f1,
            "AUC": auc,
            "Error Rate": error_rate,
            "Confusion Matrix": conf_matrix,
            # "ROC Curve": (fpr, tpr),
        }

        # Adding the report to the list of results
        results.append(model_report)

        # Printing out the results
        print(f"Model: {quantum_neural_network.__name__}")
        print(f"Validation MSE: {mse:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation F1-score: {f1:.4f}")
        print(f"Validation AUC: {auc:.4f}")
        print(f"Validation Error Rate: {error_rate:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        # print(f"ROC Curve (FPR, TPR): {list(zip(fpr, tpr))}")
        print("\n")

    return results
