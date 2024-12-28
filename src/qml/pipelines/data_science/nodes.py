import logging
import tempfile
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Spacer
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    r2_score,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO


def split_data(data: pd.DataFrame, parameters: dict) -> tuple:

    X = data.drop(columns=["infected"])
    y = data["infected"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train.to_frame(), y_test.to_frame()


def train_neural_network_1(X_train: pd.DataFrame, y_train: pd.Series) -> Model:

    # Creating a sequential model
    net = Sequential()
    net.add(Dense(3, input_dim=X_train.shape[1], activation="relu"))
    net.add(Dense(1, activation="sigmoid"))
    net.compile(loss="binary_crossentropy", optimizer="adam")
    net.fit(X_train, y_train, epochs=50, batch_size=50, validation_split=0.2)
    return net


def train_neural_network_2(X_train: pd.DataFrame, y_train: pd.Series) -> Model:
    net = Sequential()
    net.add(
        Dense(
            2,
            input_dim=X_train.shape[1],
            activation="relu",
            kernel_regularizer=l2(0.01),
        )
    )
    net.add(Dropout(0.2))
    net.add(Dense(1, activation="sigmoid"))

    # Compilation of the model
    net.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Early stop
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    # Teaching the model with validation
    net.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=10,
        validation_split=0.2,
        callbacks=[early_stopping],
    )
    return net


def train_neural_network_3(X_train: pd.DataFrame, y_train: pd.Series) -> Model:
    net = Sequential()
    net.add(
        Dense(
            64,
            input_dim=X_train.shape[1],
            activation="relu",
            kernel_regularizer=l2(0.001),
        )
    )
    net.add(Dropout(0.2))
    net.add(Dense(32, activation="relu", kernel_regularizer=l2(0.001)))
    net.add(Dropout(0.2))
    net.add(Dense(16, activation="relu", kernel_regularizer=l2(0.001)))
    net.add(Dropout(0.2))
    net.add(Dense(1, activation="sigmoid"))

    # Compilation of the model
    net.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Early stopping with more flexible settings
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    # Teaching the model with validation
    net.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
    )
    return net


def evaluate_loaded_model(model, X_test, y_test, model_name):
    # Calculating forecasts
    predictions = model.predict(X_test)
    predicted_labels = (predictions > 0.5).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, predictions)
    gini = 2 * auc - 1
    accuracy = accuracy_score(y_test, predicted_labels)
    f1 = f1_score(y_test, predicted_labels)
    error_rate = 1 - accuracy

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, predicted_labels)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, predictions)

    return {
        "Model": model_name,
        "AUC": auc,
        "Gini": gini,
        "Accuracy": accuracy,
        "F1-score": f1,
        "Error Rate": error_rate,
        "Confusion Matrix": conf_matrix,
        "ROC Curve": (fpr, tpr),
    }


def generate_report(model_1, model_2, model_3, X_test, y_test):
    summary = []

    # Create a canvas object for the PDF
    file_path = "data/08_reporting/model_report.pdf"
    c = SimpleDocTemplate(file_path, pagesize=letter)
    elements = []

    # Report title
    title = "Raport z oceny modeli"
    title_style = TableStyle(
        [
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 16),
        ]
    )
    title_table = Table([[title]], style=title_style)
    elements.append(title_table)

    # We add space after the title
    elements.append(Spacer(1, 20))  # Adding more space

    # Models and their results
    models = [(model_1, "NN1"), (model_2, "NN2"), (model_3, "NN3")]

    # Generating charts and results
    plt.figure(figsize=(6, 4))
    for model, model_name in models:
        results = evaluate_loaded_model(model, X_test, y_test, model_name)

        # Adding results to the summary
        summary.append(
            {
                "Model": results["Model"],
                "AUC": results["AUC"],
                "Gini": results["Gini"],
                "Accuracy": results["Accuracy"],
                "F1-score": results["F1-score"],
                "Error Rate": results["Error Rate"],
            }
        )

        # ROC chart
        fpr, tpr = results["ROC Curve"]
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {results['AUC']:.2f})")

        # Adding a title to a chart
        plt.title("Krzywa ROC dla modeli")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid()

        # Save the chart to an image file
        plot_filename = f"data/08_reporting/{model_name}_roc_curve.png"
        plt.savefig(plot_filename, format="png")
        plt.close()

        # Adding a chart to a PDF
        elements.append(
            Image(plot_filename, width=400, height=300)
        )  # Determine the size of the image

        # Insert model results into PDF
        model_results = [
            ["Model", "AUC", "Gini", "Accuracy", "F1-score", "Error Rate"],
            [
                model_name,
                f"{results['AUC']:.2f}",
                f"{results['Gini']:.2f}",
                f"{results['Accuracy']:.2f}",
                f"{results['F1-score']:.2f}",
                f"{results['Error Rate']:.2f}",
            ],
        ]
        results_table = Table(
            model_results,
            style=[
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ],
        )
        elements.append(results_table)

        # Confusion matrix
        conf_matrix = results["Confusion Matrix"]
        conf_matrix_data = [
            ["True Positives", "False Positives", "False Negatives", "True Negatives"],
            [
                conf_matrix[1, 1],
                conf_matrix[0, 1],
                conf_matrix[1, 0],
                conf_matrix[0, 0],
            ],
        ]
        conf_matrix_table = Table(
            conf_matrix_data,
            style=[
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ],
        )
        elements.append(conf_matrix_table)

        # Adding space between elements (e.g., before the next chart).
        elements.append(Spacer(1, 20))  # Increased space

    # Saving the PDF
    c.build(elements)
