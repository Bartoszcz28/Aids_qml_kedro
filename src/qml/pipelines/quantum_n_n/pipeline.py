"""
This is a boilerplate pipeline 'quantum_n_n'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    train_quantum_model_1,
    train_quantum_model_2,
    train_quantum_model_3,
    validate_quantum_models,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_quantum_model_1,
                inputs=["X_train", "y_train"],
                outputs="quantum_network_model_1",
                name="train_quantum_network_node_1",
            ),
            node(
                func=train_quantum_model_2,
                inputs=["X_train", "y_train"],
                outputs="quantum_network_model_2",
                name="train_quantum_network_node_2",
            ),
            node(
                func=train_quantum_model_3,
                inputs=["X_train", "y_train"],
                outputs="quantum_network_model_3",
                name="train_quantum_network_node_3",
            ),
            node(
                func=validate_quantum_models,
                inputs=[
                    "X_test",
                    "y_test",
                    "quantum_network_model_1",
                    "quantum_network_model_2",
                    "quantum_network_model_3",
                ],
                outputs=None,
                name="validate_quantum_models_node",
            ),
        ]
    )
