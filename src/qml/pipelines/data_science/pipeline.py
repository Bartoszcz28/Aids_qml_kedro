from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    split_data,
    train_neural_network_1,
    train_neural_network_2,
    train_neural_network_3,
    generate_report,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["model_input_table", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_node",
            ),
            node(
                func=train_neural_network_1,
                inputs=["X_train", "y_train"],
                outputs="neural_network_model_1",
                name="train_model_aids_node_1",
            ),
            node(
                func=train_neural_network_2,
                inputs=["X_train", "y_train"],
                outputs="neural_network_model_2",
                name="train_model_aids_node_2",
            ),
            node(
                func=train_neural_network_3,
                inputs=["X_train", "y_train"],
                outputs="neural_network_model_3",
                name="train_model_aids_node_3",
            ),
            node(
                func=generate_report,
                inputs=[
                    "neural_network_model_1",
                    "neural_network_model_2",
                    "neural_network_model_3",
                    "X_test",
                    "y_test",
                ],
                outputs=None,
                name="generate_report_node",
            ),
        ]
    )
