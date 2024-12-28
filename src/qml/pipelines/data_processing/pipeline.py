from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    preprocess_aids,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_aids,
                inputs="aids",
                outputs="model_input_table",
                name="model_input_node",
            ),
        ]
    )
