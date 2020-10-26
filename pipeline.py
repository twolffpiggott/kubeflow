import kfp
from kfp import dsl


def preprocess_op(base_dir: str = "/app"):
    return dsl.ContainerOp(
        name="Preprocess Sequential",
        image="preprocess_sequential:latest",
        arguments=[],
        file_outputs={
            "train_interactions_config": f"{base_dir}/train_interactions_config.json",
            "test_interactions_config": f"{base_dir}/test_interactions_config.json",
        },
    )


def train_op(train_interactions_config: str, base_dir: str = "/app"):
    return dsl.ContainerOp(
        name="Train ImplicitSequenceModel",
        image="train:latest",
        arguments=["--train_interactions_config", train_interactions_config],
        file_outputs={"model": f"{base_dir}/implicit_sequence_model.pkl"},
    )


@dsl.pipeline(
    name="Implicit sequence model pipeline",
    description="A pipeline that trains a Spotlight ImplicitSequenceModel.",
)
def implicit_sequence_model_pipeline():
    _preprocess_op = preprocess_op()

    _train_op = train_op(
        dsl.InputArgumentPath(_preprocess_op.outputs["train_interactions_config"]),
    ).after(_preprocess_op)
    # TODO model evaluation using _preprocess_op.outputs["test_interactions_config"]


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        implicit_sequence_model_pipeline, "implicit_sequence_model_pipeline.yaml"
    )
