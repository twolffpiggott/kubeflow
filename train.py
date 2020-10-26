import argparse
import json
import logging
import torch
from pathlib import Path
from spotlight.sequence.implicit import ImplicitSequenceModel
from utils import configure_logger, read_sequence_interactions


def train_implicit_sequence_model(
    train_interactions_config: str, n_iter: int = 3, base_dir=Path("/app")
):
    """
    Train an implicit sequence model for <n_iter> iterations.

    :param train_interactions_config: path to json manifest from which to
        initialise train sequence interactions dataset
    :param n_iter: number of iterations to train for
    :param base_dir: dir in which to save trained model
    """
    logger.info(
        "Instantiating training dataset from config: %s", train_interactions_config
    )
    with open(train_interactions_config, "r") as f:
        train_config_dict = json.load(f)
    train_dataset = read_sequence_interactions(train_config_dict)

    model = ImplicitSequenceModel(n_iter=n_iter, representation="cnn", loss="bpr")
    logger.info("Instantiated model %s", model)

    logger.info("Training for %d iterations", n_iter)
    model.fit(train_dataset)

    # TODO give the trained model a unique identifier
    model_path = base_dir / "implicit_sequence_model.pt"
    torch.save(model, model_path.as_posix())


if __name__ == "__main__":
    logger = logging.getLogger()
    configure_logger(logger)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_interactions_config")
    args = parser.parse_args()
    train_implicit_sequence_model(args.train_interactions_config)
