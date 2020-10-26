import json
import logging
import numpy as np
from logging import Logger
from pathlib import Path
from typing import Dict
from spotlight.interactions import SequenceInteractions


def write_sequence_interactions(
    sequence_interactions: SequenceInteractions, identifier: str, base_dir=Path("/app")
) -> str:
    """
    Minimal util to persist SequenceInteractions to disk and save a simple manifest.

    :param sequence_interactions: SequenceInteractions object
    :param identifier: str modifier for asset paths
    :param base_dir: base dir in which to persist manifest and assets
    :return: path to json manifest of kwargs and asset paths
    """
    interactions_config = dict()
    sequences_path = (base_dir / f"{identifier}_sequences.npy").as_posix()
    np.save(sequences_path, sequence_interactions.sequences)
    interactions_config["sequences_path"] = sequences_path

    if sequence_interactions.user_ids is not None:
        user_ids_path = (base_dir / f"{identifier}_user_ids.npy").as_posix()
        np.save(user_ids_path, sequence_interactions.user_ids)
        interactions_config["user_ids_path"] = user_ids_path

    if sequence_interactions.num_items is not None:
        interactions_config["num_items"] = sequence_interactions.num_items

    config_output_path = base_dir / f"{identifier}_sequence_interactions_config.json"
    with config_output_path.open("w") as f:
        json.dump(interactions_config, f)

    return config_output_path


def read_sequence_interactions(
    interactions_config: Dict[str, str]
) -> SequenceInteractions:
    """
    Minimal util to read SequenceInteractions from a simple manifest.

    :param interactions_config: dict giving kwargs and paths to saved assets
    :return: SequenceInteractions object
    """
    sequences = np.load(interactions_config["sequences_path"])
    user_ids_path = interactions_config.get("user_ids_path")
    if user_ids_path is not None:
        user_ids = np.load(user_ids_path)
    else:
        user_ids = None
    num_items = interactions_config.get("num_items")

    return SequenceInteractions(
        sequences=sequences, user_ids=user_ids, num_items=num_items
    )


def configure_logger(
    logger: Logger,
    log_level: str = "INFO",
):
    """
    Minimal configuration for logging to stream.

    :param logger: root logger object
    :param log_level: level at which to log
    """
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)7s - " "%(module)15s - %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
