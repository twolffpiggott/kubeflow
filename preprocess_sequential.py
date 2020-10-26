import logging
import numpy as np
from spotlight.cross_validation import user_based_train_test_split
from spotlight.datasets.synthetic import generate_sequential
from utils import configure_logger, write_sequence_interactions


def preprocess_sequential():
    """
    Generate sequential data for implicit factorisation and persist
    a train/test split to disk.
    """
    dataset = generate_sequential(
        num_users=100,
        num_items=1000,
        num_interactions=10000,
        concentration_parameter=0.01,
        order=3,
    )
    logger.info("Generated sequential dataset: %s", dataset)

    test_percentage = 0.2
    logger.info(
        "Generating a %d/%d train/test split",
        100 * (1 - test_percentage),
        100 * test_percentage,
    )
    train, test = user_based_train_test_split(
        dataset, test_percentage=test_percentage, random_state=np.random.RandomState(87)
    )

    train = train.to_sequence()
    logger.info("Generated train sequence dataset: %s", train)
    test = test.to_sequence()
    logger.info("Generated test sequence dataset: %s", test)

    train_config_path = write_sequence_interactions(
        train,
        identifier="train",
    )
    logger.info("Wrote train config to %s", train_config_path)
    test_config_path = write_sequence_interactions(
        test,
        identifier="test",
    )
    logger.info("Wrote test config to %s", test_config_path)


if __name__ == "__main__":
    logger = logging.getLogger()
    configure_logger(logger)

    logger.info("Preprocessing sequential data")
    preprocess_sequential()
