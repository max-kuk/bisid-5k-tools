"""
This module contains the function for building a tfrecords dataset.
"""

import hydra
from omegaconf import DictConfig

from dataset_builders.dataset_builder import DatasetBuilder


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Builds a dataset using the specified configuration.

    Args:
        cfg (DictConfig): The configuration to use for building the dataset.
    """

    dataset_builder = DatasetBuilder(
        src_dir=cfg.tfrecords_dataset_builder.src_dir,
        dst_dir=cfg.tfrecords_dataset_builder.dst_dir,
        num_samples=cfg.tfrecords_dataset_builder.num_samples,
        saving_mode=cfg.tfrecords_dataset_builder.saving_mode,
    )

    dataset_builder.build(
        species=cfg.species, num_workers=cfg.tfrecords_dataset_builder.num_workers
    )


if __name__ == "__main__":
    main()
