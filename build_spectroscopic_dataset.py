import os
import numpy as np
import hydra
import logging
from omegaconf import DictConfig


def extract_spectra(src_dir, dst_dir, species):
    for i, c in enumerate(species):

        logging.info(f"Loading data for class {c}")
        # create class in target folder
        os.makedirs(os.path.join(dst_dir, c), exist_ok=True)

        for f in os.listdir(os.path.join(src_dir, c)):
            spectrum = np.load(os.path.join(src_dir, c, f, "HSSeed.npy"))
            # take central part (5x5) of the hyperspectral cube (128x128x300)
            spectrum = spectrum[61:66, 61:66, :]
            # take the average of the 25 pixels
            spectrum = np.mean(spectrum, axis=(0, 1))
            # apply min-max normalization
            spectrum = (spectrum - np.min(spectrum)) / (
                np.max(spectrum) - np.min(spectrum)
            )

            min_vals = np.min(spectrum, keepdims=True, axis=-1)
            max_vals = np.max(spectrum, keepdims=True, axis=-1)

            # Ensure that the range is not zero to avoid division by zero
            range_nonzero = np.where(min_vals != max_vals, max_vals - min_vals, 1.0)

            # Normalize each pixel by subtracting the minimum and dividing by the range
            spectrum_output = (spectrum - min_vals) / range_nonzero

            # save spectrum to target folder
            np.save(os.path.join(dst_dir, c, f), spectrum_output)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logging.info("Starting building spectroscopic dataset")

    species = list(cfg.species.keys())

    # create target folder
    os.makedirs(cfg.spectroscopic_dataset_builder.dst_dir, exist_ok=True)

    extract_spectra(
        cfg.spectroscopic_dataset_builder.src_dir,
        cfg.spectroscopic_dataset_builder.dst_dir,
        species,
    )

    logging.info("Finished building spectroscopic dataset")


if __name__ == "__main__":
    main()
