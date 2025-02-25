import glob
import logging
import os
import random
import time
from concurrent import futures
from typing import Union

import cv2
import numpy as np
import tensorflow as tf


class DatasetBuilder:
    """
    Builds a tfrecord dataset from the raw data. (for paper dataset)

    Args:
        src_dir (str, optional): Source directory.
        dst_dir (str, optional): Target directory.
        num_samples (int, optional): Number of samples per tfrecord file. Defaults to 2500.
        logger (str, optional): Name of the log file. Defaults to "log.txt".
        saving_mode (str, optional): Saving mode. Defaults to "per_num_samples". Available options: per_num_samples, per_sample.
    """

    def __init__(
        self,
        src_dir: str,
        dst_dir: str,
        num_samples: int = 2500,
        saving_mode: str = "per_num_samples",
        **kwargs,
    ):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.logger = logging.getLogger(__name__)
        self.num_samples = num_samples
        self.saving_mode = saving_mode
        self.logger.info("Initialising %s...", self.__class__.__name__)

    def __set_num_workers(self, num_workers: int, num_tfrecords: int) -> int:
        if num_workers == 1:
            # Use the number of TFRecord files or the available CPU cores, whichever is smaller
            return min(num_tfrecords, os.cpu_count() or 1)
        else:
            return num_workers

    def __set_num_tfrecords(self, folder_paths) -> int:
        num_tfrecords = len(folder_paths) // self.num_samples
        if len(folder_paths) % self.num_samples:
            num_tfrecords += 1

        return num_tfrecords

    def __set_folder_paths(self, species: Union[str, list]) -> list:
        if isinstance(species, str):
            species = [species]

        folder_paths = []
        self.logger.info("Loading all folders for classes: %s", species)
        for class_ in species:
            self.logger.info(f"Loading folders for class: {class_}")

            #
            folders = glob.glob(os.path.join(self.src_dir, class_, "*"))

            self.logger.info(f"Found {len(folders)} folders for class: {class_}")
            folder_paths.extend(glob.glob(os.path.join(self.src_dir, class_, "*")))

        random.shuffle(folder_paths)

        return folder_paths

    def read_files(self, folder: str) -> tuple[np.ndarray, np.ndarray, str]:
        """Reads the RGB and HS images from the given folder.

        Args:
            folder (str): Path to the folder containing the RGB and HS images.

        Returns:
            Tuple[np.ndarray, np.ndarray, str]: RGB and HS images, and label of the image.
        """

        rgb_img = cv2.imread(os.path.join(folder, "RGBSeed.png"))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        # convert to numpy array
        hs_img = np.load(os.path.join(folder, "HSSeed.npy"))

        # meta_file = json.load(open(os.path.join(folder, "Meta.JSON"), encoding="utf-8"))
        # label = meta_file["species"]
        # label is a first level parent folder where the image is located
        label = os.path.basename(os.path.dirname(folder))
        label = label.lower().strip()

        return (
            hs_img,
            rgb_img,
            label,
        )

    def serialize_example(
        self,
        file_id: str,
        hs_img: np.ndarray,
        rgb_img: np.ndarray,
        label: str,
    ) -> bytes:
        """Creates a tf.train.Example ready to be written to a file.

        Args:
            file_id (str): File id of the image.
            hs_img (np.array): hs image.
            rgb_img (np.array): rgb image.
            label (str): label of the image.

        Returns:
            bytes: Serialized tf.train.Example.
        """
        # Convert and serialize rgb_img
        rgb_img = rgb_img.astype("uint8")
        # rgb_img_raw = rgb_img.tobytes()

        # Convert and serialize hs_img
        hs_img = hs_img.astype("float32")

        hs_img_raw = hs_img.tobytes()

        # Create the Example
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "id": self.__bytes_feature(file_id.encode("utf-8")),
                    "hs_image": self.__bytes_feature(hs_img_raw),
                    "hs_size": self.__int64_feature_list(list(hs_img.shape)),
                    "rgb_image": self.__image_feature(rgb_img),
                    "rgb_size": self.__int64_feature_list(list(rgb_img.shape)),
                    "class": self.__bytes_feature(label.encode("utf-8")),
                }
            )
        )

        return example.SerializeToString()

    def write_record(self, folder_paths, tfrec_num: int) -> None:
        """
        Writes a tfrecord file for the given samples

        Args:
            folder_paths (list): List of folder paths to write.
            tfrec_num (int): Index of the tfrecord file to write.
        """
        samples = folder_paths[
            (tfrec_num * self.num_samples) : ((tfrec_num + 1) * self.num_samples)
        ]

        if self.saving_mode == "per_sample":
            nb_samples = len(folder_paths)
        else:
            nb_samples = len(samples)

        with tf.io.TFRecordWriter(
            os.path.join(
                self.dst_dir, f"dataset_{tfrec_num:02d}-{nb_samples}.tfrecords"
            ),
            options=tf.io.TFRecordOptions(compression_type="GZIP"),
        ) as writer:
            for sample in samples:
                file_id = os.path.basename(sample)

                try:
                    hs_img, rgb_img, label = self.read_files(sample)
                except Exception as e:
                    label = os.path.basename(os.path.dirname(sample))
                    self.logger.warning(
                        "Error reading file: %s/%s. See %s", label, file_id, e
                    )
                    continue
                else:
                    example = self.serialize_example(file_id, hs_img, rgb_img, label)
                    writer.write(example)

    def build(self, species: Union[str, list], num_workers: int = 1) -> None:
        """
        Builds the dataset.

        Args:
            species (Union[str, list], optional): Species to build the dataset for. Defaults to list.
            num_workers (int, optional): Number of workers to use for parallel processing. Defaults to 1 (will parallelize over tfrecord files)
        """
        self.logger.info("Start processing...")

        folder_paths = self.__set_folder_paths(species)

        self.logger.info("Found %s folders", len(folder_paths))

        if self.saving_mode == "per_sample":
            num_tfrecords = len(folder_paths)
        elif self.saving_mode == "per_num_samples":
            num_tfrecords = self.__set_num_tfrecords(folder_paths)
        else:
            raise ValueError(
                f"Invalid saving mode: {self.saving_mode}. Available options: per_num_samples, per_sample."
            )

        num_workers = self.__set_num_workers(num_workers, num_tfrecords)

        start_time = time.time()

        if not os.path.exists(self.dst_dir):
            os.makedirs(self.dst_dir)

        # Create a ThreadPoolExecutor with a suitable number of workers
        with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit tasks to the executor to write each TFRecord file in parallel
            future_to_tfrec_num = {}
            for tfrec_num in range(num_tfrecords):
                future = executor.submit(self.write_record, folder_paths, tfrec_num)
                future_to_tfrec_num[future] = tfrec_num

            # Wait for all tasks to complete
            for future in futures.as_completed(future_to_tfrec_num):
                tfrec_num = future_to_tfrec_num[future]
                try:
                    future.result()
                except Exception as exception:
                    logging.info(
                        "Writing tfrecord %s encountered an error: %s",
                        tfrec_num + 1,
                        exception,
                    )

        end_time = time.time()
        time_taken = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))

        self.logger.info("Processing finished. Time taken: %s", time_taken)

    def __int64_feature(self, value: int) -> tf.train.Feature:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def __int64_feature_list(self, value: list[int]) -> tf.train.Feature:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def __bytes_feature(self, value: bytes) -> tf.train.Feature:
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def __image_feature(self, value: np.ndarray) -> tf.train.Feature:
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.encode_png(value).numpy()])  # type: ignore
        )
