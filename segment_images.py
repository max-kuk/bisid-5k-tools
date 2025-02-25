import os

os.environ["KERAS_BACKEND"] = "jax"
import zipfile
import os
import tempfile
from keras_cv.models import SegmentAnythingModel
from sam_keras import SAMAutomaticMaskGenerator, SAMPredictor
import spectral
import numpy as np
import matplotlib.pyplot as plt
import logging
import hydra
from omegaconf import DictConfig


import cv2
import spectral.io.envi as envi


def read_bil_from_zip(zip_path):
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Extract the .zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # Find the .hdr and .bil files in the extracted content
        hdr_file = None
        bil_file = None
        for file in os.listdir(temp_dir):
            if file.endswith(".hdr"):
                hdr_file = os.path.join(temp_dir, file)
            elif file.endswith(".bil"):
                bil_file = os.path.join(temp_dir, file)

        if not hdr_file or not bil_file:
            raise FileNotFoundError(
                "Could not find .hdr and/or .bil files in the zip archive."
            )

        # Load the image using spectral library
        img = envi.open(hdr_file, bil_file)
        hs_image = img.load()
        logging.info(f"Loaded image with shape: {hs_image.shape}")

        fake_rgb_image = spectral.get_rgb(
            hs_image, bands=[126, 83, 39], stretch=(0.02), stretch_all=True
        )

        fake_rgb_image = (fake_rgb_image * 255).astype(np.uint8)

        # Optionally, do something with the image
        return hs_image, fake_rgb_image

    finally:
        # Clean up the temporary directory
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(temp_dir)


def resize_image(image, target_size=192):
    width, height = image.shape[1], image.shape[0]
    ratio = target_size / max(width, height)
    return cv2.resize(image, (int(ratio * width), int(ratio * height))), ratio


def cut_out_bbox(img, bbox):
    return img[
        int(bbox[1]) : int(bbox[1] + bbox[3]), int(bbox[0]) : int(bbox[0] + bbox[2])
    ]


# extrapolate selected bbox to original image
def extrapolate_bbox(bbox, ratio):
    return [
        int(bbox[0] / ratio),
        int(bbox[1] / ratio),
        int(bbox[2] / ratio),
        int(bbox[3] / ratio),
    ]


def show_bboxes(img, anns, area_min=100, area_max=10000, type="rgb"):
    if len(anns) == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)

    lower_blue = np.array([110, 100, 100])
    upper_blue = np.array([130, 255, 255])

    suitable_anns = []
    for i in range(len(anns)):
        bbox = anns[i]["bbox"]
        area = anns[i]["area"]
        avg_color = np.median(img[anns[i]["segmentation"]], axis=0)
        avg_color_hsv = cv2.cvtColor(np.uint8([[avg_color]]), cv2.COLOR_RGB2HSV)[0][0]

        logging.info(f"Object {i} Area {area} Type {type} Avg color {avg_color_hsv}")

        if area > area_min and area < area_max:

            logging.info("Area is suitable")

            # Get HSV range values
            hue_min, sat_min, val_min = lower_blue
            hue_max, sat_max, val_max = upper_blue

            # Check if avg_color_hsv falls outside the defined range
            if (
                avg_color_hsv[0] < hue_min
                or avg_color_hsv[0] > hue_max
                or avg_color_hsv[1] < sat_min
                or avg_color_hsv[1] > sat_max
                or avg_color_hsv[2] < val_min
                or avg_color_hsv[2] > val_max
            ):

                suitable_anns.append(anns[i])

    if len(suitable_anns) == 0:
        raise ValueError("No suitable annotations found.")
    elif len(suitable_anns) > 1:
        logging.info(
            "More than one suitable annotation found. Selecting the one closest to the center."
        )

        # select object which is in the middle of the image
        img_center = img.shape[1] / 2
        suitable_anns.sort(
            key=lambda x: abs(x["bbox"][0] + x["bbox"][2] / 2 - img_center)
        )

    final_ann = suitable_anns[0]

    return final_ann["bbox"]


# expand the bouding box so the image have size 256x256
def expand_bbox(bbox, img_shape, size=192):
    target_size_w = max(img_shape[0], size)
    target_size_h = max(img_shape[1], size)
    bbox_w = bbox[2]
    bbox_h = bbox[3]
    bbox_x = bbox[0]
    bbox_y = bbox[1]
    if bbox_w < target_size_w:
        bbox_x = max(0, bbox_x - (target_size_w - bbox_w) // 2)
        bbox_w = target_size_w
    if bbox_h < target_size_h:
        bbox_y = max(0, bbox_y - (target_size_h - bbox_h) // 2)
        bbox_h = target_size_h
    return [bbox_x, bbox_y, bbox_w, bbox_h]


def load_model():
    model = SegmentAnythingModel.from_preset("sam_base_sa1b")
    predictor = SAMPredictor(model)

    return SAMAutomaticMaskGenerator(predictor, min_mask_region_area=100)


def find_white_rectangle(image, threshold=200):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply dilation
    kernel = np.ones((30, 30), np.uint8)
    dilated = cv2.dilate(gray, kernel, iterations=1)
    # Apply a threshold to the image
    _, thresh = cv2.threshold(dilated, threshold, 255, cv2.THRESH_BINARY)

    # Find contours in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # we are looking for the biggest rectangle with white color inside
    max_area = 0
    best_rect = None
    for contour in contours:
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > max_area:
            max_area = area
            best_rect = (x, y, w, h)

    if best_rect is None:
        return None
    x, y, w, h = best_rect

    # we are looking for the region inside the rectangle so crop 12.5% of the rectangle from each side
    x = int(x + w * 0.125)
    y = int(y + h * 0.125)
    w = int(w * 0.75)
    h = int(h * 0.75)

    return x, y, w, h


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.info("Start processing the images")

    mask_generator: SAMAutomaticMaskGenerator[SAMPredictor] = load_model()
    # create folder where to save the segmented images
    if not os.path.exists(cfg.segmenter.dst_dir):
        os.makedirs(cfg.segmenter.dst_dir)

    species = list(cfg.species.keys())

    for seed in species:
        # create folder for each seed
        seed_dir = os.path.join(cfg.segmenter.dst_dir, seed)
        if not os.path.exists(seed_dir):
            os.makedirs(seed_dir)

        logging.info(f"Processing species {seed}")

        start_idx = 0
        # find all folders in the seed folder
        seed_ids = [
            int(folder)
            for folder in os.listdir(f"{cfg.segmenter.src_dir}/{seed}")
            if os.path.isdir(f"{cfg.segmenter.src_dir}/{seed}/{folder}")
        ]

        seed_ids = sorted(seed_ids)

        if start_idx is not None:
            # remove images which have index less than start_idx
            seed_ids = [id for id in seed_ids if id >= start_idx]

        for id in seed_ids:
            logging.info(f"Processing image {id}")
            seed_dir = os.path.join(cfg.segmenter.dst_dir, seed, str(id))
            if not os.path.exists(seed_dir):
                os.makedirs(seed_dir)

            rgb_img_path = os.path.join(cfg.segmenter.src_dir, seed, id, "RGB_Raw.jpg")
            hs_img_path = os.path.join(cfg.segmenter.src_dir, seed, id, "HS_Raw.zip")

            try:
                rgb_image = cv2.imread(rgb_img_path)
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

                rgb_white_rect = find_white_rectangle(rgb_image, threshold=100)
                # crop the image
                rgb_image = rgb_image[
                    rgb_white_rect[1] : rgb_white_rect[1] + rgb_white_rect[3],
                    rgb_white_rect[0] : rgb_white_rect[0] + rgb_white_rect[2],
                ]

                resized_rgb_image, rgb_ratio = resize_image(rgb_image, 512)

                hs_image, fake_rgb_image = read_bil_from_zip(hs_img_path)
                fake_rgb_white_rect = find_white_rectangle(
                    fake_rgb_image, threshold=150
                )
                # crop the image
                fake_rgb_image = fake_rgb_image[
                    fake_rgb_white_rect[1] : fake_rgb_white_rect[1]
                    + fake_rgb_white_rect[3],
                    fake_rgb_white_rect[0] : fake_rgb_white_rect[0]
                    + fake_rgb_white_rect[2],
                ]
                # crop hs_image
                hs_image = hs_image[
                    fake_rgb_white_rect[1] : fake_rgb_white_rect[1]
                    + fake_rgb_white_rect[3],
                    fake_rgb_white_rect[0] : fake_rgb_white_rect[0]
                    + fake_rgb_white_rect[2],
                ]
                resized_fake_rgb_image, hs_ratio = resize_image(fake_rgb_image, 512)

            except Exception as e:
                logging.error(f"Error processing image {seed}, {id}: {e}")

            try:
                # predict the masks
                rgb_masks = mask_generator.generate(resized_rgb_image, verbose=0)
                hs_masks = mask_generator.generate(resized_fake_rgb_image, verbose=0)
            except Exception as e:
                logging.error(f"Error predicting masks for image {seed}, {id}: {e}")

            # sort by area and show the area with the smallest area
            rgb_masks = sorted(rgb_masks, key=(lambda x: x["area"]))
            hs_masks = sorted(hs_masks, key=(lambda x: x["area"]))

            for i in range(len(hs_masks)):
                logging.info("Object", i, "Area", rgb_masks[i]["area"], "Type", "rgb")

            selected_rgb_bbox = show_bboxes(
                resized_rgb_image, rgb_masks, area_min=2000, area_max=10000, type="rgb"
            )
            selected_hs_bbox = show_bboxes(
                resized_fake_rgb_image,
                hs_masks,
                area_min=1000,
                area_max=12000,
                type="hs",
            )

            try:
                original_rgb_selected_bbox = extrapolate_bbox(
                    selected_rgb_bbox, rgb_ratio
                )
            except Exception as e:
                logging.error(
                    f"Error extrapolating RGB bbox for image {seed}, {id}: {e}"
                )

            rgb_img1 = cut_out_bbox(rgb_image, original_rgb_selected_bbox)
            increased_rgb_bbox = expand_bbox(original_rgb_selected_bbox, rgb_img1.shape)
            final_rgb_img = cut_out_bbox(rgb_image, increased_rgb_bbox)
            # resize the image to 192x192
            final_rgb_img = cv2.resize(final_rgb_img, (192, 192))

            try:
                original_hs_selected_bbox = extrapolate_bbox(selected_hs_bbox, hs_ratio)
            except Exception as e:
                logging.error(
                    f"Error extrapolating HS bbox for image {seed}, {id}: {e}"
                )

            hs_img1 = cut_out_bbox(fake_rgb_image, original_hs_selected_bbox)

            increased_hs_bbox = expand_bbox(
                original_hs_selected_bbox, hs_img1.shape, size=128
            )
            final_fake_rgb_img = cut_out_bbox(fake_rgb_image, increased_hs_bbox)

            final_hs_img = cut_out_bbox(hs_image, increased_hs_bbox)

            # resize the image to 128x128
            final_hs_img = cv2.resize(final_hs_img, (128, 128))

            # convert to BGR
            final_rgb_img = cv2.cvtColor(final_rgb_img, cv2.COLOR_RGB2BGR)
            final_fake_rgb_img = cv2.cvtColor(final_fake_rgb_img, cv2.COLOR_RGB2BGR)

            # save the images
            cv2.imwrite(os.path.join(seed_dir, "RGBSeed.png"), final_rgb_img)
            cv2.imwrite(os.path.join(seed_dir, "HSSeed_fake.png"), final_fake_rgb_img)

            # save the HS image as npy
            np.save(os.path.join(seed_dir, "HSSeed.npy"), final_hs_img)

            # free up memory
            del rgb_image
            del resized_rgb_image
            del fake_rgb_image
            del resized_fake_rgb_image
            del hs_image
            del rgb_masks
            del hs_masks
            del selected_rgb_bbox
            del selected_hs_bbox
            del original_rgb_selected_bbox
            del rgb_img1
            del increased_rgb_bbox
            del final_rgb_img
            del original_hs_selected_bbox
            del hs_img1
            del increased_hs_bbox
            del final_fake_rgb_img
            del final_hs_img

    logging.info("Finished processing the images")


if __name__ == "__main__":

    main()
