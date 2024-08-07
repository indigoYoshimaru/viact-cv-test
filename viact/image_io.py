from typing import Text
from loguru import logger


def read_image(image_path: Text):
    import cv2

    try:
        logger.info(f"Reading image from {image_path}")
        image = cv2.imread(image_path)
    except Exception as e:
        logger.error(f"{type(e).__name__}: {e}. Cannot read image from {image_path}")
        raise e
    else:
        logger.success(f"Read image from {image_path} with shape {image.shape}")
        return image, image.shape


def save_image(image_path: Text): ...
