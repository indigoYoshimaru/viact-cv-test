from typing import Text
from loguru import logger
import cv2


def read_image(image_path: Text):

    try:
        logger.info(f"Reading image from {image_path}")
        image = cv2.imread(image_path)
    except Exception as e:
        logger.error(f"{type(e).__name__}: {e}. Cannot read image from {image_path}")
        raise e
    else:
        logger.success(f"Read image from {image_path} with shape {image.shape}")
        return image, image.shape


def save_image(image: cv2.Mat, image_path: Text): ...


def view_image(image: cv2.Mat, title: Text):
    while True: 
        cv2.imshow(title, image)
        k = cv2.waitKey(33)
        if k ==27: 
            cv2.destroyAllWindows()
            break

