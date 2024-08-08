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
        if k == 27:
            cv2.destroyAllWindows()
            break


def draw_grid(img, grid_shape, color=(0, 255, 0), thickness=1):
    import numpy as np

    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    # draw vertical lines
    for x in np.linspace(start=dx, stop=w - dx, num=cols - 1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h - dy, num=rows - 1):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img
