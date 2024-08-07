import cv2
from abc import ABC
from pydantic import BaseModel
from loguru import logger
import numpy as np


class HorizonDetector(ABC):

    def level_horizon(self): ...

    def crop_black_background(self): ...

    def post_process(self): ...


class HorizonDetectorOpenCV(HorizonDetector, BaseModel):
    with_color_segment: bool
    houghline_thres: int

    class Config:
        arbitrary_types_allowed = True

    def horizon_pos_estimate_by_color(
        self,
        image: cv2.Mat,
        num_cluster: int = 3,
        **kwargs,
    ):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # reshape the image to a 2D array of pixels
        pixel_values = image_rgb.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        # define criteria and apply k-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixel_values, num_cluster, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        # apply centers' colors to segmented image
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(image.shape)

        result_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
        return result_image

    def horizone_pos_estimate_by_y(
        self,
        image: cv2.Mat,
        horizon_estimate_y: float,
        **kwargs,
    ):
        blur_img = image.copy()
        height, width, depth = image.shape
        blur_img[int(height * horizon_estimate_y) : height, :] = cv2.blur(
            blur_img[int(height * horizon_estimate_y) : height, :], (50, 50)
        )
        return blur_img

    def __polar_to_points(self, line: np.ndarray, image_shape: tuple):
        height, width, depth = image_shape
        rho, theta = line[0][0], line[0][1]

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x0 = cos_theta * rho
        y0 = sin_theta * rho
        p1 = (int(x0 + width * (-sin_theta)), int(y0 + height * (cos_theta)))
        p2 = (int(x0 - width * (-sin_theta)), int(y0 - height * (cos_theta)))

        return p1, p2

    def __call__(
        self,
        image: cv2.Mat,
        horizon_estimate_y: float = 1 / 4,
        num_cluster: int = 4,
        houghline_thres: int = 200,
        verbose: bool = False,
    ):
        from viact.utils import view_image

        horizon_pos_func_dict = {
            True: self.horizon_pos_estimate_by_color,
            False: self.horizone_pos_estimate_by_y,
        }

        houghline_low_threshold = {
            True: 1,
            False: 5,
        }

        try:
            horizon_pos_est_func = horizon_pos_func_dict[self.with_color_segment]
            logger.info(f"Estimating horizon position using {horizon_pos_est_func}")
            preprocessed_image = horizon_pos_est_func(
                image,
                **dict(horizon_estimate_y=horizon_estimate_y, num_cluster=num_cluster),
            )
            if verbose:
                view_image(preprocessed_image, "Preprocessed image")
        except Exception as e:
            logger.error(f"{type(e).__name__}: {e}. Cannot preprocess image")
            raise e

        try:
            gray_img = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
            edges = cv2.Canny(thresh, 50, 150, apertureSize=5)

            if verbose:
                view_image(edges, "Edges detection")

            lines = cv2.HoughLines(
                edges,
                houghline_low_threshold[self.with_color_segment],
                np.pi / 180,
                houghline_thres,
            )
            p1, p2 = self.__polar_to_points(lines[0], image.shape)
            if verbose:
                line_draw_img = image.copy()
                cv2.line(line_draw_img, p1, p2, (0, 255, 0), 2)
                view_image(line_draw_img, "Horizon detected")

        except Exception as e:
            raise e
