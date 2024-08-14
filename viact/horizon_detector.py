import cv2
from abc import ABC
from pydantic import BaseModel
from loguru import logger
import numpy as np
from typing import Dict, Tuple
from viact.utils import view_image, draw_grid


class HorizonDetector(ABC):

    def level_horizon(self, image: cv2.Mat, start_point: Tuple, end_point: Tuple):
        try:
            # calculate the level angle
            level_angle = np.degrees(
                np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
            )
            logger.info(f"Rotating image with angle {level_angle}")

            # create the rotation matrix to rotate around the start point by the level angle
            rotate_matrix = cv2.getRotationMatrix2D(
                start_point, angle=level_angle, scale=1
            )
            rotated_image = cv2.warpAffine(
                src=image, M=rotate_matrix, dsize=(image.shape[1], image.shape[0])
            )

        except Exception as e:
            logger.error(f"{type(e).__name__}: {e}. Cannot level the horizon line.")
            raise e
        else:
            logger.success(f"Image rotated successfully.")
            return rotated_image

    def detect_contours(self, image):
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresholded = cv2.threshold(grayscale, 0, 255, 0)
        contours, hier = cv2.findContours(
            thresholded,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        return contours[0]

    def find_contours_boundary(
        self,
        arg_arr: np.ndarray,
        project_arr: np.ndarray,
        mode=np.argmax,
        reverse: bool = True,
    ):
        if reverse:
            arg = len(arg_arr) - mode(arg_arr[::-1]) - 1
        else:
            arg = mode(arg_arr)

        return project_arr[arg]

    def crop_black_background(self, image: cv2.Mat):
        try:
            contours = self.detect_contours(image)
            x_arr, y_arr = contours.T[0][0], contours.T[1][0]
            y_at_x_max = self.find_contours_boundary(x_arr, y_arr)
            x_at_y_max = self.find_contours_boundary(y_arr, x_arr)
            x_min = self.find_contours_boundary(y_arr, x_arr, mode=np.argmin)

            height = image.shape[0]
            cropped_image = image[
                y_at_x_max + 2 : y_at_x_max + height - 2, x_min:x_at_y_max
            ]

        except Exception as e:
            logger.error(f"{type(e).__name__}: {e}. Cannot crop black background.")
            raise e
        else:
            logger.success(f"Image cropped successfully")
            return cropped_image

    def post_process(self, image, result_dict: Dict):

        theta = result_dict["theta"]
        # the line is already level
        if np.sin(theta) == 1:
            logger.info(f"Sin(theta) = 1. Line already level")
            if self.verbose:
                image_vis = result_dict["image_line_drawn"]
                image_vis = draw_grid(image_vis, color=(0, 0, 255))
                view_image(
                    image_vis,
                    "Leveled and Black boundary cropped with grid",
                )
                result_dict["image_postprocess_grid"] = image_vis

            del result_dict["image_line_drawn"]
            result_dict["image_postprocess"] = image
            return result_dict

        start_point = result_dict["start_point"]
        end_point = result_dict["end_point"]
        line_draw_img = result_dict["image_line_drawn"]

        rotated_image = self.level_horizon(
            image,
            start_point=start_point,
            end_point=end_point,
        )

        if self.verbose:
            vis_rotated_image = self.level_horizon(
                line_draw_img,
                start_point=start_point,
                end_point=end_point,
            )
            view_image(vis_rotated_image, "Rotated image")

        cropped_image = self.crop_black_background(rotated_image)
        if self.verbose:

            vis_cropped_image = self.crop_black_background(vis_rotated_image)
            vis_cropped_image_with_grid = draw_grid(
                vis_cropped_image, color=(0, 0, 255)
            )
            view_image(
                vis_cropped_image_with_grid,
                "Leveled and Black boundary cropped with grid",
            )

            result_dict["image_postprocess_grid"] = vis_cropped_image_with_grid

        del result_dict["image_line_drawn"]
        result_dict["image_postprocess"] = cropped_image
        return result_dict


class HorizonDetectorOpenCV(HorizonDetector, BaseModel):
    with_color_segment: bool
    houghline_thres: int
    verbose: bool 

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

    def __polar_to_points(self, line: np.ndarray, image_shape: Tuple):
        height, width, depth = image_shape
        rho, theta = line[0][0], line[0][1]

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x0 = cos_theta * rho
        y0 = sin_theta * rho
        p1 = (int(x0 + width * (-sin_theta)), int(y0 + height * (cos_theta)))
        p2 = (int(x0 - width * (-sin_theta)), int(y0 - height * (cos_theta)))

        return p1, p2

    def detect_lines(
        self,
        preprocessed_image: cv2.Mat,
        houghline_thres: int,
        houghline_rho: int,
    ):

        gray_img = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_TOZERO)
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

        if self.verbose:
            view_image(edges, "Edges detection")

        lines = cv2.HoughLines(
            edges,
            houghline_rho,
            np.pi / 180,
            houghline_thres,
        )
        logger.info(f"{len(lines)} lines detected")
        return lines

    def __call__(
        self,
        image: cv2.Mat,
        horizon_estimate_y: float = 1 / 4,
        num_cluster: int = 4,
    ):
        from viact.utils import view_image

        horizon_pos_func_dict = {
            True: self.horizon_pos_estimate_by_color,
            False: self.horizone_pos_estimate_by_y,
        }

        houghline_rho_dict = {
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
            if self.verbose:
                view_image(preprocessed_image, "Preprocessed image")
        except Exception as e:
            logger.error(f"{type(e).__name__}: {e}. Cannot preprocess image")
            raise e

        try:
            lines = self.detect_lines(
                preprocessed_image=preprocessed_image,
                houghline_thres=self.houghline_thres,
                houghline_rho=houghline_rho_dict[self.with_color_segment],
            )
            line = lines[0]
            start_point, end_point = self.__polar_to_points(line, image.shape)
            image_line_drawn = image.copy()
            if self.verbose:
                cv2.line(image_line_drawn, start_point, end_point, (0, 255, 0), 2)
                view_image(image_line_drawn, "Horizon detected")

        except Exception as e:
            logger.error(f"{type(e).__name__}: {e}. Cannot detect lines")
            raise e
        else:
            return dict(
                start_point=start_point,
                end_point=end_point,
                image_line_drawn=image_line_drawn,
                rho=line[0][0],
                theta=line[0][1],
            )
