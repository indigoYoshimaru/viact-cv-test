import cv2
from abc import ABC
from pydantic import BaseModel
from loguru import logger
from viact.utils import view_image


class ShipDetector(ABC):
    pass


class ShipDetectorOpenCV(ShipDetector, BaseModel):
    verbose: bool

    class Config:
        arbitrary_types_allowed = True

    def detect_contours(self, image: cv2.Mat, y_loc: int, ship_loc_is_upper: bool):

        if ship_loc_is_upper:
            cropped_image = image[:y_loc, :]
        else:
            cropped_image = image[y_loc:, :]
            cropped_image = cv2.GaussianBlur(cropped_image, (5,5), 0)

        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_TOZERO)
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
        kernel_size = (5, 5)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if self.verbose:
            view_image(edges, title="Edges")
            view_image(cropped_image, title="Cropped image by horizon")

        return contours

    def __call__(
        self,
        image: cv2.Mat,
        houghline_thres: int,
        ship_loc_is_upper: bool = True,
    ):
        from viact.horizon_detector import HorizonDetectorOpenCV

        try:
            horizon_detector = HorizonDetectorOpenCV(
                with_color_segment=True,
                houghline_thres=houghline_thres,
                verbose=self.verbose,
            )
            horizon_detect_img = image.copy()
            result_dict = horizon_detector(
                image=horizon_detect_img,
                num_cluster=4,
            )
            start_point = result_dict["start_point"]
            end_point = result_dict["end_point"]
            if ship_loc_is_upper:
                y_loc = min(start_point[1], end_point[1]) 
            else:
                y_loc = max(start_point[1], end_point[1]) + 6
        except Exception as e:
            logger.error(f"{type(e).__name__}: {e}. Cannot detect horizon")
            raise e
        else:
            logger.info(f"Detecting ships with {ship_loc_is_upper=} and yloc {y_loc}")

        try:
            contours = self.detect_contours(
                image,
                y_loc=y_loc,
                ship_loc_is_upper=ship_loc_is_upper,
            )

            height, width, depth = image.shape
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if not ship_loc_is_upper:
                    y += y_loc
                # close to camera point but too small area -> high chances is not ship!
                if (y+w)/height>0.7 and cv2.contourArea(contour)<2000: 
                    continue
                logger.info(f"Bounding box: {x=}; {y=}; {w=}; {h=}")
                cv2.rectangle(image, (x, y), (x + w, y + h + 2), (0, 255, 0), 2)

            if self.verbose:
                view_image(image, "Ship detected")

        except Exception as e:
            logger.error(f"{type(e).__name__}: {e}. Cannot detect ships")
            raise e
        else:
            logger.success(f"Detected ships successfully")
            return image
