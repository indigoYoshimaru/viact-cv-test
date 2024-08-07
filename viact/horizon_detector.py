from abc import ABC
from pydantic import BaseModel
import cv2


class HorizonDetector(ABC):

    def level_horizon(self): ...

    def crop_black_background(self): ...

    def post_process(self): ...


class HorizonDetectorOpenCV(HorizonDetector, BaseModel):
    with_color_segment: bool
    houghline_thres: int

    class Config:
        arbitrary_types_allowed = True

    def segment_by_color(self, image: cv2.Mat): ...

    def __call__(self, image: cv2.Mat, image_shape: tuple): ...
