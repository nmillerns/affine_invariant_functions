import cv2 
import numpy as np
import typing
from math import sqrt, atan2, cos, sin

def toCoords(X: np.array) -> typing.Tuple[float, float]:
    return X[0, 0], X[1, 0]

def toVec(x, y) -> np.array:
    return np.array([[x], [y]])

def crop_max_square_from_img(image: np.array) -> np.array:
    rows,cols,n = image.shape
    d = min(rows, cols)
    return image[int((rows-d)/2):int((rows+d)/2),int((cols-d)/2):int((cols+d)/2)]

def A_b_from_params(*, rotation_angle: float, scale: float, 
                    b: np.array = np.array([[0], [0]]), b_scale: float = 0.) -> typing.Tuple[np.array, np.array]:
        A = np.array([[cos(rotation_angle), -sin(rotation_angle)], 
                      [sin(rotation_angle), cos(rotation_angle)]]) * scale
        return A, b * b_scale

class SurfaceDomain:
    def __init__(self, x_low: float, y_low: float, x_high: float, y_high: float, 
                 x_low_inclusive: bool=True, x_high_inclusive: bool=True, y_low_inclusive: bool=True, y_high_inclusive: bool=True):
        self.x_low = x_low
        self.x_high = x_high
        self.y_low = y_low
        self.y_high = y_high
        self.x_low_inclusive = x_low_inclusive
        self.x_high_inclusive = x_high_inclusive
        self.y_low_inclusive = y_low_inclusive
        self.y_high_inclusive = y_high_inclusive

    @staticmethod
    def out_of_interval(low: float, low_inclusive: bool, val: float, high: float, high_inclusive):
        below = val <= low if not low_inclusive else val < low
        above = val >= high if not high_inclusive else val > high
        return below or above

    def __contains__(self, xy: typing.Tuple[float, float]):
        x, y = xy
        return not self.out_of_interval(self.x_low, self.x_low_inclusive, x, self.x_high, self.x_high_inclusive)\
               and not self.out_of_interval(self.y_low, self.y_low_inclusive, y, self.y_high, self.y_high_inclusive)

class ColorSurfaceFunctionBase:
    def __init__(self, domain: SurfaceDomain):
        self.domain = domain

    def __call__(self, x: float, y: float) -> typing.Tuple[int, int, int]:
        Fxy = (0, 0, 0)
        return Fxy


class ImageSurface(ColorSurfaceFunctionBase):
    def __init__(self, img: np.array):
        super().__init__(SurfaceDomain(-1, -1, 1, 1))
        self.img = img
        H, W, n = img.shape
        self.width = W
        self.height = H

    def __call__(self, X: np.array) -> typing.Tuple[int, int, int]:
        x, y = self.toCoords(X)
        row = int((1 - y)/2 * (self.height - 1))
        col = int((x + 1)/2 * (self.width - 1))
        if row < 0 or col < 0 or row >= self.height or col >= self.width:
            return (0, 0, 0)
        return self.img[row, col, :]


class ColorSurfacePlotter:
    def __init__(self, output_width: int, output_height: int, *, show_axis: bool = False, axis_thickness: int = 1):
        self.canvas = np.zeros((output_height, output_width,3), dtype=np.uint8)
        self.output_width = output_width
        self.output_height = output_height
        self.show_axis = show_axis
        self.axis_thickness = axis_thickness

    def plot_affine(self, 
             f: ColorSurfaceFunctionBase, *, 
             A: np.array = np.array([[1, 0], [0, 1]]),
             b: np.array = np.array([[0], [0]]),
             window: typing.Optional[SurfaceDomain] = None):
        if window is None: window = f.domain
        for row in range(self.output_height):
            y = window.y_high + row/(self.output_height - 1) * (window.y_low - window.y_high)
            for col in range(self.output_width):
                x = window.x_low + col/(self.output_width - 1) * (window.x_high - window.x_low)
                X = np.array([[x],[y]])
                AX_b = toCoords(np.dot(A, X) + b)
                if self.show_axis and (abs(x) < self.axis_thickness or abs(y) < self.axis_thickness):
                    self.canvas[row, col, :] = (255, 255, 255)
                elif (x, y) in window and AX_b in f.domain:
                    self.canvas[row, col, :] = f(*AX_b)
                else:
                    self.canvas[row, col, :] = (0, 0, 0)

    def save(self, dst_path: str):
        cv2.imwrite(dst_path, self.canvas)

