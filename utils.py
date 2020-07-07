import cv2 
import numpy as np
import typing

def crop_max_square_from_img(image: np.array) -> np.array:
    rows,cols,n = image.shape
    d = min(rows, cols)
    return image[int((rows-d)/2):int((rows+d)/2),int((cols-d)/2):int((cols+d)/2)]

class SurfaceDomain:
    def __init__(self, x_low: float, y_low: float, x_high: float, y_high: float):
        self.x_low = x_low
        self.x_high = x_high
        self.y_low = y_low
        self.y_high = y_high

class ColorSurfaceFunctionBase:
    def __init__(self, domain: SurfaceDomain):
        self.domain = domain

    @staticmethod
    def toCoords(X: np.array) -> typing.Tuple[float, float]:
        return X[0, 0], X[1, 0]

    def __call__(self, X: np.array) -> typing.Tuple[int, int, int]:
        x, y = self.toCoords(X)
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
    def __init__(self, output_width: int, output_height: int):
        self.canvas = np.zeros((output_height, output_width,3), dtype=np.uint8)
        self.output_width = output_width
        self.output_height = output_height

    def plot_affine(self, 
             f: ColorSurfaceFunctionBase, *, 
             A: np.array = np.array([[1, 0], [0, 1]]),
             b: np.array = np.array([[0], [0]]),
             domain: typing.Optional[SurfaceDomain] = None):
        if domain == None: domain = f.domain
        for row in range(self.output_height):
            y = domain.y_high + row/(self.output_height - 1) * (domain.y_low - domain.y_high)
            for col in range(self.output_width):
                x = domain.x_low + col/(self.output_width - 1) * (domain.x_high - domain.x_low)
                X = np.array([[x],[y]])
                AX_b = np.dot(A, X) + b
                self.canvas[row, col, :] = f(AX_b)

    def save(self, dst_path: str):
        cv2.imwrite(dst_path, self.canvas)

