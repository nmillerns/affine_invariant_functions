import cv2
import sys
import numpy as np
from utils import OpenBoundary as Open
from utils import *
from math import sqrt, log, pi, sin, cos, floor

def sawtooth(x: float) -> float:
    return x - int(x)

class ChequePattern(ColorSurfaceFunctionBase):
    def __init__(self):
        super().__init__(SurfaceDomain(0, 0, 100, 100))

    def __call__(self, x: float, y: float):
        u, v = sawtooth(x), sawtooth(y)
        if u < 0.5 and v < 0.5 or u > 0.5 and v > 0.5:
            return to_color(1)
        else:
            return to_color(0)

def main(args: typing.List[str]) -> int:
    frame = 0
    plotter = ColorSurfacePlotter(100, 100)
    cheques = ChequePattern()
    N = 6
    for i in range(N):
        A, b = A_b_from_params(rotation_angle=0, scale=1, b=np.array([[1],[1]]), b_scale=i/N*0.5)
        plotter.plot_affine(cheques, A=A, b=b, window=SurfaceDomain(0, 0, Open(4), Open(4)))
        plotter.save(f"cheque{frame}.png")
        frame += 1
    print("See results in cheque*.png")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
