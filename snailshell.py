import cv2
import sys
import numpy as np
from utils import *
from math import sqrt, log, pi, sin, cos, floor, atan2

class SnailShell(ColorSurfaceFunctionBase):
    def __init__(self):
        super().__init__(SurfaceDomain(-8, -8, 8, 8))

    def __call__(self, x: float, y: float):
        r = sqrt(x**2 + y**2)/10.
        angle = (atan2(y, x) + pi)/2/pi
        h = sin(pi * r * 2.**(-angle) / (2**int(log(r)/log(2)-angle)) + pi)
        return to_color((1-h)/2)


def main(args: typing.List[str]) -> int:
    frame = 0
    plotter = ColorSurfacePlotter(2048, 2048)
    plotter.plot_affine(SnailShell())
    plotter.save(f"snailshell_image.png")
    print("See results in snailshell_image.png")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
