import cv2
import sys
import numpy as np
from utils import *
from math import sqrt, log, pi, sin, cos, floor

class TrianglePattern(ColorSurfaceFunctionBase):
    def __init__(self):
        super().__init__(SurfaceDomain(-100, -100, 100, 100))
        self.basis = np.array([[1, cos(pi/3.)], [0, sin(pi/3.)]])
        self.T = np.linalg.inv(self.basis)

    def __call__(self, x: float, y: float):
        u, v = np.dot(self.T, toVec(x, y))
        r = u - floor(u)
        s = v - floor(v)
        B1 = self.basis[:,0]
        B2 = self.basis[:,1]
        P = np.array([0, 0]) if r < 1 - s else B1 + B2
        Q = B1
        R = B2
        C = (P + Q + R)/3
        rxy = B1 * r + B2 * s
        d = sqrt(np.dot(rxy - C, rxy - C))
        intensity = int(255 * (2 - d))
        return (intensity,intensity,intensity)

def main(args: typing.List[str]) -> int:
    frame = 0
    N = 9
    plotter = ColorSurfacePlotter(400, 400)
    triangles = TrianglePattern()
    for i in range(N+1):
        theta = pi / 3 * i / N
        A, b = A_b_from_params(rotation_angle=theta, scale=1)
        plotter.plot_affine(triangles, A=A, b=b, window=SurfaceDomain(-2.5, -2.5, 2.5, 2.5))
        plotter.save(f"tanimation{frame}.png")
        frame += 1

    N = 6
    for i in range(N+1):
        theta = 0
        A, b = A_b_from_params(rotation_angle=theta, scale=1, b=np.array([[-cos(pi/3)],[-sin(pi/3)]]), b_scale=i/N)
        plotter.plot_affine(triangles, A=A, b=b, window=SurfaceDomain(-2.5, -2.5, 2.5, 2.5))
        plotter.save(f"tanimation{frame}.png")
        frame += 1
    print("See resultsin tanimation*.png")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
