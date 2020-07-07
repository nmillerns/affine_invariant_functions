import cv2
import numpy as np
from utils import *
from math import sqrt, log, pi, sin, cos, floor

class TrianglePattern(ColorSurfaceFunctionBase):
    def __init__(self):
        super().__init__(SurfaceDomain(-2.5, -2.5, 2.5, 2.5))
        self.basis = np.array([[1, cos(pi/3.)], [0, sin(pi/3.)]])
        self.T = np.linalg.inv(self.basis)

    def __call__(self, X: np.array):
        U = np.dot(self.T, X)
        u, v = self.toCoords(U)
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

frame = 0
N = 9
plotter = ColorSurfacePlotter(400, 400)
triangles = TrianglePattern()
for i in range(N+1):
    theta = pi / 3 * i / N
    A = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    b = np.array([[0],[0]])
    plotter.plot_affine(triangles, A=A, b=b)
    plotter.save(f"tanimation{frame}.png")
    frame += 1

N = 6
for i in range(N+1):
    theta = 0
    A = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    b = np.array([[-cos(pi/3)],[-sin(pi/3)]]) * i/N
    plotter.plot_affine(triangles, A=A, b=b)
    plotter.save(f"tanimation{frame}.png")
    frame += 1

