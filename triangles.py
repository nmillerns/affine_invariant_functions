import cv2
import numpy as np
from math import sqrt, log, pi, sin, cos, floor

def plot_image(canvas, A, b):
    B = np.array([[1, cos(pi/3.)], [0, sin(pi/3.)]])
    T = np.linalg.inv(B)
    B1 = B[:,0]
    B2 = B[:,1]
    for i in range(canvas.shape[0]):
        y = (i*1./canvas.shape[0] - 0.5) * -H
        for j in range(canvas.shape[1]):
            x = (j*1./canvas.shape[1] - .5) * W
            xy = np.array([[x],[y]])
            xy = np.dot(A, xy) + b
            uv = np.dot(T, xy)
            u = uv[0,0]
            v = uv[1,0]
            r = u - floor(u)
            s = v - floor(v)
            P = np.array([0, 0]) if r < 1 - s else B1 + B2
            Q = B1
            R = B2
            C = (P + Q + R)/3
            rxy = B1 * r + B2 * s
            d = sqrt(np.dot(rxy - C, rxy - C))
            canvas[i,j,:] = 255 * (2 - d)

W = 5.
H = 5.
frame = 0
N = 9

canvas = np.zeros((400, 400, 3), dtype=np.uint8)
for i in range(N+1):
    theta = pi / 3 * i / N
    A = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    b = np.array([[0],[0]])
    plot_image(canvas, A, b)
    cv2.imwrite(f"tanimation{frame}.png", canvas)
    frame += 1

N = 6
for i in range(N+1):
    theta = 0
    A = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    b = np.array([[-cos(pi/3)],[-sin(pi/3)]]) * i/N
    plot_image(canvas, A, b)
    cv2.imwrite(f"tanimation{frame}.png", canvas)
    frame += 1

