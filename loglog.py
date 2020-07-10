import cv2
import numpy as np
import sys

from utils import *

def logsawtooth(x: float) -> float:
    return (x+1)/(2**np.floor(np.log2(x+1))) - 1.

class LogLogImagePattern(ImageSurface):
    def __init__(self, img: np.array):
        super().__init__(img)
        self.domain = SurfaceDomain(-1, -1, 7, 7, False, False, False, False)
        
    def __call__(self, x: float, y: float) -> typing.Tuple[int, int, int]:
        u, v = logsawtooth(x), 1. - logsawtooth(y)
        return self.img[int(v*(self.height-1)),int(u*(self.width-1)),:]
        
def main(args: typing.List[str]) -> int:
    if len(args) != 1:
        print("Usage: loglog.py (imgfile.png)")
        return 1
    plotter = ColorSurfacePlotter(900, 900, show_axis = True, axis_thickness = .02)
    f = LogLogImagePattern(crop_max_square_from_img(cv2.imread(args[0])))
    scale = 1.
    translation = 0.
    frame = 0
    while scale > 0.5:
        A, b = A_b_from_params(rotation_angle=0, scale=scale, b=np.array([[1],[1]]), b_scale=translation)
        plotter.plot_affine(f, A=A, b=b, window=SurfaceDomain(-1.99, -1.99, 7, 7))
        print(frame, scale, translation)
        plotter.save(f'animation{frame}.png')
        scale -= .031250
        frame += 1
    while translation >= -0.5:
        A, b = A_b_from_params(rotation_angle=0, scale=scale, b=np.array([[1],[1]]), b_scale=translation)
        plotter.plot_affine(f, A=A, b=b, window=SurfaceDomain(-1.99, -1.99, 7, 7))
        print(frame, scale, translation)
        plotter.save(f'animation{frame}.png')
        translation -= 0.0625
        frame += 1

    print("See results in animation*.png and pattern.png")  
    plotter.save('pattern.png')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
