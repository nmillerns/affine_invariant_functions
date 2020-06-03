import cv2
import numpy as np
import sys

def get_square_image(image):
    rows,cols,n = image.shape
    d = min(rows, cols)
    return image[int((rows-d)/2):int((rows+d)/2),int((cols-d)/2):int((cols+d)/2)]


def kernel_read(x, y, k):
    rows,cols,n = k.shape
    return k[int(y*rows),int(x*rows),:]

def populate_large(large, k, A, b):
    W,W,n = large.shape
    for row in range(W):
        y = -1.99 + .01*row
        y = A * y + b
        for col in range(W):
            x = -1.99 + .01*col
            x = A * x + b
            if y > -1 and x > -1:
                large[row,col,:] = kernel_read((x+1)/(2**np.floor(np.log2(x+1))) - 1., (y+1)/(2**np.floor(np.log2(y+1))) - 1., k)
            else:
                large[row, col, :] = 0

            if abs(-1.99 + .01*col) <= .02 or abs(-1.99 + .01*row) <= .02:
                large[row, col, :] = 255



def main(args):
    if len(args) != 1:
        print("Usage: loglog (imgfile.png)")
        return 1
    ifile = args[0]
    sq = get_square_image(cv2.imread(ifile))
    cv2.imshow('kernel', sq)
    P,P,n = sq.shape
    W = 900
    large = np.zeros((W,W,n), dtype=sq.dtype)
    scale = 1.
    translation = 0.
    frame = 0
    while scale > 0.5:
        populate_large(large, sq, scale, translation)
        print(frame, scale, translation)
        cv2.imwrite(f'animation{frame}.png', large)
        scale -= .031250
        frame += 1
    while translation >= -0.5:
        populate_large(large, sq, scale, translation)
        print(frame, scale, translation)
        cv2.imwrite(f'animation{frame}.png', large)
        translation -= 0.0625
        frame += 1

    cv2.imwrite('pattern.png', large)
    return 0



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
