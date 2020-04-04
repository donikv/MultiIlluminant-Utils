import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import dataset_utils as du
from transformation_utils import adjust_gamma

root = r'.\path\to\root'
root = r'my_img.jpg'

if __name__ == '__main__':

    filename = "./data/dataset_568_shi_gehler/cs/chroma/data/canon_dataset/568_dataset/png/IMG_0293.png"
    im = cv2.imread(filename, cv2.COLOR_BGR2RGB)
    im2 = adjust_gamma(im, gamma=3.5)

    r, g, b = cv2.split(im2)

    img_sum = np.sum(im2, axis = 2) # NOTE: This dtype will be uint32.
                                    #       Each channel can be up to
                                    #       255 (dtype = uint8), but
                                    #       since uint8 can only go up
                                    #       to 255, sum naturally uint32

    # "Normalized" channels
    # NOTE: np.ma is the masked array library. It automatically masks
    #       inf and nan answers from result

    n_r = np.ma.divide(1.*r, g)
    n_b = np.ma.divide(1.*b, g)

    log_rg = np.ma.log( n_r )
    log_bg = np.ma.log( n_b )

    n_r_2 = np.ma.exp(log_rg)
    n_b_2 = np.ma.exp(log_bg)
    r2 = np.ma.multiply(n_r_2, g)
    b2 = np.ma.multiply(n_b_2, g)
    r2, b2 = np.ma.filled(r2, 0).astype(int), np.ma.filled(b2, 0).astype(int)
    img = np.dstack((r2, g, b2))
    du.visualize(im, im2, img)

    plt.scatter(log_rg, log_bg, s = 2)
    plt.xlabel('Log(R/G)')
    plt.ylabel('Log(B/G)')
    plt.title('2D Log Chromaticity')
    plt.show()