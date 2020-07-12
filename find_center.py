import numpy as np
import matplotlib
matplotlib.use('agg') # need this for openlab
import matplotlib.pyplot as plt
import subprocess
from galaxy import Star

class CurveFitError(Exception):
    def __init__(self, message):
        self.message = message


def estimate_center(img, star, percent_img_to_explore = 0.1, visualize = False, savename = None):
    """
    img: 2D array to fit the 3D surface to
    star: Star object
    percent_of_img_to_explore: since the sextractor points are not very accurate a search is done around the point
        to try and find the actual maximum point.  It will explore in a square 2 * img.height * percent wide and high.
    visualize: if True then a plot showing the original image as a surface, a plot showing the fitted surface,
        and contour plot are shown
    savename: if the savedir is specified then the plot will be saved with this name
    """
    point = (int(star.x), int(star.y))
    h, w = img.shape
    img_dist = int(h * percent_img_to_explore)

    # calculate the x-y points for the zoomed up image
    y_min = max(0, point[1] - img_dist)
    y_max = min(h, point[1] + img_dist)
    x_min = max(0, point[0] - img_dist)
    x_max = min(w, point[0] + img_dist)
    
    sub_img = img[y_min : y_max, x_min : x_max]

    # brightness pixel of the zoomed up image (roughly the center of the star)
    max_pt = np.unravel_index(np.argmax(sub_img), sub_img.shape)
    max_pt = (max_pt[1] + x_min, max_pt[0] + y_min)

    if np.sqrt((point[0] - max_pt[0]) ** 2 + (point[1] - max_pt[1]) ** 2) > len(img) * 0.01:
        raise CurveFitError('Distance between sextractor point and max point is too big.')

    # zoom up on the image so that the brightest pixel is in the center (should give a better fit)
    s_size = 5
    
    ys_min = max(0, max_pt[1] - s_size)
    ys_max = min(h, max_pt[1] + s_size)
    xs_min = max(0, max_pt[0] - s_size)
    xs_max = min(w, max_pt[0] + s_size)
    
    max_sub_img = img[ys_min : ys_max, xs_min : xs_max]

    # create a set of (x, y) coordinate points for alglib
    x = '['
    for i in range(max_sub_img.shape[1]):
        for j in range(max_sub_img.shape[0]):
            x += '[{}, {}], '.format(str(j), str(i))
    x = x[:-2] + ']'
    
    # convert the image into a string alglib can use
    y = '['
    for value in max_sub_img.ravel():
        y += str(value) + ', '
    y = y[:-2] + ']'
    
    # input paramaters (xShift, yShift, xAlpha, yAlpha, Amplitude)
    params = '[5, 5, 1, 1, 100]' #, -1.5, 5, 5, 1, 3, -4, -1, 1.5, 1.5, 6, 4, 1, 1.5, 1.5, 6.5]'
    
    # run and wait for runFit to return the point
    proc = subprocess.Popen(['./CurveFit/runFit', x, y, params], stdout = subprocess.PIPE)
    out, err = proc.communicate()
    res = proc.wait()
    
    if res != 0:
        raise CurveFitError(err)
  
    output = np.array(out.rstrip().split(' ')).astype(float)
    # shift the point back to the original coordinates
    shifted_fit_max_point = (output[0] + xs_min, output[1] + ys_min)

    if visualize:
        plt.figure()
        plt.imshow(max_sub_img, cmap = 'gray')
        plt.scatter(point[0] - xs_min, point[1] - ys_min, s = 2, color = 'blue')
        plt.scatter(real_max_point[0], real_max_point[1], s = 2, color = 'red')
        plt.savefig(savename + '.png')

    return Star(shifted_fit_max_point[0], shifted_fit_max_point[1], x_spread = output[2], y_spread = output[3])
