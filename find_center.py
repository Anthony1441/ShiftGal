import numpy as np
from galaxy import Star
from astropy.modeling import functional_models, fitting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import subprocess


class CurveFitError(Exception):
    def __init__(self, message):
        self.message = message


def estimate_center(img, star, outdir = None, color = 'no color', min_peak_value = 1.0, percent_img_to_explore = 0.025):
    """
    img: Full galaxy image
    star: Star object
    percent_of_img_to_explore: since the sextractor points are not very accurate a search is done around the point
        to try and find the actual maximum point.  It will explore in a square 2 * img.height * percent wide and high
    """
    s_size = 4
    if star.x < s_size or star.x > img.shape[1] - s_size or star.y < s_size or star.y > img.shape[0] - s_size:
        raise CurveFitError('Star located at {} is too close to the edge of the image'.format(str(star)))

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
 
    if img[max_pt[1], max_pt[0]] < min_peak_value:
        raise CurveFitError('Star at {} is too dim: {} of {} needed'.format(max_pt, img[max_pt[1], max_pt[0]], min_peak_value))

    dist = np.sqrt((point[0] - max_pt[0]) ** 2 + (point[1] - max_pt[1]) ** 2)
    
    if dist > 10:
        raise CurveFitError('Distance between sextractor point and max point is too big: {} pixels.'.format(dist))

    # zoom up on the image so that the brightest pixel is in the center (should give a better fit)
    ys_min = max(0, max_pt[1] - s_size)
    ys_max = min(h, max_pt[1] + s_size)
    xs_min = max(0, max_pt[0] - s_size)
    xs_max = min(w, max_pt[0] + s_size)
    
    # fit a Moffat curve to te star
    centered_img = img[ys_min : ys_max, xs_min : xs_max]
    
    x = '['
    for i in range(centered_img.shape[1]):
        for j in range(centered_img.shape[0]):
            x += '[{}, {}], '.format(str(j), str(i))
    x = x[:-2] + ']'

    y = '['
    for value in centered_img.ravel():
        y += '{}, '.format(str(value))
    y = y[:-2] + ']'

    # input parameter Amplitude, x0, y0, gamma, alpha
    params = '[100, 2, 2, 2, 1]'

    proc = subprocess.Popen(['./CurveFit/runFit', x, y, params], stdout = subprocess.PIPE)
    out, err = proc.communicate()
    if proc.wait() != 0:
        raise CurveFitError('Error fitting curve to star at {}'.format(point))

    output = np.array(out.rstip().split(' ')).astype(float)
    print output
    return None
    
    if outdir is not None:

        xmin, xmax, nx = 0, xb, xb
        ymin, ymax, ny = 0, yb, yb
        x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(x, y)

        z_lim = f.amplitude + 2

        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        ax.plot_surface(X, Y, centered_img, cmap = 'plasma')
        ax.set_zlim(0, z_lim)
        plt.savefig(os.path.join(outdir, 'star_fig_{}_{}'.format(color, point)))

        xmin, xmax, nx = 0, xb, xb * 10
        ymin, ymax, ny = 0, yb, yb * 10
        x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
        
        upscaled = np.kron(centered_img, np.ones((10, 10)))
        
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        ax.plot_surface(X, Y, Z, cmap = 'plasma')
        cset = ax.contourf(X, Y, upscaled, zdir = 'z', offset = -4, cmap = 'plasma')
        ax.set_zlim(-4, z_lim)
        plt.savefig(os.path.join(outdir, 'fit_fig_{}_{}'.format(color, point)))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(upscaled, origin = 'bottom', cmap = 'plasma', extent = (x.min(), x.max(), y.min(), y.max()))
        ax.contour(X, Y, Z, colors = 'w')
        plt.savefig(os.path.join(outdir, 'cont_fig_{}_{}'.format(color, point)))

    return Star(f.x_0 + xs_min, f.y_0 + ys_min, gamma = f.gamma, alpha = f.alpha)
