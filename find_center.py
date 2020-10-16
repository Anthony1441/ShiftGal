import numpy as np
from galaxy import Star
from astropy.modeling import functional_models, fitting

class CurveFitError(Exception):
    def __init__(self, message):
        self.message = message


def estimate_center(img, star, percent_img_to_explore = 0.1):
    """
    img: Full galaxy image
    star: Star object
    percent_of_img_to_explore: since the sextractor points are not very accurate a search is done around the point
        to try and find the actual maximum point.  It will explore in a square 2 * img.height * percent wide and high
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
    s_size = 8
    
    ys_min = max(0, max_pt[1] - s_size)
    ys_max = min(h, max_pt[1] + s_size)
    xs_min = max(0, max_pt[0] - s_size)
    xs_max = min(w, max_pt[0] + s_size)
    
    # fit a Moffat curve to te star
    centered_img = img[ys_min : ys_max, xs_min : xs_max]
    amp = np.max(centered_img)
    yb, xb = centered_img.shape
    y_grid, x_grid = np.mgrid[:yb, :xb]
    f_init = functional_models.Moffat2D(amplitude = np.max(centered_img), x_0 = xb / 2.0, y_0 = yb / 2.0, bounds = {'x_0': (0, xb), 'y_0': (0, yb), 'amplitude': (amp - 0.5 * amp, amp + 0.5 * amp), 'gamma': (0, 2 *s_size)})
    fit_f = fitting.LevMarLSQFitter()
    f = fit_f(f_init, x_grid, y_grid, centered_img, maxiter = 1000000)
    
    return Star(f.x_0 + xs_min, f.y_0 + ys_min, gamma = f.gamma, alpha = f.alpha)
