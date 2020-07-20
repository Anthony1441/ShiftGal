import matplotlib
matplotlib.use('agg') # needed for openlab
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import find_center
import load_gals
from shift_gal import find_like_points


def calc_star_residual(s1, s2):
    """Returns a normalized array of the residual between the two stars."""
    #s1, s2  = s1 - np.min(s1), s2 - np.min(s2)
    s1, s2 = s1 / np.max(s1), s2 / np.max(s2)
    return np.abs(s1 - s2)


def calc_residuals(galpath, outdir, star_class_perc):
    """Calcualtes the star redisuals for each waveaband in all other wavebands present.
       Expects galpath to conatin each waveband as 'color.fits', output is saved by star."""
    
    if not os.path.exists(galpath):
        print '{} is not a valid directory'.format(galpath)
        return

    try:
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.mkdir(outdir)
    except:
        print 'Error creating {}'.format(outdir)
        return

    # load shifted galaxies
    gal = load_gals.load_galaxy_separate(galpath, star_class_perc)
    
    # for each combination of wavebands, find the stars that are in both
    # then calculate a redisual for both
    for c1 in gal.colors():
        for c2 in gal.colors():
            if c1 == c2: continue
            c1_stars, c2_stars = find_like_points(gal.stars(c1), gal.stars(c2))
            
            # for each pair (which should be the same star), isolate the image
            # containing it and compare thier difference
            for c1_star, c2_star in zip(c1_stars, c2_stars):
                # fit each star to find (roughly) the size of it
                # this doesn't need to be perfect but should give a rough
                # estimage of how big the image needs to be for the residual
                try:
                    c1_fit = find_center.estimate_center(gal.images(c1), c1_star)
                    c2_fit = find_center.estimate_center(gal.images(c2), c2_star)
                    
                except find_center.CurveFitError:
                    pass

                else:
                    # add 2 pixels on either size, and 0.5 so that rounding is done correctly 
                    size = int(max(c1_fit.x_spread, c1_fit.y_spread, c2_fit.x_spread, c2_fit.y_spread) + 2.5)
                    s1x, s1y, s2x, s2y = int(c1_star.x), int(c1_star.y), int(c2_star.x), int(c2_star.y)
                    left = max(0, s1x - size, s2x - size)
                    right = min(gal.images(c1).shape[1], s1x + size, s2x + size)
                    top = max(0, s1y - size, s2y - size)
                    bottom = min(gal.images(c1).shape[0], s1y + size, s2y + size)
                    
                    if left == right or top == bottom: continue

                    s1 = gal.images(c1)[top : bottom, left : right]
                    s2 = gal.images(c2)[top : bottom, left : right]
                    residual = calc_star_residual(s1, s2)

                    # scale up for easier image viewing
                    residual = np.kron(residual, np.ones((10, 10)))
                    plt.imsave(os.path.join(outdir, '{}_and_{}_at_{}_{}_diff_{}.png'.format(c1, c2, s1x, s1y, np.max(residual) - np.min(residual))), residual, cmap = 'gray')





