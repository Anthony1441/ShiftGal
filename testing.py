from astropy.io import fits
import matplotlib
matplotlib.use('agg') # needed for openlab
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import find_center
import shift_gal
import copy
import sys

def test_smearing(outdir, img, shift_method, vcells, cycles = 30, save_data = True):
    """Shifts the galaxies in galpath back and forth to create a smearing effect"""
    try:
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.mkdir(outdir)
    except:
        print 'Error creating {}'.format(outdir)
        return
    
    if save_data: sum_diff, max_diff = [], []

    org = np.copy(img)

    for i in range(cycles):
        print i
        vector = np.random.random(2)
        img = shift_gal.shift_img(img, vector, shift_method, vcells)
        img = shift_gal.shift_img(img, vector * -1, shift_method, vcells)
        if save_data:
            sum_diff.append(100 * np.sum(np.abs(org - img)) / np.sum(org))
            max_diff.append(abs(np.max(org) - np.max(img)))
    
    nf = fits.PrimaryHDU()
    nf.data = img
    gal = fits.HDUList([nf])
    gal.writeto(os.path.join(outdir, 'smear.fits'))
    nf.data = np.abs(org - img)
    gal = fits.HDUList([nf])
    gal.writeto(os.path.join(outdir, 'smear_residual.fits'))
        
    if save_data:
        plt.figure()
        plt.plot(np.arange(len(sum_diff)), sum_diff)
        plt.title('Absolute Difference in Photon Count')
        plt.xlabel('Number of Repeated Shifts')
        plt.ylabel('Difference in Count as a % of Total Count')
        plt.savefig(os.path.join(outdir, 'sum_diff.png'))
        plt.figure()
        plt.plot(np.arange(len(max_diff)), max_diff)
        plt.title('Difference in Max Pixel Value')
        plt.xlabel('Number of Repeated Shifts')
        plt.ylabel('Differnece in Photon Count')
        plt.savefig(os.path.join(outdir, 'max_diff.png'))
        sum_diff, max_diff = [], []

"""
def test_shift_params():

    if os.path.exists('paramtest'):
        shutil.rmtree('paramtest')
    os.mkdir('paramtest')
    
    gal = fits.open('testgal/1237648702967251093/i.fits', ignore_missing_end = True)
    org_img = gal[0].data
    s_img = np.copy(org_img)
 
    for const in np.arange(19, 20, 1):
        for sigma in np.arange(0, 1, 2):
            for mean_const in np.arange(30, 31, 1):
                for _ in range(10):
                    s_img = shift_gal.shift_img(s_img, (0.5, 0.5), 'gradient', const = const, sigma = sigma, mean_const = mean_const)
                    s_img = shift_gal.shift_img(s_img, (-0.5, -0.5), 'gradient', const = const, sigma = sigma, mean_const = mean_const)
                
                print const, sigma, mean_const, np.sum(np.abs(org_img - s_img))
                
                
                s_img = np.copy(org_img)
"""
def calc_shift_residuals(galpath, outdir, shifts, shift_method):
    """Calculates the residual of the original waveband shifted and then shifted back"""

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
    
    galname = os.path.basename(galpath)
    output = open(os.path.join(galpath, 'output.txt'), 'w')
    colors = ('_g', '_i', '_r', '_u', '_z')
    output_paths = sorted([os.path.join(galpath, f) for f in os.listdir(galpath) if '.fits' in f])
    input_paths = sorted([os.path.join(galpath, 'inputs', f) for f in os.listdir(os.path.join(galpath, 'inputs')) if '.fits' in f])

    assert len(input_paths) == len(output_paths) == len(shifts)
    
    for inp, out, vec in zip(input_paths, output_paths, shifts):
        inp_img = fits.open(inp, ignore_missing_end = True)
        out_img = fits.open(out, ignore_missing_end = True)
        assert inp_img[0].data.shape == out_img[0].data.shape
        residual = inp_img[0].data - shift_gal.shift_img(out_img[0].data, np.array(vec) * -1, shift_method)
        inp_img[0].data = residual
        inp_img.writeto(os.path.join(outdir, '{}_{}_{}.fits'.format(os.path.basename(inp).split('.')[0], np.min(residual), np.max(residual))))
    


def calc_star_residual(s1, s2):
    """Returns a normalized array of the residual between the two stars."""
    #s1, s2  = s1 - np.min(s1), s2 - np.min(s2)
    s1, s2 = s1 / np.max(s1), s2 / np.max(s2)
    sub = s1 - s2
    return sub + np.min(sub)


def calc_star_residuals(galpath, outdir, star_class_perc):
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
            c1_stars, c2_stars = shift_gal.find_like_points(gal.stars(c1), gal.stars(c2))
            
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
                    size = int(max(c1_fit.x_spread, c1_fit.y_spread, c2_fit.x_spread, c2_fit.y_spread) + 10)
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
                    scale = (10, 10)
                    s1_scale, s2_scale = np.kron(s1, np.ones(scale)), np.kron(s2, np.ones(scale))
                    residual = np.kron(residual, np.ones(scale))
                    
                    
                    plt.imsave(os.path.join(outdir, '{}_and_{}_at_{}_{}_diff_{}a.png'.format(c1, c2, s1x, s1y, np.max(residual) - np.min(residual))), s1_scale, cmap = 'gray')
                    plt.imsave(os.path.join(outdir, '{}_and_{}_at_{}_{}_diff_{}b.png'.format(c1, c2, s1x, s1y, np.max(residual) - np.min(residual))), s2_scale, cmap = 'gray')
                    plt.imsave(os.path.join(outdir, '{}_and_{}_at_{}_{}_diff_{}c.png'.format(c1, c2, s1x, s1y, np.max(residual) - np.min(residual))), residual, cmap = 'gray')

#test_shift_params()
