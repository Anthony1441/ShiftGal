from astropy.io import fits
import matplotlib
matplotlib.use('agg') # needed for openlab
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import os
import shutil
import find_center
import shift_gal
import copy
import sys
import subprocess


class SpArcFiReError(Exception): pass


def test_smearing(outdir, img, shift_method, vcells = None, cycles = 5, run_sp = False, sp_path = None):
    """Shifts the galaxy back and forth randomly, running it through SpArcFiRe
       each cycle and recorind the galaxy position"""
       
    outdir = os.path.abspath(outdir)
    
    try:
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.mkdir(outdir)
    except:
        print 'Error creating {}, smearing not tested.'.format(outdir)
        return
   
    def run_sparcfire(org, img_data):
        """Runs SpArcFiRe on the files in sf_in, then returns the estimated position of the galaxy"""
        try:
            sf_in = os.path.join(outdir, 'sf_in')
            sf_tmp = os.path.join(outdir, 'sf_tmp')
            sf_out = os.path.join(outdir, 'sf_out')

            os.mkdir(sf_in)
            os.mkdir(sf_tmp)
            os.mkdir(sf_out)
            
            for i in range(cycles):
                os.mkdir(os.path.join(sf_in, str(i)))
                os.mkdir(os.path.join(sf_tmp, str(i)))
                os.mkdir(os.path.join(sf_out, str(i)))
                fits.HDUList([fits.PrimaryHDU(data = img_data[i])]).writeto(os.path.join(sf_in, str(i), 'temp.fits'))
           
            # get starting image center
            os.mkdir(os.path.join(sf_in, 'org'))
            os.mkdir(os.path.join(sf_tmp, 'org'))
            os.mkdir(os.path.join(sf_out, 'org'))
            fits.HDUList([fits.PrimaryHDU(data = org)]).writeto(os.path.join(outdir, 'org', 'temp.fits'))
            proc = subprocess.Popen([sp_path, '-convert-FITS', os.path.join(sf_in, 'org'), os.path.join(sf_tmp, 'org'), os.path.join(sf_out, 'org'), '-generateFitQuality', '0', '-writeBulgeMask', '1']) 

            # run each cycled image in parallel
            procs = [subprocess.Popen([sp_path, '-convert-FITS', os.path.join(sf_in, str(i)), os.path.join(sf_tmp, str(i)), os.path.join(sf_out, str(i)), '-generateFitQuality', '0', '-writeBulgeMask', '1']) for i in range(cycles)]
            for p in procs:
                p.wait()
            
            proc.wait() 
            f = genfromtxt(os.path.join(sf_out, 'org', 'galaxy.tsv'), skip_header = 1)
            pos = (f[20], f[21])

            #once they're done, calculate the change in position
            pos_diff = []
            for i in range(cycles):
                f = genfromtxt(os.path.join(sf_out, str(i), 'galaxy.tsv'), skip_header = 1)
                pos_diff.append(np.sqrt((f[20] - pos[0])**2 + (f[21] - pos[1])**2))

            return pos_diff
        
        except IndexError:
            raise SpArcFiReError
        
        finally:
            try:
                shutil.rmtree(sf_in)
                shutil.rmtree(sf_tmp)
                shutil.rmtree(sf_out)
                for p in os.listdir('.'):
                    if '_settings.txt' in p:
                        os.remove(p)
            except: pass

    sum_diff = []
    org = np.copy(img)
    if run_sp: cycle_imgs = [org]

    for i in range(cycles):
        vector = np.random.random(2)
        print 'Running {}, cycle {}'.format(vector, i)
        img = shift_gal.shift_img(img, vector, shift_method, vcells, check_count = False)
        img = shift_gal.shift_img(img, vector * -1, shift_method, vcells, check_count = False)
        sum_diff.append(100 * np.sum(np.abs(org - img)) / np.sum(org))
        if run_sp: cycle_imgs.append(np.copy(img))

    fits.HDUList([fits.PrimaryHDU(data = img)]).writeto(os.path.join(outdir, 'smear.fits'))
    fits.HDUList([fits.PrimaryHDU(data = np.abs(org - img))]).writeto(os.path.join(outdir, 'smear_residual.fits'))
   
    plt.figure()
    plt.plot(np.arange(len(sum_diff)) + 1, sum_diff)
    plt.title('Difference in Count as a % of Total Original Count')
    plt.xlabel('Cycle')
    plt.ylabel('Percent')
    plt.savefig(os.path.join(outdir, 'sum_diff.png'))
    
    if run_sp:
        
        pos_diff = run_sparcfire(org, cycle_imgs)
        plt.figure()
        plt.plot(np.arange(len(pos_diff)) + 1, pos_diff)
        plt.title('Difference in Predicted Galaxy Center')
        plt.ylabel('Difference in Pixels')
        plt.xlabel('Cycle')
        plt.savefig(os.path.join(outdir, 'pos_diff.png'))
        
       # for p in os.listdir('.'):
       #     if '

def test_params(org_img, vcells, name):
    
    diff = []
    s_img = np.copy(org_img)
    
    s_img = shift_gal.shift_img(s_img, (0.5, 0.5), 'constant')
    s_img = shift_gal.shift_img(s_img, (-0.5, -0.5), 'constant')
    c = np.sum(np.abs(org_img - s_img))
    s_img = np.copy(org_img)
    
    r = np.arange(1, 10, 1) 
    for const in r:
        for _ in range(20):
            s_img = shift_gal.shift_img(s_img, (0.5, 0.5), 'gradient', vcells, range_const = const)
            s_img = shift_gal.shift_img(s_img, (-0.5, -0.5), 'gradient', vcells, range_const = const)
                
        print const, np.sum(np.abs(org_img - s_img))
        diff.append(np.sum(np.abs(org_img - s_img)))
        s_img = np.copy(org_img)
    
    plt.figure()
    plt.plot(r, diff)
    plt.title('Range Constant Value vs. Photon Count Difference (C was {})'.format(c))
    plt.savefig('{}_range_const.png'.format(name))


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

