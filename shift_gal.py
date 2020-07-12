import numpy as np
import matplotlib
matplotlib.use('agg') # need this for openlab
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
import argparse
import glob
import os
import shutil
import sys
import load_gals
import find_center
import random


class StarsNotWithinSigmaError(Exception): pass


def find_highest_contrast_waveband(galaxy):
    """Returns the color of the highest contrast waveband"""
    diffs = {color: np.amax(img) - np.amin(img) for color, img in galaxy.images()}
    return sorted(diffs.items(), key = lambda x: x[1])[-1][0]


def find_max_stars_waveband(galaxy):
    """Returns the color of the waveband with the most stars"""
    return sorted(galaxy.stars_dict.items(), key = lambda x: len(x[1]))[-1][0]
         

def filter_stars(src, trg, min_perc_spread_agreement = 0.8, max_star_spread = 2, min_star_spread = 0.1):
    """Takes in a list of target and filters them out based
       on various parameters, assumes that the source has already been filtered"""
    
    print '# pre agreement filter', len(trg)
    # filter out stars whose x and y spread are not within 
    # the min_perc_spread_agreement of eachother
    temp_src, temp_trg = [], []
    for i in range(len(trg)):
        if trg[i].x_spread is not None and min(trg[i].x_spread, trg[i].y_spread) / max(trg[i].x_spread, trg[i].y_spread) >= min_perc_spread_agreement:
            temp_src.append(src[i])
            temp_trg.append(trg[i])
    
    src, trg = temp_src, temp_trg
    assert len(src) == len(trg)

    print '# pre max spread filter', len(trg)
    # filter out stars who have a spread on either axis that is more
    # then max_star_spread, should remove very large stars (more smeared?)
    temp_src, temp_trg = [], []
    for i in range(len(trg)):
        if trg[i].x_spread is not None and trg[i].x_spread <= max_star_spread and trg[i].y_spread <= max_star_spread:
            temp_src.append(src[i])
            temp_trg.append(trg[i])

    src, trg = temp_src, temp_trg
    assert len(src) == len(trg)
    
    print '# pre min spread filter', len(trg)
    # filter out stars who have a spread on either a xis that is less
    # than the min_star_spread, should remove stars that are too small to be used
    temp_src, temp_trg = [], []
    for i in range(len(trg)):
        if trg[i].x_spread is not None and trg[i].x_spread >= min_star_spread and trg[i].y_spread >= min_star_spread:
            temp_src.append(src[i])
            temp_trg.append(trg[i])
    
    src, trg = temp_src, temp_trg
    assert len(src) == len(trg)
    print '# remaining', len(trg)
    return src, trg


def find_like_points(src, trg, max_dist = 10):
    """Finds points in trg that are within max_dist distance
       from of a point in src and choses the smallest one.  
       Returns the points found in the same order."""

    INF = 10000
    m_src, m_trg = [], []

    for trg_star in trg:
        m_star, m_dist = None, INF
        for src_star in src:
            temp_dist = np.sqrt((src_star.x - trg_star.x)**2 + (src_star.y - trg_star.y)**2)
            if temp_dist <= max_dist and temp_dist < m_dist:
                m_star = src_star
                m_dist = temp_dist
        
        if m_star is not None:
            m_src.append(m_star)
            m_trg.append(trg_star)

        m_star, m_dist = None, INF

    assert len(m_src) == len(m_trg)    
    return m_src, m_trg


def average_vector(m_src, m_trg, maxsigma = 1):
    """Returns the average vector between the source and target points.
       It excludes points outside of maxsigma * std_dev of the mean."""
    # calculate the vector bettween each star and its source star
    vecs = np.array(m_src) - np.array(m_trg)
    mu, dev = np.mean(vecs), np.std(vecs) * maxsigma
    # only vectors that are within one sigma on either side of the mean
    avg = np.mean([v for v in vecs if abs(v[0] - mu[0]) < dev[0] and abs(v[1] - mu[1]) < dev[1]], axis = 0)
    # if there are no points in that range, then the stars don't agree enough for the shift to be trusted
    if np.any(np.isnan(avg)): raise StarsNotWithinSigmaError
    return avg


def shift_img(gal, vector):
    """shifts the image by the 2D vector given"""

    # shift the integer part of the x-axis
    gal = np.roll(gal, int(vector[0]), axis = 1)
    # shift over the decimal part
    x_rem = vector[0] % int(vector[0]) if abs(vector[0]) >= 1 else vector[0]
    x_shift = np.roll(gal, int(np.sign(x_rem)), axis = 1)
    gal = gal - x_rem * gal + x_rem * x_shift

    # shift along y-axis
    gal = np.roll(gal, int(vector[1]), axis = 0)
    y_rem = vector[1] % int(vector[1]) if abs(vector[1]) >= 1 else vector[1]
    y_shift = np.roll(gal, int(np.sign(y_rem)), axis = 0)
    gal = gal - y_rem * gal + y_rem * y_shift

    return gal


def process_galaxy(galaxy, out_dir, include_border, out_type, verbose_output, template, min_stars_template, min_stars_all, save_inputs, save_points, save_info):
 
    print '\n--- Processing galaxy {} ---'.format(galaxy.name)
    p = out_dir + '/' + galaxy.name

    # find the template waveband based on the template argument
    if template == 'max_contrast':
        template_color = find_highest_contrast_waveband(galaxy)
    elif template == 'max_stars':
        template_color = find_max_stars_waveband(galaxy)
    else:
        raise Exception

    if save_info: out_file = open('{}/info.txt'.format(p), 'w')

    # if there aren't enough stars in the template galaxy then return
    if len(galaxy.stars(template_color)) < min_stars_template:
        print 'Reference waveband', template_color, 'does not have enough stars.'
        if save_info: out_file.write('Reference waveband: {}\n'.format(template_color))
        return

    # calculate the center points for the refernce galaxy
    ref_gal = galaxy.images(template_color)

    if verbose_output:
        print 'Reference waveband chosen is {}.'.format(template_color)
        print 'Calculating reference waveband {}\'s center points.'.format(template_color)

    ref_stars = []
    for star in galaxy.stars(template_color):
        
        try:
            fit_star = find_center.estimate_center(ref_gal, star)
            ref_stars.append(fit_star)
            # if verbose_output: print '{} -> {}'.format(star, fit_star)
        
        except find_center.CurveFitError:
            pass
            # if verbose_output: print 'Skipping a star at position {}, no curve could be fit.'.format(star)       

    ref_stars = filter_stars(ref_stars, ref_stars)[0]
    if save_info: out_file.write('Number of stars in reference waveband {}\n'.format(len(ref_stars)))
   
    # try to make a new directory to store the images in
    # if it already exists then the contents will be cleared
    try:
        os.mkdir(p)
    except:
        shutil.rmtree(p)
        os.mkdir(p)
    
    if save_inputs: os.mkdir(p + '/inputs')
    if save_points: os.mkdir(p + '/points')
    
    # loop over the information for each waveband
    for color, img, stars in galaxy.gen_img_star_pairs():
        # only find matching stars if it is not the template galaxy
        if color == template_color:
            vector = (0, 0)
            m_src, fit_stars = ref_stars, ref_stars
        else:
            m_src, m_trg = find_like_points(ref_stars, stars)
        
            # if no points in it match the template, skip it
            if len(m_src) < min_stars_all:
                if verbose_output: print 'Skipping waveband {} it does not have enough viable stars to use for realignment.'.format(color)
                continue
            else:
                if verbose_output: print 'Calculating waveband {}\'s center points.'.format(color)

            fit_src, fit_stars = [], []
            # try to calculate to subpixel accuracy the center of each star
            for star in m_src:
            
                try:
                    fit_star = find_center.estimate_center(img, star)
                    fit_stars.append(fit_star)
                    fit_src.append(star)
                    # if verbose_output: print '{} -> {}'.format(star, fit_star)
            
                except find_center.CurveFitError:
                    #if verbose_output: print 'Skipping a star at position {} in galaxy {} waveband {}, no curve could be fit.'.format(star, galaxy.name, color)
                    pass

            fit_src, fit_stars = filter_stars(fit_src, fit_stars)

            # if less than the min stars were fit then skip this waveband
            if len(fit_stars) < min_stars_all:
                if verbose_output: print 'Skipping waveband {}, not enough stars could be fit.'.format(color)
                continue

            # calculate the average vector from the reference image and this image
            try:
                vector = average_vector(fit_src, fit_stars)
            except StarsNotWithinSigmaError:
                if verbose_output: print 'Skipping waveband {}, stars disagree too much.'.format(color)
                continue
        
        # END IF

        # shift the image over the average vector, padding it so that no information is lost on the edges
        padded_img = np.pad(img, int(len(img) * 0.1), 'constant', constant_values = 0) if include_border else img
        shifted_img = shift_img(padded_img, vector)
        print 'Shifted waveband {} by vector {}.'.format(color, tuple(vector))
        if save_info: out_file.write('Waveband {} used {} stars for vector {}\n'.format(color, len(fit_stars), tuple(vector)))

        def save_png():
            threshold = np.min(img) + (np.max(img) - np.min(img)) * 0.1
            
            if save_inputs: 
                _padded_img = np.copy(padded_img)
                _padded_img[_padded_img > threshold] = threshold
                plt.imsave(p + '/inputs/' + galaxy.name + '_' + color + '.png', _padded_img, cmap = 'gray')
            
            _shifted_img = np.copy(shifted_img)
            _shifted_img[_shifted_img > threshold] = threshold
            plt.imsave(p + '/' + galaxy.name + '_' + color + '.png', _shifted_img, cmap = 'gray')
       

        def save_fits():
            # copy the header info from the original image into the new fits file
            f = galaxy.gal_dict[color]
            w = wcs.WCS(f[0].header, fix = False)
            nf = fits.PrimaryHDU()
            nf.data = shifted_img
            nf.header = f[0].header
            nf.header.update(w.to_header())
            hdu = fits.HDUList([nf])
            hdu.writeto(p + '/' + galaxy.name + '_' + color + '.fits')
            if save_inputs: f.writeto(p + '/inputs/' + galaxy.name + '_' + color + '.fits')
        

        if out_type == 'png':
            save_png()

        elif out_type == 'fits':
            save_fits()

        elif out_type == 'both':
            save_png()
            save_fits()
        
        if save_points:
            # save the image with points (as png)
            plt.figure()
            plt.imshow(img, cmap = 'gray')
            m_x, m_y = [s.x for s in m_src], [s.y for s in m_src]
            plt.scatter(m_x, m_y, s = 2, color = 'blue')
            f_x, f_y = [s.x for s in fit_stars], [s.y for s in fit_stars]
            plt.scatter(f_x, f_y, s = 2, color = 'red')
            plt.savefig(p + '/points/' + galaxy.name + '_' + color + '.png')


if __name__ == '__main__':
    
    # get all of the arguments / options
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', help = 'A directory containing the input galaxies, see -in_format for how this should be structured.')
    parser.add_argument('out_dir', help = 'A directory for the output images.  If it does not exist then it will be created, if it already exists then all files in it will be deleted.')
    parser.add_argument('-star_class_perc', default = 0.7, type = float, help = 'The minimum probablity confidence needed of the sextractor classification that the object is a star.  Value should be in range (0,1], default is 0.7.')
    parser.add_argument('-include_border', default = 'True', choices = ['True', 'False', 'true', 'false', '0', '1'], help = 'Controls if the output images will include a black border around them so that no information in the image is lost in the shift.  Default is true.')
    parser.add_argument('-sub_dir', default = None, help = 'If -in_format is "SDSS" then this controls which sub directory of galaxies is processed, if not set then all galaxies are processed.  If in SDSS format but does not contain sub directories set this to "no_sub_dir".  If in_format is "separate" then this option is ignored.')
    parser.add_argument('-in_format', default = 'separate', choices = ['SDSS', 'separate'], help = 'Sets the file structure that the input galaxies will follow.  If "SDSS" then it expects in_dir to contain 5 wavebands labeled "g", "i", "r", "u", and "z", where in each waveband directory there can either be sub-directories (i.e. 000, 001...) with galaxy files in them or the galaxy files can be in the waveband directories directly.  If "separate" then in_dir should contain one directory for each galaxy and each one of those should have fits images labeled g.fits, r.fits, etc...')
    parser.add_argument('-out_type', default = 'fits', choices = ['fits', 'png', 'both'], help = 'The output format of the shifted images, either "fits", "png", or "both".  Default is fits.')
    parser.add_argument('-verbose_output', default = 'True', choices = ['True', 'False', 'true', 'false', '0', '1'], help = 'If True output for each star is shown along with other error messages, if False only the galaxy names are outputted.  Default is True') 
    parser.add_argument('-template', default = 'max_stars', choices = ['max_stars', 'max_contrast'], help = 'Controls how the template galaxy (the one that the other wavebands will be shifted to match) is chosen.  If "max_stars" then the waveband with the most stars is chosen.  If "max_contrast" then the waveband with the highest contrast is chosen.  Default is "max_stars".')  
    parser.add_argument('-min_stars_template', default = 10, type = int, help = 'The minimum number of stars needed in the template galaxy (the one that the other wavebands will shifted to match) for a shift to be attempted.  Default is 10')
    parser.add_argument('-min_stars_all', default = 3, type = int, help = 'The minimum number of stars needed in all waveabnds of the galaxy for a shift to be attempted.  Any wavebands that do not satisfy this property are ignored.  Default is 3.')
    parser.add_argument('-save_inputs', default = 'False', choices = ['True', 'true', '1', 'False', 'false', '0'], help = 'Contols if the input images are saved as part of the output.  Default is false.')
    parser.add_argument('-save_points', default = 'False', choices = ['True', 'true', '1', 'False', 'false', '0'], help = 'Controls if an image with the sextractor and fitted points if saved as part of the output.  Default is false.')
    parser.add_argument('-save_info', default = 'True', choices = ['True', 'true', '1', 'False', 'false', '0'], help = 'If true then a txt file is saved with the shifts for each waveband listed and other useful information.')
    args = parser.parse_args()
    
    # check that the in directory exists and follows the format required
    if not os.path.isdir(args.in_dir):
        print args.in_dir, 'is not a directory.'
        exit(1)
    
    elif args.in_format == 'SDSS':
        paths = (args.in_dir + '/g', args.in_dir + '/i', args.in_dir + '/r', args.in_dir + '/u', args.in_dir + '/z')

        for p in paths:
            if not os.path.isdir(p):
                print args.in_dir, 'does not follow the required directory heirarchy, (', p, ') does not exist.'
                exit(1) 
    
    # check that the star_class_parc is valid
    if args.star_class_perc <= 0 or args.star_class_perc > 1:
        print 'Given star class percentage', args.star_class_perc, 'does not fall within the range (0, 1].'
        exit(1)

    # if the output directory does not exist then create it
    try:
        if not os.path.isdir(args.out_dir):
            os.mkdir(args.out_dir)
    except:
        print 'out_dir', args.out_dir, 'could not be created.'
        exit(1)

    t = ('True', 'true', '1')
    args.include_border = True if args.include_border in t else False
    args.verbose_output = True if args.verbose_output in t else False
    args.save_inputs = True if args.save_inputs in t else False
    args.save_points = True if args.save_points in t else False
    args.save_info = True if args.save_info in t else False

    if not os.path.exists('temp_fits'): os.mkdir('temp_fits')
        
    # run ShiftGal, if an error occurs still remove all temp files
    try:
        
        if args.in_format == 'SDSS':
            for gal in load_gals.load_galaxies_SDSS(args.in_dir, args.sub_dir, args.star_class_perc):
                if gal is None: continue
                process_galaxy(gal, args.out_dir, args.include_border, args.out_type, args.verbose_output, args.template, args.min_stars_template, args.min_stars_all, args.save_inputs, args.save_points, args.save_info)

        else:
            for gal in load_gals.load_galaxies_separate(args.in_dir, args.star_class_perc):
                if gal is None: continue
                process_galaxy(gal, args.out_dir, args.include_border, args.out_type, args.verbose_output, args.template, args.min_stars_template, args.min_stars_all, args.save_inputs, args.save_points, args.save_info)
    
    except NameError as e:    
        raise e

    finally:
        try:
            if os.path.exists('temp_fits'):
                shutil.rmtree('temp_fits')
        except:
            print 'Failed to delete temp fits.'
            pass
        # if the star couldn't be fit (often when near the border of the image)
        # then remove the core dump(s) it produced
        for path in glob.glob('core.*'):
            try:
                os.remove(path)
            except:
                pass


