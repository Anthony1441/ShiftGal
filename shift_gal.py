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
import testing
from collections import OrderedDict
import cv2

class NoViableTemplateError(Exception): pass

class NoStarsFitError(Exception): pass

class StarsNotWithinSigmaError(Exception): pass

class PhotonCountNotPreservedAfterShiftError(Exception): pass

# used to see errors when testing
class NoError(Exception): pass

# file used to write output to
output = None

def filter_stars(src, trg, min_perc_spread_agreement = 0.8, max_star_spread = 10, min_star_spread = 0.1, min_matching_spread = 0.8):
    """Takes in a list of target and filters them out based
       on various parameters, assumes that the source has already been filtered"""
    # filter out stars whose x and y spread are not within 
    # the min_perc_spread_agreement of eachother, this should
    # get rid of non-circular stars
    temp_src, temp_trg = [], []
    for i in range(len(trg)):
        if trg[i].x_spread is not None and min(trg[i].x_spread, trg[i].y_spread) / max(trg[i].x_spread, trg[i].y_spread) >= min_perc_spread_agreement:
            temp_src.append(src[i])
            temp_trg.append(trg[i])
    
    src, trg = temp_src, temp_trg
    assert len(src) == len(trg)

    # filter out stars who have a spread on either axis that is more
    # then max_star_spread, should remove very large stars (more smeared?)
    temp_src, temp_trg = [], []
    for i in range(len(trg)):
        if trg[i].x_spread is not None and trg[i].x_spread <= max_star_spread and trg[i].y_spread <= max_star_spread:
            temp_src.append(src[i])
            temp_trg.append(trg[i])

    src, trg = temp_src, temp_trg
    assert len(src) == len(trg)
    
    # filter out stars who have a spread on either a xis that is less
    # than the min_star_spread, should remove stars that are too small to be used
    temp_src, temp_trg = [], []
    for i in range(len(trg)):
        if trg[i].x_spread is not None and trg[i].x_spread >= min_star_spread and trg[i].y_spread >= min_star_spread:
            temp_src.append(src[i])
            temp_trg.append(trg[i])
    
    src, trg = temp_src, temp_trg
    assert len(src) == len(trg)

    # filter out stars whoose matching star has too disimilar
    # spreads (meaning that one is more smeared/bright than the other
    # and the centers are not to be trusted)
    '''
    temp_src, temp_trg = [], []
    for i in range(len(trg)):
        if (trg[i].x_spread is not None and src[i].x_spread is not None 
            and min(trg[i].x_spread, src[i].x_spread) / max(trg[i].x_spread, src[i].x_spread) >= min_perc_spread_agreement
            and min(trg[i].y_spread, src[i].y_spread) / max(trg[i].y_spread, src[i].y_spread) >= min_perc_spread_agreement):
            
            temp_src.append(src[i])
            temp_trg.append(trg[i])
        
    src, trg = temp_src, temp_trg
    '''
    assert len(src) == len(trg)
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


def shift_img(gal, vector, upscale_factor = 50, check_count = True):
    
    if vector[0] == 0 and vector[1] == 0: return gal, 0
    
    upscale = cv2.resize(gal, dsize = tuple(np.array(gal.shape) * upscale_factor), interpolation = cv2.INTER_LANCZOS4)
    upscale = np.roll(upscale, int(vector[0] * upscale_factor), axis = 1)
    upscale = np.roll(upscale, int(vector[1] * upscale_factor), axis = 0)
    upscale = cv2.resize(upscale, dsize = tuple(gal.shape), interpolation = cv2.INTER_LANCZOS4)
    
    input_count, current_count = np.sum(gal), np.sum(upscale)
    # check to make sure that the output photon count is within 0.05% of the original
    if check_count and (current_count > input_count + input_count * 0.0005 or current_count < input_count - input_count * 0.0005):
        print input_count, current_count
        raise PhotonCountNotPreservedAfterShiftError
    
    return upscale, np.abs(current_count - input_count)


def prints(line, f):
    """Prints line and saves it to f"""
    print line
    f.write(line + '\n')


def find_template_gal_and_stars(galaxy, min_stars_template):
    """Determines which galaxy (if any) should be the template galaxy that 
       all others are shifted to.  Returns the color chosen and the fitted stars."""
    
    # finds the color of the galaxy with the most stars
    template_color = sorted(galaxy.stars_dict.items(), key = lambda x: len(x[1]))[-1][0]
        
    # if there aren't enough stars in the template galaxy then raise error
    if len(galaxy.stars(template_color)) < min_stars_template: raise NoViableTemplateError

    # calculate the center points for the refernce galaxy
    ref_gal, ref_stars = galaxy.images(template_color), []
    
    for star in galaxy.stars(template_color): 
        try:
            fit_star = find_center.estimate_center(ref_gal, star)
            ref_stars.append(fit_star)

        except find_center.CurveFitError:
            pass

    ref_stars = filter_stars(ref_stars, ref_stars)[0]
    return template_color, ref_stars
        

def get_galaxy_vectors(galaxy, template_color, template_stars, min_stars_all):
    """Returns a list of 5 vectors representing the shift needed to align each to
       the template, if None then the a vector could not be found for the galaxy.""" 
    color_vectors = OrderedDict()

    for color, img, stars in galaxy.gen_img_star_pairs():
        # only find matching stars if it is not the template galaxy
        if color == template_color:
            color_vectors.update({color: np.array((0, 0))})
        
        else:
            m_src, m_trg = find_like_points(template_stars, stars)
        
            # if no points in it match the template, skip it
            if len(m_src) < min_stars_all:
                prints('Skipping waveband {} it does not have enough viable stars to use for realignment ({} stars)'.format(color, len(m_src)), output)
                continue

            fit_src, fit_stars = [], []
            
            # try to calculate to subpixel accuracy the center of each star
            for star in m_src:
                try:
                    fit_star = find_center.estimate_center(img, star)
                    fit_stars.append(fit_star)
                    fit_src.append(star)
            
                except find_center.CurveFitError:
                    pass

            fit_src, fit_stars = filter_stars(fit_src, fit_stars)

             # if less than the min stars were fit then skip this waveband
            if len(fit_stars) < min_stars_all:
                prints('Skipping waveband {}, not enough stars could be fit ({} stars)'.format(color, len(fit_stars)), output)
                color_vectors.update({color: None})
                continue

            # calculate the average vector from the reference image and this image
            try:
                color_vectors.update({color: average_vector(fit_src, fit_stars)})
            
            except StarsNotWithinSigmaError:
                prints('Skipping waveband {}, stars disagree too much.'.format(color), output)
                color_vectors.update({color: None})
                continue

    return color_vectors


def shift_wavebands(galaxy, shift_vectors, template_color):
    """Returns a dict of the shifted images (only those that a vector was found for)"""
    
    shifted_imgs = OrderedDict()
    for color, vector in shift_vectors.items():
         if vector is not None:
            org_count = np.sum(galaxy.images(color))
            img, count_diff = shift_img(galaxy.images(color), vector)    
            shifted_imgs.update({color: img})    
            prints('Shifted waveband {} by {} with a flux error of {} / {}'.format(color, tuple(vector), count_diff, org_count), output)
    
    return shifted_imgs


def save_output(outdir, galaxy, shifted_imgs, shift_vectors, save_type, save_originals):
    """Saves the output as fits (and/or png) files"""
    if save_originals: os.mkdir(os.path.join(outdir, 'originals'))
    
    for color in shifted_imgs.keys():

        if save_type in ('png', 'both'):
            # make copies and modify the image so that it is visible as a png
            thres = np.max(galaxy.images(color)) * 0.015
            org, shift = np.copy(galaxy.images(color)), np.copy(shifted_imgs[color])
            org[org > thres] = thres
            shift[shift > thres] = thres
            if save_originals: plt.imsave(os.path.join(outdir, 'originals', galaxy.name + '_' + color + '.png'), org, cmap = 'gray')
            plt.imsave(os.path.join(outdir, galaxy.name + '_' + color + '.png'), shift, cmap = 'gray')

        if save_type in ('fits', 'both'):
            # copy the header info from the original image into the new fits file
            f = galaxy.gal_dict[color]
            w = wcs.WCS(f[0].header, fix = False)
            nf = fits.PrimaryHDU()
            nf.header = f[0].header
            hdu = fits.HDUList([nf])
            if save_originals: nf.writeto(os.path.join(outdir, 'originals', galaxy.name + '_' + color + '.fits'))
            nf.data = shifted_imgs[color]
            # update shifted header and save
            hdu[0].header['CRPIX1'] += int(shift_vectors[color][0])
            hdu[0].header['CRPIX2'] += int(shift_vectors[color][1])
            hdu[0].header['CRVAL1'] += (shift_vectors[color][1] - int(shift_vectors[color][1])) * hdu[0].header['CD1_2']
            hdu[0].header['CRVAL2'] += (shift_vectors[color][0] - int(shift_vectors[color][0])) * hdu[0].header['CD2_1']
            hdu.writeto(os.path.join(outdir,  galaxy.name + '_' + color + '.fits'))



def process_galaxy(galaxy, out_dir, border_size, save_type, min_stars_template, min_stars_all, save_originals, min_wavebands, run_tests, sp_path):
    
    # set up output directory 
    p = os.path.join(out_dir, galaxy.name)
    if os.path.exists(p): shutil.rmtree(p)
    os.mkdir(p)
    
    # set up output files / directories
    global output
    output = open(os.path.join(p, 'output.txt'), 'w')
    
    prints('--- Processing galaxy {} ---'.format(galaxy.name), output)

    try:
        template_color, template_stars = find_template_gal_and_stars(galaxy, min_stars_template)
    except NoViableTemplateError:
        prints('No viable template could be found', output)
        output.close()
        galaxy.close()
        shutil.rmtree(p)
        return
   
    prints('Reference waveband chosen is {} with {} stars'.format(template_color, len(template_stars)), output) 

    shift_vectors = get_galaxy_vectors(galaxy, template_color, template_stars, min_stars_all)
    num_viable = len([1 for v in shift_vectors.values() if v is not None])
    if num_viable < min_wavebands:
        print 'Skipping galaxy, not enough viable wavebands ({} of {} needed)'.format(num_viable, min_wavebands)
        output.close()
        galaxy.close()
        shutil.rmtree(p)
    
    else:  
        galaxy.add_borders(int(len(galaxy.images('g')) * border_size))
        shift_imgs = shift_wavebands(galaxy, shift_vectors, template_color)    
        save_output(p, galaxy, shift_imgs, shift_vectors, save_type, save_originals)
     
        if run_tests:
            prints('Running tests...', output)
            testing.test_shifts(os.path.join(p, 'testing'), galaxy.images(template_color), 10, sp_path)
        
        output.close()

   
   
if __name__ == '__main__':
    # get all of the arguments / option
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', help = 'A directory containing the input galaxies, see -in_format for how this should be structured.')
    parser.add_argument('out_dir', help = 'A directory for the output images.  If it does not exist then it will be created, if it already exists then all files in it will be deleted.')
    parser.add_argument('-star_class_perc', default = 0.65, type = float, help = 'The minimum probablity confidence needed of the sextractor classification that the object is a star.  Value should be in range (0,1), default is 0.7.')
    parser.add_argument('-border_size', default = 0.025, type = float, help = 'Controls size of the border (as a percentage of image height) added to the image to allow room for shifting.')
    parser.add_argument('-save_type', default = 'fits', choices = ['fits', 'png', 'both'], help = 'The output file type of the shifted images, either "fits", "png", or "both".  Default is fits.  WARNING png images are manipulated to display correctly, but should only be used for visual purposes.')
    parser.add_argument('-min_stars_template', default = 3, type = int, help = 'The minimum number of stars needed in the template galaxy (the one that the other wavebands will shifted to match) for a shift to be attempted.  Default is 5')
    parser.add_argument('-min_stars_all', default = 2, type = int, help = 'The minimum number of stars needed in all waveabnds of the galaxy for a shift to be attempted.  Any wavebands that do not satisfy this property are ignored.  Default is 2.')
    parser.add_argument('-save_originals', default = '0', choices = ['True', 'true', '1', 'False', 'false', '0'], help = 'Contols if the original images are saved as part of the output.  Default is false.')
    parser.add_argument('-min_wavebands', default = 4, type = int, help = 'The minimum viable wavebands needed for the output to be saved, if <= 0 then any amount will be saved.')
    parser.add_argument('-run_tests', default = '0', choices = ['True', 'true', '1', 'False', 'false', '0'], help = 'If not 0 then random shifts will be applied to the template waveband (back and forth) the number of times given.  This is used to see the error in shifting.')
    parser.add_argument('-sp_path', default = '{}/SpArcFiRe/scripts/SpArcFiRe'.format(os.getenv('HOME')), help = 'The path to the local install of SpArcFiRe.  This is only needed if cycles_count is not 0.')
    args = parser.parse_args() 
    
    # check that the in directory exists and follows the format required
    if not os.path.isdir(args.in_dir):
        print args.in_dir, 'is not a directory.'
        exit(1)
    
    # check that the star_class_parc is valid
    if args.star_class_perc <= 0 or args.star_class_perc > 1:
        print 'Given star class percentage', args.star_class_perc, 'does not fall within the range (0, 1].'
        exit(1)
    
    # check that border_size is valid
    if args.border_size < 0:
        print 'Border size much be a positive value.'
        exit(1)
    
    # if the output directory does not exist then create it
    try:
        if not os.path.isdir(args.out_dir):
            os.mkdir(args.out_dir)
    except:
        print 'out_dir', args.out_dir, 'could not be created.'
        exit(1)

    t = ('True', 'true', '1')
    args.save_originals = True if args.save_originals in t else False
    args.run_tests = True if args.run_tests in t else False
 
    if args.run_tests and not os.path.exists(args.sp_path):
        print 'The path to your local install of SpArcFiRe install is not valid: {}.'.format(args.sp_path)
        exit(1)
     
    for gal in load_gals.load_galaxies(args.in_dir, args.star_class_perc):
        if type(gal) == str: 
            print 'Failed to load {}'.format(gal)
            continue
        try:
            process_galaxy(gal, args.out_dir, args.border_size, args.save_type, args.min_stars_template, args.min_stars_all, args.save_originals, args.min_wavebands, args.run_tests, args.sp_path)
            print

        except NoError as e:    
            print 'Failed to shift', gal.name
    
    for path in glob.glob('core.*'):
        try: os.remove(path)
        except: pass


