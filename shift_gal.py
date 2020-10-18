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
import multiprocessing
from psutil import virtual_memory

class NoViableTemplateError(Exception): pass

class NoStarsFitError(Exception): pass

class StarsNotWithinSigmaError(Exception): pass

class PhotonCountNotPreservedAfterShiftError(Exception): pass

class ImageTooLargeError(Exception): pass

class NotEnoughMemoryError(Exception):
    def __init__(self, needed_memory):
        print '\n\n{} GB of memory needed, try disabling running in parallel or lowering the upscale factor.\n\n'.format(needed_memory / 1024.0**3)

# used to see errors when testing
class NoError(Exception): pass

# file used to write output to
tsv_out = None


def filter_stars(src, trg, min_gamma = 1, max_gamma = 3, max_dist = np.sqrt(2)):
    """ Filters out stars that are too small, too wide, or whose matching stzr is too far"""

    m_src, m_trg = [], []
    for i in range(len(src)):
        dist = np.sqrt((src[i].x - trg[i].x)**2 + (src[i].y - trg[i].y)**2)
        if dist > max_dist: continue
        if trg[i].gamma < min_gamma or trg[i].gamma > max_gamma: continue
        
        m_src.append(src[i])
        m_trg.append(trg[i])

    assert len(m_src) == len(m_trg)
    return m_src, m_trg


def find_like_points(src, trg):
    """Finds points in trg that are within max_dist distance
       from of a point in src and choses the smallest one.  
       Returns the points found in the same order."""

    m_src, m_trg = [], []

    for trg_star in trg:
        m_star, m_dist = None, np.inf
        for src_star in src:
            temp_dist = np.sqrt((src_star.x - trg_star.x)**2 + (src_star.y - trg_star.y)**2)
            if temp_dist < m_dist:
                m_star = src_star
                m_dist = temp_dist
        
        if m_star is not None:
            m_src.append(m_star)
            m_trg.append(trg_star)

        m_star, m_dist = None, np.inf

    assert len(m_src) == len(m_trg)    
    return m_src, m_trg


def average_vector(m_src, m_trg, maxsigma = 2):
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


def shift_img(gal, vector, upscale_factor, gal_dict = dict(), color = 'NoColor', check_count = True):
    """Shifts the image using Lanczos interoplation, updates the image in gal_dict"""    
    print('Started processing shift {} on waveband {}...'.format(tuple(vector), color)) 
    if vector[0] == 0 and vector[1] == 0:
        gal_dict.update({color: gal}) 
        print('Shifted waveband {} by {} with a flux error of {} / {}'.format(color, tuple(vector), 0, np.sum(gal)))
        return
    
    padding = 0
    # if the image is large enough then its area is greater than 2^31 pixels which causes an overflow error
    # to solve this we pad the image so that it is greater than 2^32 and it believes that it is positive
    size = gal.shape[0] * upscale_factor
    if (size**2) % (2**31) >= 0 and (size**2) % (2**32) >= 2**31:
        while (size**2) % (2**31) >= 0 and (size**2) % (2**32) >= 2**31:
            size += 10
        padding = int((size - gal.shape[0] * upscale_factor) / (2 * upscale_factor)) + 1
        gal = np.pad(gal, padding, 'constant')
    
    upscale = cv2.resize(gal, dsize = tuple(np.array(gal.shape) * upscale_factor), interpolation = cv2.INTER_LANCZOS4)
    upscale = np.roll(upscale, int(round(vector[0] * upscale_factor)), axis = 1)
    upscale = np.roll(upscale, int(round(vector[1] * upscale_factor)), axis = 0)
    upscale = cv2.resize(upscale, dsize = tuple(gal.shape), interpolation = cv2.INTER_LANCZOS4)

    input_count, current_count = np.sum(gal), np.sum(upscale)
    # check to make sure that the output photon count is within 0.05% of the original
    if check_count and (current_count > input_count + input_count * 0.0005 or current_count < input_count - input_count * 0.0005):
        raise PhotonCountNotPreservedAfterShiftError
 
    print('Shifted waveband {} by {} with a flux error of {} / {}'.format(color, tuple(vector), np.abs(input_count - current_count), input_count))

    if padding == 0:
        gal_dict.update({color: upscale})
    else:
        upscale = upscale[padding : -1 * padding, padding : -1 * padding]
        gal_dict.update({color: upscale})

    return upscale    


def tsv_print(*args):
    """Combines the args, separated by tabs, and saves it to the tsv file"""
    tsv_out.write('\t'.join([' ' if arg is None else str(arg) for arg in args]) + '\n')


def find_template_gal_and_stars(galaxy, min_stars_template, outdir = '.', save_star_figs = True):
    """Determines which galaxy (if any) should be the template galaxy that 
       all others are shifted to.  Returns the color chosen and the fitted stars."""
    
    # finds the color of the galaxy with the most stars
    template_color = sorted(galaxy.stars_dict.items(), key = lambda x: len(x[1]))[-1][0]
        
    # if there aren't enough stars in the template galaxy then raise error
    if len(galaxy.stars(template_color)) < min_stars_template: raise NoViableTemplateError

    # calculate the center points for the refernce galaxy
    ref_gal, ref_stars = galaxy.images(template_color), []
   
    outdir = os.path.join(outdir, 'star_figs')
    if os.path.exists(outdir): shutil.rmtree(outdir)
    os.mkdir(outdir)

    for star in galaxy.stars(template_color): 
        try:
            if save_star_figs:
                fit_star = find_center.estimate_center(ref_gal, star, outdir, template_color)
            else:
                fit_star = find_center.estimate_center(ref_gal, star)

            ref_stars.append(fit_star)

        except find_center.CurveFitError, e:
            print e.message

    ref_stars = filter_stars(ref_stars, ref_stars)[0]
    return template_color, ref_stars
        

def get_galaxy_vectors(galaxy, template_color, template_stars, min_stars_all):
    """Returns a list of 5 vectors representing the shift needed to align each to
       the template, if None then the a vector could not be found for the galaxy.""" 
    color_vectors, stars_dict = OrderedDict(), OrderedDict()

    for color, img, stars in galaxy.gen_img_star_pairs():
        # only find matching stars if it is not the template galaxy
        if color == template_color:
            color_vectors.update({color: np.array((0, 0))})
            stars_dict.update({color: template_stars}) 
        else:
            m_src, m_trg = find_like_points(template_stars, stars)
        
            # if no points in it match the template, skip it
            if len(m_src) < min_stars_all:
                print('Skipping waveband {} it does not have enough viable stars to use for realignment ({} stars)'.format(color, len(m_src)))
                color_vectors.update({color: None})
                stars_dict.update({color: None})
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
                print('Skipping waveband {}, not enough stars could be fit ({} stars)'.format(color, len(fit_stars)))
                color_vectors.update({color: None})
                stars_dict.update({color: None})
                continue

            # calculate the average vector from the reference image and this image
            try:
                color_vectors.update({color: average_vector(fit_src, fit_stars)})
                stars_dict.update({color: fit_stars})
            except StarsNotWithinSigmaError:
                print('Skipping waveband {}, stars disagree too much.'.format(color))
                color_vectors.update({color: None})
                stars_dict.update({color: None})
                continue

    return color_vectors, stars_dict


def shift_wavebands(galaxy, shift_vectors, template_color, upscale_factor, run_in_parallel, max_memory):
    """Returns a dict of the shifted images (only those that a vector was found for)"""
    
    shifted_imgs = multiprocessing.Manager().dict()
    procs = [multiprocessing.Process(target = shift_img, args = (galaxy.images(color), vector, upscale_factor, shifted_imgs, color)) for color, vector in shift_vectors.items() if vector is not None]

    if run_in_parallel:
        needed_memory = galaxy.width * galaxy.height * upscale_factor**2 * 8 * len(procs)
        if needed_memory > max_memory:
            raise NotEnoughMemoryError(needed_memory)
        
        for p in procs: p.start()
        for p in procs: p.join()
    
    else:
        needed_memory = galaxy.width * galaxy.height * upscale_factor**2 * 8
        if needed_memory > max_memory:
            raise NotEnoughMemoryError(needed_memory)

        for p in procs:
            p.start()
            p.join()

    return shifted_imgs


def save_output(outdir, galaxy, shifted_imgs, shift_vectors, save_type, save_originals):
    """Saves the output as fits (and/or png) files"""
    if save_originals: os.mkdir(os.path.join(outdir, 'originals'))
    
    for color in shifted_imgs.keys():

        if save_type in ('png', 'both'):
            # make copies and modify the image so that it is visible as a png
            thres = np.mean(galaxy.images(color)) + 10 * np.std(galaxy.images(color))
            org, shift = np.copy(galaxy.images(color)), np.copy(shifted_imgs[color])
            org[org > thres] = thres
            shift[shift > thres] = thres
            if save_originals: plt.imsave(os.path.join(outdir, 'originals', galaxy.name + '_' + color + '.png'), org, cmap = 'gray', origin = 'lower')
            plt.imsave(os.path.join(outdir, galaxy.name + '_' + color + '.png'), shift, cmap = 'gray', origin = 'lower')

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


def process_galaxy(galaxy, out_dir, border_size, save_type, min_stars_template, min_stars_all, save_originals, min_wavebands, run_tests, sp_path, upscale_factor, crop_images, run_in_parallel, max_memory):
    
    # set up output directory 
    p = os.path.join(out_dir, galaxy.name)
    if os.path.exists(p): shutil.rmtree(p)
    os.mkdir(p)
    global tsv_out
    
    print('--- Processing galaxy {} ---'.format(galaxy.name))

    try:
        template_color, template_stars = find_template_gal_and_stars(galaxy, min_stars_template, p)
        template_cpy = np.copy(galaxy.images(template_color))
    except NoViableTemplateError:
        print('No viable template could be found')
        output.close(); galaxy.close(); tsv_out.close()
        shutil.rmtree(p)
        return
   
    print('Reference waveband chosen is {} with {} stars'.format(template_color, len(template_stars))) 
    
    shift_vectors, stars = get_galaxy_vectors(galaxy, template_color, template_stars, min_stars_all)
    num_viable = len([1 for v in shift_vectors.values() if v is not None])
    if num_viable < min_wavebands:
        print 'Skipping galaxy, not enough viable wavebands ({} of {} needed)'.format(num_viable, min_wavebands)
        galaxy.close()
        shutil.rmtree(p)
    
    else:
        if crop_images:
            left, right, top, bottom = galaxy.crop_images_to_galaxy()
            print('Cropped images down to include only the galaxy | X: ({}, {}) | Y: ({}, {})'.format(left, right, top, bottom))
        galaxy.add_borders(int(galaxy.width * border_size))

        try:
            shift_imgs = shift_wavebands(galaxy, shift_vectors, template_color, upscale_factor, run_in_parallel, max_memory)    
        except ImageTooLargeError:
            print 'Input images are too large to be upscaled, skipping galaxy'
            galaxy.close()
            shutil.rmtree(p)
            return
        
        def len_stars(stars):
            return 0 if stars is None else len(stars)

        def print_stars(stars):
            return 'NULL' if stars is None else [str(s) for s in stars]

        save_output(p, galaxy, shift_imgs, shift_vectors, save_type, save_originals)
        tsv_print(galaxy.name, min_stars_template, min_stars_all, upscale_factor,
                  shift_vectors['g'], shift_vectors['i'], shift_vectors['r'], shift_vectors['u'], shift_vectors['z'],
                  len_stars(stars['g']), len_stars(stars['i']), len_stars(stars['r']), len_stars(stars['u']), len_stars(stars['z']),
                  print_stars(stars['g']), print_stars(stars['i']), print_stars(stars['r']), print_stars(stars['u']), print_stars(stars['z']))

        if run_tests:
            print('Running tests...')
            testing.test_shifts(os.path.join(p, 'testing'), galaxy.images(template_color), template_cpy, galaxy.name, 10, sp_path)
        
        
        galaxy.close()

   
   
if __name__ == '__main__':
    # get all of the arguments / option
    parser = argparse.ArgumentParser()
    parser.add_argument('inDir', help = 'A directory containing the input galaxies, see -in_format for how this should be structured.')
    parser.add_argument('outDir', help = 'A directory for the output images.  If it does not exist then it will be created, if it already exists then all files in it will be deleted.')
    parser.add_argument('-saveType', default = 'fits', choices = ['fits', 'png', 'both'], help = 'The output file type of the shifted images, either "fits", "png", or "both".  Default is fits.  WARNING png images are manipulated to display correctly, but should only be used for visual purposes.')
    parser.add_argument('-saveOriginals', default = '0', choices = ['True', 'true', '1', 'False', 'false', '0'], help = 'Contols if the original images are saved as part of the output.  Default is false.')
    parser.add_argument('-starClassPerc', default = 0.65, type = float, help = 'The minimum probablity confidence needed of the sextractor classification that the object is a star.  Value should be in range (0,1), default is 0.7.')
    parser.add_argument('-cropImages', default = '0', choices = ['True', 'true', '1', 'False', 'false', '0'], help = 'If true then the input images will be cropped to only the galaxy using Source Extractor.')
    parser.add_argument('-borderSize', default = 0.01, type = float, help = 'Controls size of the border (as a percentage of image height) added to the image to allow room for shifting.')
    parser.add_argument('-minStarsTemplate', default = 3, type = int, help = 'The minimum number of stars needed in the template galaxy (the one that the other wavebands will shifted to match) for a shift to be attempted.  Default is 5')
    parser.add_argument('-minStarsAll', default = 2, type = int, help = 'The minimum number of stars needed in all waveabnds of the galaxy for a shift to be attempted.  Any wavebands that do not satisfy this property are ignored.  Default is 2.')
    parser.add_argument('-minWavebands', default = 0, type = int, help = 'The minimum viable wavebands needed for the output to be saved, if <= 0 then any amount will be saved.')
    parser.add_argument('-upscaleFactor', default = 100, type = int, help = 'The amount that each image is upscaled using Lanczos interpolation prior to shifting.')
    parser.add_argument('-runInParallel', default = '1', choices = ['True', 'true', '1', 'False', 'false', '0'], help = 'Will process wavebands in parallel, this requires the system to have enough memory to store all upscaled wavebands simultaneously.')
    mem = virtual_memory().total / 1024.0**3
    parser.add_argument('-maxMemory', default = mem, type = float, help = 'The maxmimum amount of memory (in GB) the process can use.  At least 16GB is recommended but more will be needed for larger images and larger upscale factors.')

    parser.add_argument('-runTests', default = '0', choices = ['True', 'true', '1', 'False', 'false', '0'], help = 'If not 0 then random shifts will be applied to the template waveband (back and forth) the number of times given.  This is used to see the error in shifting.')
    parser.add_argument('-spPath', default = '{}/scripts/SpArcFiRe'.format(os.getenv('SPARCFIRE_HOME')), help = 'The path to the local install of SpArcFiRe.')
    args = parser.parse_args() 
    
    # check that the in directory exists and follows the format required
    if not os.path.isdir(args.inDir):
        print args.inDir, 'is not a directory.'
        exit(1)
    
    # check that the star_class_parc is valid
    if args.starClassPerc <= 0 or args.starClassPerc > 1:
        print 'Given star class percentage', args.starClassPerc, 'does not fall within the range (0, 1].'
        exit(1)
    
    # check that border_size is valid
    if args.borderSize < 0:
        print 'Border size much be a positive value.'
        exit(1)

    # check that upscale_factor is valid
    if args.upscaleFactor < 10:
        print 'Upscale factor must be at least 10'
        exit(1)

    # check that the memory is valid
    if args.maxMemory <= 0:
        print 'Maximum memory must be a positive value in bytes'
        exit(1)
    
    # if the output directory does not exist then create it
    try:
        if not os.path.isdir(args.outDir):
            os.mkdir(args.outDir)
    
    except:
        print 'out_dir', args.outDir, 'could not be created.'
        exit(1)

    t = ('True', 'true', '1')
    args.saveOriginals = True if args.saveOriginals in t else False
    args.runTests = True if args.runTests in t else False
    args.cropImages = True if args.cropImages in t else False
    args.runInParallel = True if args.runInParallel in t else False
    
    if args.runTests and not os.path.exists(args.spPath):
        print 'The path to your local install of SpArcFiRe install is not valid: {}.'.format(args.spPath)
        exit(1)
    
    tsv_out = open(os.path.join(args.outDir, 'info.tsv'), 'w')
    tsv_print('objID', 'min_stars_template', 'min_stars_all', 'upscale_factor', 
    'g_vec', 'i_vec', 'r_vec', 'u_vec', 'z_vec', 
    'g_num_stars', 'i_num_stars', 'r_num_stars', 'u_num_stars', 'z_num_stars',
    'g_star_list', 'i_star_list', 'r_star_list', 'u_star_list', 'z_star_list')


    for gal in load_gals.load_galaxies(args.inDir, args.starClassPerc):
        if type(gal) == str: 
            print 'Failed to load {}'.format(gal)
            continue
        try:
            process_galaxy(gal, args.outDir, args.borderSize, args.saveType, args.minStarsTemplate, args.minStarsAll, args.saveOriginals, args.minWavebands, args.runTests, args.spPath, args.upscaleFactor, args.cropImages, args.runInParallel, args.maxMemory * 1024**3)
            print

        except NoError:    
            print 'Failed to shift {}\n'.format(gal.name)

    tsv_out.close()

