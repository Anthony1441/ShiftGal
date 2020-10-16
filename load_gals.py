import numpy as np
import os
import subprocess
from collections import OrderedDict
from astropy.io import fits
from galaxy import Galaxy
from galaxy import Star


class SextractorError(Exception): pass

def load_fits(path):
    f = fits.open(path, ignore_missing_end = True)
    img = np.copy(f[0].data)
    f.close()
    return img

def save_fits(img, path):
    fits.HDUList([fits.PrimaryHDU(data = img)]).writeto(path, overwrite = True)

def get_seg_img(img):
    """Runs Source Extractor on the given FITS image and
       returns the segmenation image"""
    
    seg = None
    save_fits(img, 'temp.fits')
    try:
        proc = subprocess.Popen(['./sex', 'temp.fits', '-CHECKIMAGE_TYPE', 'SEGMENTATION', '-CHECKIMAGE_NAME', 'temp_seg.fits', '-CATALOG_NAME', 'temp_seg.txt'], stderr = subprocess.PIPE) 
        if proc.wait() != 0: 
            raise SextractorError
    
        seg = fits.open('temp_seg.fits', ignore_missing_end = True)
        seg_img = seg[0].data
    
    except:
        raise SextractorError
    
    finally:
        if os.path.exists('temp_seg.fits'): os.remove('temp_seg.fits')
        if os.path.exists('temp_seg.txt'): os.remove('temp_seg.txt')
        if os.path.exists('temp.fits'): os.remove('temp.fits')
        if seg is not None: seg.close()

    return seg_img


def get_sextractor_points(path):
    """Runs the sextractor on the given FITS image, returns
       an array of star objects"""
    f = None 
    try:
        proc = subprocess.Popen(['./sex', path, '-CATALOG_NAME', 'star_out.txt'], stderr = subprocess.PIPE)
        if proc.wait() != 0: 
            raise Exception
        stars = []
    
        f = open('star_out.txt', 'r') 
        for line in f.readlines()[4:]:
            values = ' '.join(line.rstrip().split()).split()
            stars.append(Star(float(values[0]), float(values[1]), float(values[3])))

        return stars
    
    except:
        raise SextractorError
    
    finally:
        if f is not None:
            f.close()
            os.remove('star_out.txt')


def load_galaxy(galpath, star_class_perc):
    """Loads the galaxy loacated at galpath and returns a Galaxy object""" 
    gal_dict = OrderedDict()
    galname = os.path.basename(galpath)
    for color in ('g', 'i', 'r', 'u', 'z'):
        p = os.path.join(galpath, color + '.fits')
        if os.path.exists(p):
            gal_dict.update({color: fits.open(p, ignore_missing_end = True)})
            continue
    
        p2 = os.path.join(galpath, galname + '_' + color + '.fits')    
        if os.path.exists(p2):
            gal_dict.update({color: fits.open(p2, ignore_missing_end = True)})
            continue

        p3 = os.path.join(galpath, galname + '-' + color + '.fits')
        if os.path.exists(p3):
            gal_dict.update({color: fits.open(p3, ignore_missing_end = True)})
            continue

    # if no images were found
    if not gal_dict: return galpath
        
    # find the stars in the galaxies
    star_dict = OrderedDict()
    for color in gal_dict.keys():
        try:
            p = os.path.join(galpath, color + '.fits')
            if os.path.exists(p):
                star_dict.update({color: get_sextractor_points(p)})
                continue

            p2 = os.path.join(galpath, galname + '_' + color + '.fits')
            if os.path.exists(p2):    
                star_dict.update({color: get_sextractor_points(p2)})
                continue

            p3 = os.path.join(galpath, galname + '-' + color + '.fits')
            if os.path.exists(p3):
                star_dict.update({color: get_sextractor_points(p3)})
                continue
        except IndexError:#SextractorError:
            return galpath

    return Galaxy(gal_dict, star_dict, star_class_perc, galname)


def load_galaxies(in_dir, star_class_perc):
    """Generator that yields galaxy objects for each galaxy in in_dir.  This assumes
       that in_dir exists and that the the directory structure follows what is expected."""

    for galname in os.listdir(in_dir):
        yield load_galaxy(os.path.join(in_dir, galname), star_class_perc)
     
