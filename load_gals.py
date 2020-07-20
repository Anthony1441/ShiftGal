import gzip
import shutil
import numpy as np
import os
import subprocess
from collections import OrderedDict
from astropy.io import fits
from galaxy import Galaxy
from galaxy import Star

colors = ('g', 'i', 'r', 'u', 'z')

class SextractorError(Exception):
    pass


def get_sextractor_points(path, star_class_perc):
    """Runs the sextractor on the given FITS image, returns
       an array of star objects"""
   
    f = None 
    try:
        proc = subprocess.Popen(['./sex', path, '-CATALOG_NAME', 'star_out.txt'], stderr = subprocess.PIPE)
        out, err = proc.communicate()
        res = proc.wait()
    
        if res != 0: raise Exception

        stars = []
    
        f = open('star_out.txt', 'r')
    
        for line in f.readlines()[4:]:
            values = ' '.join(line.rstrip().split()).split()
            # only load in the points that are likely to be stars
            if float(values[3]) >= star_class_perc:
                stars.append(Star(float((values[0])), float((values[1])), float(values[3])))

        return stars
    
    except:
        raise SextractorError
    
    finally:
        if f is not None:
            f.close()
            os.remove('star_out.txt')


def get_fits_from_tar(file_path, color = ''):
    """Extracts a fits image form a fits.gz file and returns
       an astropy fits object representing the image, DOES NOT DELETE FITS FILES"""
    
    if '.gz' in file_path:
        gal_name = file_path.split('.')[-3].split('/')[-1] + '_' + color + '.fits'
        with gzip.open(file_path, 'rb') as source, open('temp_fits/' + gal_name, 'w+') as dest:
            shutil.copyfileobj(source, dest)
    
    else:
        gal_name = file_path.split('.')[-2].split('/')[-1] + '_' + color + '.fits'
        with open(file_path, 'rb') as source, open('temp_fits/' + gal_name, 'w+') as dest:
            shutil.copyfileobj(source, dest)

    return fits.open('temp_fits/' + gal_name, ignore_missing_end = True)


def load_galaxy_SDSS_fits(path, sub_dir, name):
    """Tries to load the wavebands of the galaxy from the directory path.
       The path is expected to be a directory with a directory for each
       waveband in it where it will try to find the matching fits file.
       Returns a dictionary with the files it found."""
    
    gal_dict = dict()

    # look for the fits file in each valid directory
    for dir in os.listdir(path):
        if dir in colors:
            
            if sub_dir is None:
                gal_path = os.path.join(path, dir, name)
            else:
                gal_path = os.path.join(path, dir, sub_dir, name)
            
            if os.path.exists(gal_path):
                gal_dict.update({dir: get_fits_from_tar(gal_path, dir)})
            else:
                 print 'An image for galaxy {} in waveband {} does not exist.'.format(name, dir)
        #else:
            #print 'Skipping {}/{} since it is not a valid waveband directory.'.format(path, dir)

    return OrderedDict(sorted(gal_dict.items()))        


def load_galaxies_SDSS(in_dir, sub_dir, star_class_perc):
    """Generator that yields galaxy objects representing the galaxies found
       in in_dir/*waveband_color*/sub_dir/"""
    
    def yield_gals(sub_dir_name):
            
        for galname in os.listdir(os.path.join(in_dir, 'g', sub_dir_name)):
            gal_dict = load_galaxy_SDSS_fits(in_dir, sub_dir_name, galname)

            # if no images were found
            if not gal_dict: yield None
            star_dict = OrderedDict()
            galname = galname.split('.')[0] 
            # only look for stars in the files that exist
            for color in gal_dict.keys():
                path = 'temp_fits/' + galname + '_' + color + '.fits'
                star_dict.update({color: get_sextractor_points(path, star_class_perc)})
            
            yield Galaxy(gal_dict, star_dict, galname)
 
    
    if sub_dir is None:
        for sub_dir_name in os.listdir(os.path.join(in_dir, 'g')):
            for gal in yield_gals(sub_dir_name): yield gal
    
    elif sub_dir == 'no_sub_dir':
        for gal in yield_gals(''): yield gal
    
    else:
        for gal in yield_gals(sub_dir): yield gal


def load_galaxy_separate(galpath, star_class_perc):
     
    gal_dict = OrderedDict()
    galname = os.path.basename(galpath)
    for color in colors:
        p = os.path.join(galpath, color + '.fits')
        if os.path.exists(p):
            gal_dict.update({color: fits.open(p, ignore_missing_end = True)})
        else:
            p2 = os.path.join(galpath, galname + '_' + color + '.fits')
            if os.path.exists(p2):
                gal_dict.update({color: fits.open(p2, ignore_missing_end = True)})
        
    # if no images were found
    if not gal_dict: return None
        
    # find the stars in the galaxies
    star_dict = OrderedDict()
    for color in gal_dict.keys():
        p = os.path.join(galpath, color + '.fits')
        if os.path.exists(p):
            star_dict.update({color: get_sextractor_points(p, star_class_perc)})
        else:
            p = os.path.join(galpath, galname + '_' + color + '.fits')
            star_dict.update({color: get_sextractor_points(p, star_class_perc)})

    return Galaxy(gal_dict, star_dict, galname)




def load_galaxies_separate(in_dir, star_class_perc):
    """Generator that yields galaxy objects for each galaxy in in_dir.  This assumes
       that in_dir exists and that the the directory structure follows what is expected."""

    for galname in os.listdir(in_dir):
        yield load_galaxy_separate(os.path.join(in_dir, galname), star_class_perc)
     
