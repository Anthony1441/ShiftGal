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
import load_gals
import find_center

class SpArcFiReError(Exception): pass
class_prob = 0.06


def run_sparcfire(org, img_data, irange, outdir, sp_path):
    """Runs SpArcFiRe on the org and ima_data, then returns the difference in galaxy center for org and img_data"""
    
    try:
        sf_in, sf_tmp, sf_out = os.path.join(outdir, 'sf_in'), os.path.join(outdir, 'sf_tmp'), os.path.join(outdir, 'sf_out')
        os.mkdir(sf_in); os.mkdir(sf_tmp); os.mkdir(sf_out)
        
        for i in range(irange):
            os.mkdir(os.path.join(sf_in, str(i)))
            os.mkdir(os.path.join(sf_tmp, str(i)))
            os.mkdir(os.path.join(sf_out, str(i)))
            load_gals.save_fits(img_data[i], os.path.join(sf_in, str(i), 'temp.fits'))
       
        # get starting image center
        os.mkdir(os.path.join(sf_in, 'org'))
        os.mkdir(os.path.join(sf_tmp, 'org'))
        os.mkdir(os.path.join(sf_out, 'org'))
        load_gals.save_fits(org, os.path.join(sf_in, 'org', 'temp.fits'))
        proc = subprocess.Popen([sp_path, '-convert-FITS', os.path.join(sf_in, 'org'), os.path.join(sf_tmp, 'org'), os.path.join(sf_out, 'org'), '-generateFitQuality', '0', '-writeBulgeMask', '1']) 

        # run each cycled image in parallel
        procs = [subprocess.Popen([sp_path, '-convert-FITS', os.path.join(sf_in, str(i)), os.path.join(sf_tmp, str(i)), os.path.join(sf_out, str(i)), '-generateFitQuality', '0', '-writeBulgeMask', '1']) for i in range(irange)]
        for p in procs:
            p.wait()

        proc.wait() 
        f = genfromtxt(os.path.join(sf_out, 'org', 'galaxy.tsv'), skip_header = 1)
        pos = (f[20], f[21])

        #once they're done, calculate the change in position
        positions = []
        for i in range(irange):
            f = genfromtxt(os.path.join(sf_out, str(i), 'galaxy.tsv'), skip_header = 1)
            positions.append((f[20], f[21]))

        return np.array(positions)
    
    except:
        raise SpArcFiReError
    
    finally:
        # clean up all output
        try:
            shutil.rmtree(sf_in); shutil.rmtree(sf_tmp); shutil.rmtree(sf_out)
            for p in os.listdir('.'):
                if '_settings.txt' in p or p in ('SpArcFiRe-stdin.stdin.txt', 'arc2csv.log', 'galaxy_arcs.tsv'):
                    os.remove(p)
        except: pass


def test_single_shift(outdir, img, sp_path):
    """Shifts the image only once and checks the positional error"""
   
    print '\nRunning single shift tests...\n'

    outdir = os.path.abspath(outdir)
    r = np.arange(0, 1.1, 0.1)
    cycle_imgs = [] 
    org = np.copy(img)
    
    for delta in r:
        vector = np.array([delta, delta])
        print 'Running delta = {}'.format(vector)
        img = shift_gal.shift_img(img, vector, check_count = False)
        cycle_imgs.append(np.copy(img))
        img = np.copy(org) 
    
    print 'Saving graphs...'
    dist = [np.sqrt(s**2 + s**2) for s in r]
    
    ''' 
    try:
        pos = run_sparcfire(org, cycle_imgs, 10, outdir, sp_path)
        print len(r), len(pos)
        pos_diff = [np.sqrt((pos[0][0] + r[i] - pos[i][0])**2 + (pos[0][1] + r[i] - pos[i][1])**2) for i in range(len(r))]
    
        plt.figure()
        plt.plot(dist, pos_diff)
        plt.title('Difference in Predicted Galaxy Center')
        plt.ylabel('Difference in Pixels')
        plt.xlabel('Distance')
        plt.savefig(os.path.join(outdir, 'no_cycle_pos_diff.png'))
        
        plt.figure()
        fig, ax = plt.subplots()
        pos[:,0] -= pos[0][0]
        pos[:,1] -= pos[0][1]
        ax.scatter(pos[:,0], pos[:,1], s = 4)
        for i, lbl in enumerate(r):
            ax.annotate(round(lbl, 2), (pos[i][0], pos[i][1]))
    
        plt.title('Galaxy Center Positions')
        plt.savefig(os.path.join(outdir, 'single_shift_gal_diff.png'))
    except:
        print 'SpArcFiRe error encoutnered, continuing...'

    '''

    org_stars, star_diffs = [], []
    p = os.path.join(outdir, 'temp.fits')
    load_gals.save_fits(org, p)
    
    for s in load_gals.get_sextractor_points(p):
        if s.class_prob > class_prob:
            try: org_stars.append(find_center.estimate_center(org, s))
            except: pass

    for i in range(len(r)):
        load_gals.save_fits(cycle_imgs[i], p)
        stars = []
        for s in load_gals.get_sextractor_points(p):
            if s.class_prob > class_prob:
                try: stars.append(find_center.estimate_center(cycle_imgs[i], s))
                except: pass
        src, trg = shift_gal.find_like_points(org_stars, stars)
        total_dist = sum([np.sqrt((s.x + r[i] - t.x)**2 + (s.y + r[i]- t.y)**2) for s, t in zip(src, trg)])
        star_diffs.append(np.nan if len(src) == 0 else total_dist / len(src))
    os.remove(p)

    plt.figure()
    plt.plot(dist, star_diffs)
    plt.title('Mean Difference in Star Location')
    plt.xlabel('Shift Distance in Pixels')
    plt.ylabel('Mean Location Error')
    plt.savefig(os.path.join(outdir, 'single_shift_star_diff.png'))
    
    return dist, star_diffs


def test_double_shift(outdir, img, sp_path):
    """Shifts the galaxy back and forth and checks flux and positional error"""
        
    print '\nRunning double shift tests...\n'

    outdir = os.path.abspath(outdir)
    r = np.arange(0, 1.1, 0.1)
    cycle_imgs, mean_diff, mean_diff_gal, mean_diff_bg = [], [], [], []
    org = np.copy(img)
    
    seg_img = load_gals.get_seg_img(org)
    gal_val = seg_img[int(seg_img.shape[0] / 2.0), int(seg_img.shape[1] / 2.0)]
    
    for delta in r:
        vector = np.array([delta, delta])
        print 'Running delta = {}'.format(vector)
        img = shift_gal.shift_img(img, vector, check_count = False)
        img = shift_gal.shift_img(img, vector * -1, check_count = False)
        
        diff = np.abs(org - img)
        mean_diff.append(np.mean(diff))
        mean_diff_gal.append(np.mean(diff[seg_img == gal_val]))
        mean_diff_bg.append(np.mean(diff[seg_img == 0]))

        if delta == 0.5:
            diff = img - org
            load_gals.save_fits(img, os.path.join(outdir, 'shifted_{}.fits'.format(vector)))
            load_gals.save_fits(diff, os.path.join(outdir, 'residual_{}.fits'.format(vector)))
           
            print '\n---Range in Flux of Original, Doubly-Shifted, and Residual Images---'
            
            print 'Original:                     ({}, {})'.format(np.min(org), np.max(org))
            print 'Original (Galaxy Pixels):     ({}, {})'.format(np.min(org[seg_img == gal_val]), np.max(org[seg_img == gal_val]))
            print 'Original (Background Pixels): ({}, {})\n'.format(np.min(org[seg_img == 0]), np.max(org[seg_img == 0]))
 
            print 'Shifted:                      ({}, {})'.format(np.min(img), np.max(img))
            print 'Shifted (Galaxy Pixels):      ({}, {})'.format(np.min(img[seg_img == gal_val]), np.max(img[seg_img == gal_val]))
            print 'Shifted (Background Pixels):  ({}, {})\n'.format(np.min(img[seg_img == 0]), np.max(img[seg_img == 0]))

            print 'Residual:                     ({}, {})'.format(np.min(diff), np.max(diff))
            print 'Residual (Galaxy Pixels):     ({}, {})'.format(np.min(diff[seg_img == gal_val]), np.max(diff[seg_img == gal_val])) 
            print 'Residual (Background Pixels): ({}, {})\n'.format(np.min(diff[seg_img == 0]), np.max(diff[seg_img == 0]))

        cycle_imgs.append(np.copy(img))
        img = np.copy(org)

    print 'Saving graphs...'

    dist = [np.sqrt(s**2 + s**2) for s in r]
    
    plt.figure()
    plt.plot(dist, mean_diff)
    plt.title('Mean Absolute Flux Error')
    plt.xlabel('Shift Distance in Pixels')
    plt.ylabel('Mean Flux Error')
    plt.savefig(os.path.join(outdir, 'cycle_flux_diff.png'))
    
    plt.figure()
    plt.plot(dist, mean_diff_gal)
    plt.title('Mean Absolute Flux Error (Galaxy Only)')
    plt.xlabel('Shift Distance in Pixels')
    plt.ylabel('Mean Flux Error')
    plt.savefig(os.path.join(outdir, 'cycle_flux_diff_galaxy.png'))
    
    plt.figure()
    plt.plot(dist, mean_diff_bg)
    plt.title('Mean Absolute Flux Error (Background Only)')
    plt.xlabel('Shift Distance in Pixels')
    plt.ylabel('Mean Flux Error')
    plt.savefig(os.path.join(outdir, 'cycle_flux_diff_background.png'))
 
    for i in range(len(cycle_imgs)):
        plt.figure()
        plt.scatter(org.flatten(), np.abs(org - cycle_imgs[i]).flatten(), s = 2)
        plt.title('Initial Flux Value vs Resulting Error ({}, {})'.format(r[i], r[i]))
        plt.xlabel('Initial Flux')
        plt.ylabel('Absolute Error in Flux')
        plt.ylim(-1, 12)
        plt.savefig(os.path.join(outdir, 'cycle_flux_pixel_error_{}.png'.format(r[i])))
        
        plt.figure()
        cutoff = 5
        plt.scatter(org[org > cutoff].flatten(), 100 * np.abs(np.abs(org - cycle_imgs[i]) / org)[org > cutoff].flatten(), s = 2)
        plt.title('Initial Flux Value vs Resulting Error ({}, {})'.format(r[i], r[i]))
        plt.xlabel('Initial Flux')
        plt.ylabel('Percent Error in Flux')
        plt.ylim(-3, 60)
        plt.savefig(os.path.join(outdir, 'cycle_flux_pixel_error_{}.png'.format(r[i])))
    '''
    try:
        pos = run_sparcfire(org, cycle_imgs, 10, outdir, sp_path)
        pos_diff = [np.sqrt((pos[0][0] - pos[i][0])**2 + (pos[0][1] - pos[i][1])**2) for i in range(len(pos))]
        plt.figure()
        plt.plot(dist, pos_diff)
        plt.title('Difference in Predicted Galaxy Center')
        plt.ylabel('Difference in Pixels')
        plt.xlabel('Distance')
        plt.savefig(os.path.join(outdir, 'linear_pos_diff.png'))

        plt.figure()
        fig, ax = plt.subplots()
        pos[:,0] -= pos[0][0]
        pos[:,1] -= pos[0][1]
        ax.scatter(pos[:,0], pos[:,1], s = 4)
        for i, lbl in enumerate(r):
            ax.annotate(round(lbl, 2), (pos[i][0], pos[i][1]))
    
        plt.title('Galaxy Center Positions')
        plt.savefig(os.path.join(outdir, 'linear_pos.png'))
    except:
        print 'SpArcFiRe error encoutnered, continuing...'
    '''
    
    org_stars, star_diffs = [], []
    p = os.path.join(outdir, 'temp.fits')
    load_gals.save_fits(org, p)
    
    for s in load_gals.get_sextractor_points(p):
        if s.class_prob > class_prob:
            try: org_stars.append(find_center.estimate_center(org, s))
            except: pass

    for i in range(len(r)):
        load_gals.save_fits(cycle_imgs[i], p)
        stars = []
        for s in load_gals.get_sextractor_points(p):
            if s.class_prob > class_prob:
                try: stars.append(find_center.estimate_center(cycle_imgs[i], s))
                except: pass
        src, trg = shift_gal.find_like_points(org_stars, stars)
        total_dist = sum([np.sqrt((i.x - j.x)**2 + (i.y - j.y)**2) for i, j in zip(src, trg)])
        star_diffs.append(total_dist / len(src))
    os.remove(p)

    plt.figure()
    plt.plot(dist, star_diffs)
    plt.title('Mean Difference in Star Location')
    plt.xlabel('Shift Distance in Pixels')
    plt.ylabel('Mean Location Error')
    plt.savefig(os.path.join(outdir, 'cycle_star_diff.png'), bbox_inches = 'tight')
    
    return dist, star_diffs


def test_random_shifts(outdir, img, cycles, sp_path):
    """Shifts the galaxy randomly back and forth *cycles* times and checks flux and positional error"""
        
    print '\nRunning random shift tests...\n'

    org = np.copy(img)
    mean_diff, mean_diff_gal, mean_diff_bg, cycle_imgs = [], [], [], []
    seg_img = load_gals.get_seg_img(org)
    gal_val = seg_img[int(seg_img.shape[0] / 2.0), int(seg_img.shape[1] / 2.0)]
    
    for i in range(cycles):
        vector = np.array([-1.0, -1.0]) + 2 * np.random.random(2)
        print 'Running shift {}, cycle {}'.format(vector, i + 1)
        img = shift_gal.shift_img(img, vector)
        img = shift_gal.shift_img(img, vector * -1)
        diff = np.abs(org - img)
        mean_diff.append(np.mean(diff))
        mean_diff_gal.append(np.mean(diff[seg_img == gal_val]))
        mean_diff_bg.append(np.mean(diff[seg_img == 0]))
        cycle_imgs.append(np.copy(img))

    print 'Saving graphs...'
    plt.figure()
    plt.plot(np.arange(1, cycles + 1), mean_diff)
    plt.xlim(0, cycles + 1)
    plt.title('Mean Absolute Flux Error')
    plt.xlabel('Cycle')
    plt.ylabel('Mean Flux Error')
    plt.savefig(os.path.join(outdir, 'random_mean_diff.png'))
    
    plt.figure()
    plt.plot(np.arange(1, cycles + 1), mean_diff_gal)
    plt.xlim(0, cycles + 1)
    plt.title('Mean Absolute Flux Error (Galaxy Only)')
    plt.xlabel('Cycle')
    plt.ylabel('Mean Flux Error')
    plt.savefig(os.path.join(outdir, 'random_mean_diff_galaxy.png'))
    
    plt.figure()
    plt.plot(np.arange(1, cycles + 1), mean_diff_bg)
    plt.xlim(0, cycles + 1)
    plt.title('Mean Absolute Flux Error (Background Only)')
    plt.xlabel('Cycle')
    plt.ylabel('Mean Flux Error')
    plt.savefig(os.path.join(outdir, 'random_mean_diff_background.png'))
    
    '''
    try:
        pos = run_sparcfire(org, cycle_imgs, cycles, outdir, sp_path)
        pos_diff = [np.sqrt((pos[0][0] - pos[i][0])**2 + (pos[0][1] - pos[i][1])**2) for i in range(len(pos))]
        plt.figure()
        plt.plot(np.arange(1, cycles + 1), pos_diff)
        plt.xlim(0, cycles + 1)
        plt.title('Difference in Predicted Galaxy Center')
        plt.ylabel('Difference in Pixels')
        plt.xlabel('Cycle')
        plt.savefig(os.path.join(outdir, 'pos_diff.png'))

        plt.figure()
        fig, ax = plt.subplots()
        pos[:,0] -= pos[0][0]
        pos[:,1] -= pos[0][1]
        ax.scatter(pos[:,0], pos[:,1], s = 4)
        for i, lbl in enumerate(range(len(pos))):
            ax.annotate(lbl, (pos[i][0], pos[i][1]))
        plt.title('Galaxy Center Positions')
        plt.savefig(os.path.join(outdir, 'pos.png'))

    except:
        print 'SpArcFiRe error encountered, continuing...'

    '''
    star_diffs = []
    p = os.path.join(outdir, 'temp.fits')
    load_gals.save_fits(org, p)
    org_stars = [s for s in load_gals.get_sextractor_points(p) if s.class_prob > 0.09]
    os.remove(p)
    
    for i in range(cycles):
        load_gals.save_fits(cycle_imgs[i], p)
        stars = [s for s in load_gals.get_sextractor_points(p) if s.class_prob > 0.09]
        src, trg = shift_gal.find_like_points(org_stars, stars)
        total_dist = sum([np.sqrt((i.x - j.x)**2 + (i.y - j.y)**2) for i, j in zip(src, trg)])
        star_diffs.append(total_dist / len(src))
        os.remove(p)

    plt.figure()
    plt.plot(np.arange(1, cycles + 1), star_diffs)
    plt.xlim(0, cycles + 1)
    plt.title('Mean Difference in Star Location')
    plt.xlabel('Cycle')
    plt.ylabel('Mean Location Error')
    plt.savefig(os.path.join(outdir, 'random_star_diff.png'))


def test_upscale_save_data(outdir, data_file, img, name):

    vecs = [l.rstrip().split(' ') for l in open('shifts.txt').readlines()]
    vecs = [np.array((float(s[0]), float(s[1]))) for s in vecs]
    data = name + '\t'

    for vec in vecs:    
        for i in (100, 200, 400, 600, 800, 1000):
            print 'Running vec {}, upscale factor {}'.format(vec, i)
            img_cpy = np.copy(img)
            img_cpy = shift_gal.shift_img(img_cpy, vec, i, check_count = False)
            img_cpy = shift_gal.shift_img(img_cpy, vec * -1, i, check_count = False)
            data += str(np.mean(np.abs(img - img_cpy))) + '\t'

    data_file = open(data_file, 'a')
    data_file.write(data + '\n')
    data_file.close()


def plot_upscale_data(outdir, data_path):

    class GalData:
        def __init__(self, line, name):
            self.name = name
            self.vec_errs = [line[i * 6 + 1: 6 * (i + 1) + 1] for i in range(10)]
    
    vecs = [l.rstrip().split(' ') for l in open('shifts.txt').readlines()]
    vecs = [np.array((float(s[0]), float(s[1]))) for s in vecs]

    data = genfromtxt(data_path, skip_header = 1)
    data_str = genfromtxt(data_path, skip_header = 1, dtype = str)
    galdata = [GalData(data[i], data_str[i][0]) for i in range(len(data))]
    
    x = (100, 200, 400, 600, 800, 1000)
    
    # plot with all galaxies all shifts
    plt.figure()
    for gd in galdata:
        for ve in gd.vec_errs:
            plt.plot(x, ve)

    plt.xlabel('Upscale Factor')
    plt.ylabel('Flux Error (Nanomaggies)')
    plt.title('Mean Absolute Flux Error vs Upscale Factor\nAll Galaxies - All Shifts')
    plt.savefig(os.path.join(outdir, 'flux_err_all.png'))

    # plot for each galaxy, all shifts
    for gd in galdata:
        plt.figure()
        for i in range(len(gd.vec_errs)):
            plt.plot(x, gd.vec_errs[i], label = '{} | {:.2f}'.format(vecs[i], np.sqrt(vecs[i][0]**2 + vecs[i][1]**2)))
        
        plt.legend(title = 'Shift Vector', bbox_to_anchor = (1.05, 1), loc = 'upper left')
        plt.xlabel('Upscale Factor')
        plt.ylabel('Flux Error (Nanomaggies)')
        plt.title('Mean Absolute Flux Error vs Upscale Factor\n{} - All Shifts'.format(gd.name))
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'flux_err_{}_all_shifts.png'.format(gd.name)))


    # plot for each shift, all galaxies
    for i in range(10):
        plt.figure()
        for gd in galdata:
            plt.plot(x, gd.vec_errs[i])

        plt.xlabel('Upscale Factor')
        plt.ylabel('Flux Error (Nanomaggies)')
        plt.title('Mean Absolute Flux Error vs Upscale Factor\nAll Galaxies - Vector: {}'.format(vecs[i]))
        plt.savefig(os.path.join(outdir, 'flux_err_all_galaxies_{}.png'.format(vecs[i])))


def test_upscale_factor(outdir, img):
     
    for vec in np.random.random((20, 2)):
        print 'Testing shift vector of {}'.format(vec)
        upscale_imgs = []
        upscales = (50, 100, 200, 300, 400)
        
        for i in upscales:
            print 'Running upscale factor {}'.format(i)
            img_cpy_str = np.copy(img)
            img_cpy_str = shift_gal.shift_img(img_cpy_str, vec, i, check_count = False)
            upscale_imgs.append(np.copy(img_cpy_str))
        
        org_stars, star_diffs = [], []
        p = os.path.join(outdir, 'temp.fits')
        load_gals.save_fits(img, p)
        for s in load_gals.get_sextractor_points(p):
            if s.class_prob > class_prob:
                try: org_stars.append(find_center.estimate_center(img, s))
                except: pass
        os.remove(p)
        for up_img in upscale_imgs:
            load_gals.save_fits(up_img, p)
            stars = []
            for s in load_gals.get_sextractor_points(p):
                if s.class_prob > class_prob:
                    try: stars.append(find_center.estimate_center(up_img, s))
                    except: pass
            
            src, trg = shift_gal.find_like_points(org_stars, stars)
            total_dist = sum([np.sqrt((s.x + vec[0] - t.x)**2 + (s.y + vec[1] - t.y)**2) for s, t in zip(src, trg)])
            star_diffs.append(total_dist / len(src))
            os.remove(p)

        plt.figure()
        plt.plot(upscales, star_diffs)
        plt.title('Mean Difference in Star Location {}'.format(vec))
        plt.xlabel('Upscale Factor')
        plt.ylabel('Mean Location Error')
        plt.savefig(os.path.join(outdir, 'upscale_single_shift_star_diff_{}.png'.format(vec)))


def test_fpack_compression(outdir, img, name):
   
    path = os.path.join(outdir, 'temp.fits')
    perc_diff, mean_diff, compression = [], [], []
    values = np.arange(2, 102, 2)
    for i in values:
        load_gals.save_fits(img, path)
        uncompressed_size = os.path.getsize(path)
        subprocess.Popen(['/home/wayne/bin/bin.x86_64/fpack', '-q', str(i), '-D', '-Y', path]).wait()
        compressed_size = os.path.getsize(path + '.fz')
        subprocess.Popen(['/home/wayne/bin/bin.x86_64/funpack', path + '.fz']).wait()
        compressed_img = load_gals.load_fits(path)
        perc_diff.append(np.sum(np.abs(img - compressed_img)) / np.sum(img) * 100)
        mean_diff.append(np.mean(np.abs(img - compressed_img)))
        compression.append(compressed_size / float(uncompressed_size))
        os.remove(path + '.fz')

    os.remove(path)
       
    plt.figure()
    plt.plot(values, perc_diff)
    plt.title('FPack Compression Level vs. Abs Sum Difference in Flux')
    plt.xlabel('Compression Level')
    plt.ylabel('Total Flux Error (Percent)')
    plt.savefig(os.path.join(outdir, 'compression_sum.png'))

    plt.figure()
    plt.plot(values, mean_diff)
    plt.title('FPack Compression Level vs. Mean Abs Difference in Flux')
    plt.xlabel('Compression Level')
    plt.ylabel('Flux Error (Nanomaggies)')
    plt.savefig(os.path.join(outdir, 'compression_mean.png'))

    plt.figure()
    plt.plot(values, compression)
    plt.title('FPack Compression Level vs. Compression Percent')
    plt.xlabel('Compression Level')
    plt.ylabel('Compressed File Size (Percent of Original)')
    plt.savefig(os.path.join(outdir, 'compression.png'))


def test_shifts(outdir, cropped_img, img, name, random_cycles, sp_path):
    """Runs various tests measureing flux and positional error"""

    outdir = os.path.abspath(outdir)
    
    try:
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.mkdir(outdir)
    except:
        print 'Error creating {}, testing not ran.'.format(outdir)
        return
    
    #plot_upscale_data(outdir, '/extra/wayne1/preserve/antholn1/ShiftGal/dataCpy.csv')

    #if cropped_img.shape[0] < 107:
    #    test_upscale_save_data(outdir, '/extra/wayne1/preserve/antholn1/ShiftGal/data.csv', cropped_img, name)
    #test_upscale_factor(outdir, cropped_img, img)
    
    test_fpack_compression(outdir, img, name)

    '''
    test_random_shifts(outdir, np.copy(img), random_cycles, sp_path)
    test_random_shifts(outdir, np.copy(img), random_cycles, sp_path)
    x, y = test_double_shift(outdir, np.copy(img), sp_path)
    x2, y2 = test_single_shift(outdir, np.copy(img), sp_path)
    
    plt.figure()
    plt.plot(x, y, label = 'Double-Shift')
    plt.plot(x2, y2, label = 'Single-Shift')
    plt.legend()
    plt.title('Mean Difference in Star Location')
    plt.xlabel('Shift Distance in Pixels')
    plt.ylabel('Mean Location Error')
    plt.savefig(os.path.join(outdir, 'star_pos_both.png'))
    '''
