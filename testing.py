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


def run_sparcfire(org, img_data, irange, outdir, sp_path):
    """Runs SpArcFiRe on the org and ima_data, then returns the difference in galaxy center for org and img_data"""
    
    try:
        sf_in = os.path.join(outdir, 'sf_in')
        sf_tmp = os.path.join(outdir, 'sf_tmp')
        sf_out = os.path.join(outdir, 'sf_out')

        os.mkdir(sf_in)
        os.mkdir(sf_tmp)
        os.mkdir(sf_out)
        
        for i in range(irange):
            os.mkdir(os.path.join(sf_in, str(i)))
            os.mkdir(os.path.join(sf_tmp, str(i)))
            os.mkdir(os.path.join(sf_out, str(i)))
            fits.HDUList([fits.PrimaryHDU(data = img_data[i])]).writeto(os.path.join(sf_in, str(i), 'temp.fits'))
       
        # get starting image center
        os.mkdir(os.path.join(sf_in, 'org'))
        os.mkdir(os.path.join(sf_tmp, 'org'))
        os.mkdir(os.path.join(sf_out, 'org'))
        fits.HDUList([fits.PrimaryHDU(data = org)]).writeto(os.path.join(sf_in, 'org', 'temp.fits'))
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
            shutil.rmtree(sf_in)
            shutil.rmtree(sf_tmp)
            shutil.rmtree(sf_out)
            for p in os.listdir('.'):
                if '_settings.txt' in p or p in ('SpArcFiRe-stdin.stdin.txt', 'arc2csv.log', 'galaxy_arcs.tsv'):
                    os.remove(p)
        except: pass


def test_smearing_no_cycles(outdir, img, shift_method, vcells, cycles, sp_path, range_const):
    """Shifts the galaxy only once and checks the error"""
       
    outdir = os.path.abspath(outdir)

    r = np.arange(0, 1.0 + (1.0 / cycles), (1.0 / cycles))

    cycle_imgs = [] 
    org, seg_org = np.copy(img), np.copy(img)
    
    # run sextractor on image to mask out stars
    fits.HDUList([fits.PrimaryHDU(data = org)]).writeto(os.path.join(outdir, 'seg_temp.fits'))   
    proc = subprocess.Popen(['./sex', os.path.join(outdir, 'seg_temp.fits'), '-CHECKIMAGE_TYPE', 'SEGMENTATION', '-CHECKIMAGE_NAME', os.path.join(outdir, 'seg_out.fits'), '-CATALOG_NAME', os.path.join(outdir, 'temp.txt')], stderr = subprocess.STDOUT)
    if proc.wait() != 0: raise load_gals.SextractorError
    seg = fits.open(os.path.join(outdir, 'seg_out.fits'), ignore_missing_end = True)
    seg_img = seg[0].data
    seg.close()
    os.remove(os.path.join(outdir, 'seg_out.fits'))
    os.remove(os.path.join(outdir, 'seg_temp.fits'))
    os.remove(os.path.join(outdir, 'temp.txt'))
    
    gal_val = seg_img[int(seg_img.shape[0] / 2.0), int(seg_img.shape[1] / 2.0)]
    seg_org[(seg_img != gal_val)] = 0
    seg_img = np.copy(seg_org)
    cycle_imgs = []
    
    
    for s in r:
        vector = np.array([s, s])
        print 'Running {}'.format(vector)
        img = shift_gal.shift_img(img, vector, shift_method, vcells, range_const = range_const, check_count = False)[0]
        seg_img = shift_gal.shift_img(seg_img, vector, shift_method, vcells, range_const = range_const, check_count = False)[0]
        cycle_imgs.append(np.copy(img))
        img, seg_img = np.copy(org), np.copy(seg_org)

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
        plt.savefig(os.path.join(outdir, 'linear_pos.png'))
    except:
        print 'SpArcFiRe error encoutnered, continuing...'

    '''

    star_diffs = []
    p = os.path.join(outdir, 'temp.fits')
    fits.HDUList([fits.PrimaryHDU(data = org)]).writeto(p)
    org_stars = []
    for s in load_gals.get_sextractor_points(p):
        if s.class_prob > 0.09:
            try: org_stars.append(find_center.estimate_center(org, s))
            except: pass
    os.remove(p)

    for i in range(len(r)):
        fits.HDUList([fits.PrimaryHDU(data = cycle_imgs[i])]).writeto(p)    
        stars = []
        for s in load_gals.get_sextractor_points(p):
            if s.class_prob > 0.09:
                try: stars.append(find_center.estimate_center(cycle_imgs[i], s))
                except: pass

        src, trg = shift_gal.find_like_points(org_stars, stars)
        total_dist = sum([np.sqrt((s.x + r[i] - t.x)**2 + (s.y + r[i]- t.y)**2) for s, t in zip(src, trg)])
        print len(src), total_dist
        star_diffs.append(total_dist / len(src))
        os.remove(p)

    plt.figure()
    plt.plot(dist, star_diffs)
    plt.title('Average Difference in Star Position')
    plt.xlabel('Distance')
    plt.ylabel('Difference in Pixels')
    plt.savefig(os.path.join(outdir, 'no_cycle_star_diff_{}.png'.format(range_const)))
    
   

def test_smearing_linear(outdir, img, shift_method, vcells, cycles, sp_path):
    """Shifts the galaxy back and forth randomly, running it through SpArcFiRe
       each cycle and recorind the galaxy position"""
       
    outdir = os.path.abspath(outdir)

    r = np.arange(0, 1.0 + (1.0 / cycles), (1.0 / cycles))

    sum_diff = []
    org = np.copy(img)
    cycle_imgs = []
 
    sum_diff, seg_sum_diff = [], []
    org, seg_org = np.copy(img), np.copy(img)
    
    # run sextractor on image to mask out stars
    fits.HDUList([fits.PrimaryHDU(data = org)]).writeto(os.path.join(outdir, 'seg_temp.fits'))   
    proc = subprocess.Popen(['./sex', os.path.join(outdir, 'seg_temp.fits'), '-CHECKIMAGE_TYPE', 'SEGMENTATION', '-CHECKIMAGE_NAME', os.path.join(outdir, 'seg_out.fits'), '-CATALOG_NAME', os.path.join(outdir, 'temp.txt')], stderr = subprocess.STDOUT)
    if proc.wait() != 0: raise load_gals.SextractorError
    seg = fits.open(os.path.join(outdir, 'seg_out.fits'), ignore_missing_end = True)
    seg_img = seg[0].data
    seg.close()
    os.remove(os.path.join(outdir, 'seg_out.fits'))
    os.remove(os.path.join(outdir, 'seg_temp.fits'))
    os.remove(os.path.join(outdir, 'temp.txt'))
    
    gal_val = seg_img[int(seg_img.shape[0] / 2.0), int(seg_img.shape[1] / 2.0)]
    seg_cpy = np.copy(seg_img)
    seg_org[(seg_img != gal_val)] = 0
    seg_img = np.copy(seg_org)
    cycle_imgs = []
    
    
    for s in r:
        vector = np.array([s, s])
        print 'Running {}'.format(vector)
        img = shift_gal.shift_img(img, vector, shift_method, vcells, check_count = False)[0]
        img = shift_gal.shift_img(img, vector * -1, shift_method, vcells, check_count = False)[0]
        seg_img = shift_gal.shift_img(seg_img, vector, shift_method, vcells, check_count = False)[0]
        seg_img = shift_gal.shift_img(seg_img, vector * -1, shift_method, vcells, check_count = False)[0]
        sum_diff.append(100 * np.sum(np.abs(org - img)) / np.sum(org))
        seg_sum_diff.append(np.abs(100 * np.sum(np.abs(seg_org - seg_img)) / np.sum(seg_org)))
        if s == 0.5:
            diff = img - org
            fits.HDUList([fits.PrimaryHDU(data = img)]).writeto(os.path.join(outdir, 'shifted_{}.fits'.format(vector)))
            fits.HDUList([fits.PrimaryHDU(data = diff)]).writeto(os.path.join(outdir, 'residual_{}.fits'.format(vector)))
            
            print 'Min of res: {}, Max of res: {}'.format(np.min(diff), np.max(diff))
            print 'Min of shifted: {}, Max of shifted {}'.format(np.min(img), np.max(img))
            print 'Min of org: {}, Max of org {}'.format(np.min(org), np.max(org))

            print 'Min of res galaxy: {}, Max of res galaxy: {}'.format(np.min(diff[seg_cpy == gal_val]), np.max(diff[seg_cpy == gal_val]))
            print 'Min of shifted galaxy: {}, Max of shifted galaxy: {}'.format(np.min(img[seg_cpy == gal_val]), np.max(img[seg_cpy == gal_val]))
            print 'Min of org galaxy: {}, Max of org galaxy: {}'.format(np.min(org[seg_cpy == gal_val]), np.max(org[seg_cpy == gal_val]))

            print 'Min of res bg: {}, Max of res bg: {}'.format(np.min(diff[seg_cpy == 0]), np.max(diff[seg_cpy == 0]))
            print 'Min of shifted bg: {}, Max of shifted bg: {}'.format(np.min(img[seg_cpy == 0]), np.max(img[seg_cpy == 0]))
            print 'Min or org bg: {}, Max of org bg: {}'.format(np.min(org[seg_cpy == 0]), np.max(org[seg_cpy == 0]))

        cycle_imgs.append(np.copy(img))
        img, seg_img = np.copy(org), np.copy(seg_org)

    dist = [np.sqrt(s**2 + s**2) for s in r]
    
    plt.figure()
    plt.plot(dist, sum_diff)
    plt.title('Difference in Brightness as a Percent of Total Original Count')
    plt.xlabel('Distance')
    plt.ylabel('Percent')
    plt.savefig(os.path.join(outdir, 'linear_sum_diff.png'))
    
    plt.figure()
    plt.plot(dist, seg_sum_diff)
    plt.title('Difference in Brightness as a Percent of Total Original Count (Galaxy Only)')
    plt.xlabel('Distance')
    plt.ylabel('Percent')
    plt.savefig(os.path.join(outdir, 'linear_galaxy_sum_diff.png'))
    
    for i in range(len(cycle_imgs)):
        plt.figure()
        plt.scatter(org.flatten(), np.abs(org - cycle_imgs[i]).flatten(), s = 2)
        plt.title('Initial Pixel Brightness vs Resulting Error ({}, {})'.format(r[i], r[i]))
        plt.xlabel('Initial Brightness')
        plt.ylabel('Difference in Brightness')
        plt.ylim(-5, 150)
        plt.savefig(os.path.join(outdir, 'linear_brightness_error_{}.png'.format(r[i])))
        
        plt.figure()
        cutoff = 5
        plt.scatter(org[org > cutoff].flatten(), 100 * np.abs(np.abs(org - cycle_imgs[i]) / org)[org > cutoff].flatten(), s = 2)
        plt.title('Initial Pixel Brightness vs Resulting Error ({}, {})'.format(r[i], r[i]))
        plt.xlabel('Initial Brightness')
        plt.ylabel('Percent Difference in Brightness')
        plt.ylim(-5, 100)
        plt.savefig(os.path.join(outdir, 'linear_percent_brightness_error_{}.png'.format(r[i])))


    ''' 
    try:
        pos = run_sparcfire(org, cycle_imgs, cycles, outdir, sp_path)
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
    
    star_diffs = []
    p = os.path.join(outdir, 'temp.fits')
    fits.HDUList([fits.PrimaryHDU(data = org)]).writeto(p)
    org_stars = []
    for s in load_gals.get_sextractor_points(p):
        if s.class_prob > 0.09:
            try: org_stars.append(find_center.estimate_center(org, s))
            except: pass
    os.remove(p)

    for i in range(len(r)):
        fits.HDUList([fits.PrimaryHDU(data = cycle_imgs[i])]).writeto(p)    
        stars = []
        for s in load_gals.get_sextractor_points(p):
            if s.class_prob > 0.09:
                try: stars.append(find_center.estimate_center(cycle_imgs[i], s))
                except: pass

        src, trg = shift_gal.find_like_points(org_stars, stars)
        total_dist = sum([np.sqrt((i.x - j.x)**2 + (i.y - j.y)**2) for i, j in zip(src, trg)])
        star_diffs.append(total_dist / len(src))
        os.remove(p)

    plt.figure()
    plt.plot(dist, star_diffs)
    plt.title('Average Difference in Star Position')
    plt.xlabel('Distance')
    plt.ylabel('Difference in Pixels')
    plt.savefig(os.path.join(outdir, 'linear_star_diff.png'))
    

def test_smearing(outdir, img, shift_method, vcells, cycles, sp_path):
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
    
    org = np.copy(img)
    '''
    sum_diff, seg_sum_diff = [], []
    org, seg_org = np.copy(img), np.copy(img)
    
    # run sextractor on image to mask out stars
    fits.HDUList([fits.PrimaryHDU(data = org)]).writeto(os.path.join(outdir, 'seg_temp.fits'))   
    proc = subprocess.Popen(['./sex', os.path.join(outdir, 'seg_temp.fits'), '-CHECKIMAGE_TYPE', 'SEGMENTATION', '-CHECKIMAGE_NAME', os.path.join(outdir, 'seg_out.fits'), '-CATALOG_NAME', os.path.join(outdir, 'temp.txt')], stderr = subprocess.STDOUT)
    if proc.wait() != 0: raise load_gals.SextractorError
    seg = fits.open(os.path.join(outdir, 'seg_out.fits'), ignore_missing_end = True)
    seg_img = seg[0].data
    seg.close()
    os.remove(os.path.join(outdir, 'seg_out.fits'))
    os.remove(os.path.join(outdir, 'seg_temp.fits'))
    os.remove(os.path.join(outdir, 'temp.txt'))
    
    gal_val = seg_img[int(seg_img.shape[0] / 2.0), int(seg_img.shape[1] / 2.0)]
    seg_org[(seg_img != 0) & (seg_img != gal_val)] = 0
    seg_img = np.copy(seg_org)
    cycle_imgs = []
    
    
    for i in range(cycles):
        vector = np.array([-1.0, -1.0]) + 2 * np.random.random(2)
        print 'Running {}, cycle {}'.format(vector, i + 1)
        img = shift_gal.shift_img(img, vector, shift_method, vcells, check_count = False)[0]
        img = shift_gal.shift_img(img, vector * -1, shift_method, vcells, check_count = False)[0]
        seg_img = shift_gal.shift_img(seg_img, vector, shift_method, vcells, check_count = False)[0]
        seg_img = shift_gal.shift_img(seg_img, vector * -1, shift_method, vcells, check_count = False)[0]
        sum_diff.append(100 * np.sum(np.abs(org - img)) / np.sum(org))
        seg_sum_diff.append(100 * np.sum(np.abs(seg_org - seg_img)) / np.sum(org))
        cycle_imgs.append(np.copy(img))

    
    plt.figure()
    plt.plot(np.arange(1, cycles + 1), sum_diff)
    plt.xlim(0, cycles + 1)
    plt.title('Difference in Count as a % of Total Original Count')
    plt.xlabel('Cycle')
    plt.ylabel('Percent')
    plt.savefig(os.path.join(outdir, 'sum_diff.png'))
    
    plt.figure()
    plt.plot(np.arange(1, cycles + 1), seg_sum_diff)
    plt.xlim(0, cycles + 1)
    plt.title('Difference in Count as a % of Total Original Count (Masked)')
    plt.xlabel('Cycle')
    plt.ylabel('Percent')
    plt.savefig(os.path.join(outdir, 'masked_sum_diff.png'))
    
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

    
    star_diffs = []
    p = os.path.join(outdir, 'temp.fits')
    fits.HDUList([fits.PrimaryHDU(data = org)]).writeto(p)
    org_stars = [s for s in load_gals.get_sextractor_points(p) if s.class_prob > 0.09]
    os.remove(p)
    
    for i in range(cycles):
        fits.HDUList([fits.PrimaryHDU(data = cycle_imgs[i])]).writeto(p)    
        stars = [s for s in load_gals.get_sextractor_points(p) if s.class_prob > 0.09]
        src, trg = shift_gal.find_like_points(org_stars, stars)
        total_dist = sum([np.sqrt((i.x - j.x)**2 + (i.y - j.y)**2) for i, j in zip(src, trg)])
        star_diffs.append(total_dist / len(src))
        os.remove(p)

    plt.figure()
    plt.plot(np.arange(1, cycles + 1), star_diffs)
    plt.xlim(0, cycles + 1)
    plt.title('Average Difference in Star Position')
    plt.xlabel('Cycle')
    plt.ylabel('Difference in Pixels')
    plt.savefig(os.path.join(outdir, 'star_diff.png'))
    '''

    test_smearing_linear(outdir, org, shift_method, vcells, 10, sp_path)
    #for i in np.arange(10, 51, 5):
    test_smearing_no_cycles(outdir, org, shift_method, vcells, 10, sp_path, 15)


def test_gradient_shift_param(org_img, vcells, name):
    
    diff = []
    s_img = np.copy(org_img)
    
    s_img = shift_gal.shift_img(s_img, (0.5, 0.5), 'constant')[0]
    s_img = shift_gal.shift_img(s_img, (-0.5, -0.5), 'constant')[0]
    c = np.sum(np.abs(org_img - s_img))
    s_img = np.copy(org_img)
    
    r = np.arange(1, 10, 1) 
    for const in r:
        for _ in range(20):
            s_img = shift_gal.shift_img(s_img, (0.5, 0.5), 'gradient', vcells, range_const = const)[0]
            s_img = shift_gal.shift_img(s_img, (-0.5, -0.5), 'gradient', vcells, range_const = const)[0]
                
        print const, np.sum(np.abs(org_img - s_img))
        diff.append(np.sum(np.abs(org_img - s_img)))
        s_img = np.copy(org_img)
    
    plt.figure()
    plt.plot(r, diff)
    plt.title('Range Constant Value vs. Photon Count Difference (C was {})'.format(c))
    plt.savefig('{}_range_const.png'.format(name))


