import numpy as np
import find_center
from astropy.io import fits
from astropy import wcs
import shift_gal
import load_gals
import os
import glob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys

def load_gal_and_stars(path):
    gal = fits.open(path, ignore_missing_end = True)
    stars = load_gals.get_sextractor_points(path, 0.7)
    return gal, stars


def num_stars(path):
    return len(load_gal_and_stars(path)[1][0])


def test_shift(path, vector):
    
    gal, stars = load_gal_and_stars(path)
    img = gal[0].data
    # add a border to the image and load the points
    pad_amt = int(len(img) * 0.1)
    img = np.pad(img, pad_amt, 'constant')
    stars[0] += pad_amt
    stars[1] += pad_amt

    shift_img = shift_gal.shift_img(img, vector)
    shift_stars = np.copy(stars)
    shift_stars[0] += int(vector[0] + 0.5)
    shift_stars[1] += int(vector[1] + 0.5)

    # estimate the points for both images
    gal_center_x, gal_center_y = [], []
    
    for i in range(len(stars[0])):
        try:
            sx, sy = find_center.estimate_center(img, (stars[0][i], stars[1][i]))
            gal_center_x.append(sx)
            gal_center_y.append(sy)
            #print '{} -> {}'.format((stars[0][i], stars[1][i]), (sx, sy))
        
        except:
            pass
     
    shiftgal_center_x, shiftgal_center_y = [], []
    
    for i in range(len(shift_stars[0])):
        try:
            sx, sy = find_center.estimate_center(shift_img, (shift_stars[0][i], shift_stars[1][i]))
            shiftgal_center_x.append(sx)
            shiftgal_center_y.append(sy)
            #print '{} -> {}'.format((shift_stars[0][i], shift_stars[1][i]), (sx, sy))

        except:
            pass

    x1, y1, x2, y2 = shift_gal.find_like_points(gal_center_x, gal_center_y, shiftgal_center_x, shiftgal_center_y)
    vector = np.array(vector)
    est_vec = np.array(shift_gal.average_vector(x1, y1, x2, y2)) * -1
    #print 'Actual vector {}, estimated vector {}, difference {}'.format(vector, est_vec, vector - est_vec)
    return np.abs(vector - est_vec)




'''
sum_diff = np.array([0.0, 0.0])

for i in np.arange(0, 1.01, 0.01):
    sum_diff += test_shift('./galaxies/diff_crops_test/c0.fits', (i, 0))

print 'Result:', sum_diff / len(np.arange(0, 1.01, 0.01))


for i in np.arange(0, 1.1, 0.1):
    test_shift('./galaxies/diff_crops_test/c0.fits', (0, i))
'''

names = []
nums = []
dists = []
r = np.arange(0, 5.5, 0.5)

for gal in os.listdir('./galaxies/spinparitygals/g'):
    diff = np.array([0.0, 0.0])
    for x in r:
        for y in r:
            diff += test_shift('./galaxies/spinparitygals/g/' + gal, (x, y)) 
        
    res = diff / (len(r) * len(r))
    dist = np.sqrt(res[0]**2 + res[1]**2)
    num = num_stars('./galaxies/spinparitygals/g/' + gal)
    
    
    print gal, num, dist
    sys.stderr.write('{} {} {}'.format(gal, str(num), str(dist)))
    names.append(gal)
    nums.append(num)
    dists.append(dist)

plt.scatter(nums, dists)
plt.savefig('plot2.png')


for path in glob.glob('core.*'):
    try:
        os.remove(path)
    except:
        pass




