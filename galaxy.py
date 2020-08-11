from collections import OrderedDict
import numpy as np
from copy import copy

class InvalidGalColorError(Exception):
    pass


class GalsAndStarsDoNotContainTheSameWavebandsError(Exception):
    pass


class Galaxy:

    def __init__(self, gal_dict, stars_dict, star_class_perc, name):
        # It is exptected that gal_dict and stars_dict contain the same
        # wavebands in the same order.

        if gal_dict.keys() != stars_dict.keys():
            raise GalsAndStarsDoNotContainTheSameWavebandsError
        
        self.gal_dict = gal_dict
        self.all_stars_dict = copy(stars_dict)
        # keep a separate list of those that are likely stars
        self.stars_dict = OrderedDict()
        for color in stars_dict.keys():
            self.stars_dict.update({color: [s for s in stars_dict[color] if s.class_prob >= star_class_perc]})

        self.name = name

    def images(self, color = None):
        if color is None:
            return [(color, img[0].data) for color, img in self.gal_dict.iteritems()]

        elif color in self.gal_dict.keys():
            return self.gal_dict[color][0].data
        
        raise InvalidGalColorError


    def stars(self, color = None): 
        if color is None:
            return self.stars_dict.values()
        
        elif color in self.stars_dict.keys():
            return self.stars_dict[color]

        raise InvalidGalColorError


    def gen_img_star_pairs(self):
        for color, gal, stars in zip(self.gal_dict.keys(), self.gal_dict.values(), self.stars_dict.values()):
            yield (color, gal[0].data, stars)   


    def colors(self):
        return self.gal_dict.keys()

    
    def add_borders(self, b_size):
        for img in self.gal_dict.values():
            img[0].data = np.pad(img[0].data, b_size, 'constant')
            img[0].header['CRPIX1'] += b_size
            img[0].header['CRPIX2'] += b_size

class Star:

    def __init__(self, x, y, class_prob = None, weight = None, x_spread = None, y_spread = None):
        self.x = x
        self.y = y
        self.class_prob = class_prob
        self.weight = weight
        self.x_spread = x_spread
        self.y_spread = y_spread

    def __str__(self):
        return '({}, {})'.format(self.x, self.y)

    def __sub__(self, star):
        return np.array((self.x - star.x, self.y - star.y))
