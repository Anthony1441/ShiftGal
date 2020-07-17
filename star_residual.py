import numpy as np

def calc_star_residual(s1, s2):
    """Returns a normalized array of the residual between the two stars."""
    s1, s2  = s1 - np.min(s1), s2 - np.min(s2)
    s1, s2 = s1 / np.max(s1), s2 / np.max(s2)
    return mp.abs(s1 - s2)



