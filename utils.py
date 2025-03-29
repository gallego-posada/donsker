import argparse

import numpy as np
import scipy.stats as stats

COLORS = {
    "blue": "SteelBlue",
    "red": "Crimson",
    "green": "ForestGreen",
    "black": "Black",
}

DEFAULT_PAUSE = 0.5


class EmpiricalCDF:
    # Inspired by Statsmodels' ECDF class
    # https://www.statsmodels.org/stable/_modules/statsmodels/distributions/empirical_distribution.html

    def __init__(self, data):
        _x = np.array(data, copy=True)
        _x.sort()

        self.n = len(_x)
        assert self.n > 0, "Data must not be empty"
        _y = np.linspace(1.0 / self.n, 1, self.n)

        # Add virtual point at -inf with CDF = 0
        self.x = np.r_[-np.inf, _x]
        self.y = np.r_[0, _y]

    def __call__(self, x):
        x_ind = np.searchsorted(self.x, x, side="right") - 1
        return self.y[x_ind]


def sample_gaussian_data(n, loc=0, scale=1):
    return stats.norm(loc=loc, scale=scale).rvs(n)


def create_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_0_arrow", action="store_true", help="Show arrow between CDFs at 0.")
    parser.add_argument("--keep_open", action="store_true", help="Keep the plot open after displaying.")
    parser.add_argument("--num_frames", type=int, default=10, help="Number of frames to display.")
    parser.add_argument("--sample_size", type=int, default=1_000, help="Sample size for empirical CDF.")
    return parser
