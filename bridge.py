import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from utils import (
    COLORS,
    DEFAULT_PAUSE,
    EmpiricalCDF,
    create_default_parser,
    sample_gaussian_data,
)

parser = create_default_parser()
args = parser.parse_args()

BASE_X = np.linspace(-3, 3, 1000)

SAMPLE_SIZE = args.sample_size

fig = plt.figure(figsize=(8, 5))

for _ in range(args.num_frames):
    plt.pause(DEFAULT_PAUSE)
    plt.clf()
    plt.cla()

    # ---------------------------------------------------------------------------------
    # Draw random sample from N(0,1), calculate empirical and theoretical CDFs
    # ---------------------------------------------------------------------------------
    data = sample_gaussian_data(n=SAMPLE_SIZE, loc=0, scale=1)
    ecdf = EmpiricalCDF(data)
    ecdf_values = ecdf(BASE_X)
    true_cdf = stats.norm.cdf(BASE_X, loc=0, scale=1)

    G = np.sqrt(ecdf.n) * (ecdf_values - true_cdf)
    plt.plot(true_cdf, G)
    plt.axhline(0, color=COLORS["black"], lw=1, linestyle="--")

    # Highlight maximum deviation
    max_abs_deviation = np.abs(G).max()
    plt.axhline(max_abs_deviation, color=COLORS["red"], lw=1, linestyle="--")
    plt.axhline(-max_abs_deviation, color=COLORS["red"], lw=1, linestyle="--")

    # Mark tie-down points
    plt.scatter(0, 0, marker="o", color=COLORS["red"])
    plt.scatter(1, 0, marker="o", color=COLORS["red"])

    plt.title("Limiting Brownian Bridge")

if args.keep_open:
    plt.show(block=True)
