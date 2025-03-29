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
parser.set_defaults(sample_size=7)  # This script is most pedagogical at low sample sizes.
args = parser.parse_args()

BASE_X = np.linspace(-3, 3, 1000)

sample_size = args.sample_size

fig = plt.figure(figsize=(8, 5))

for _ in range(args.num_frames):
    plt.pause(DEFAULT_PAUSE)
    plt.clf()
    plt.cla()

    # ---------------------------------------------------------------------------------
    # Draw random sample from N(0,1), calculate empirical and theoretical CDFs
    # ---------------------------------------------------------------------------------
    data = sample_gaussian_data(n=sample_size, loc=0, scale=1)
    ecdf = EmpiricalCDF(data)
    true_cdf = stats.norm.cdf(BASE_X, loc=0, scale=1)

    # Plot empirical CDF
    plt.axhline(0, color=COLORS["black"], lw=1, linestyle="--")
    plt.axhline(1, color=COLORS["black"], lw=1, linestyle="--")
    plt.scatter(data, y=np.zeros(len(data)), marker="*", color=COLORS["blue"])
    plt.step(ecdf.x, ecdf.y, where="post", label="Empirical CDF", color=COLORS["blue"])

    # Plot true CDF
    plt.plot(BASE_X, true_cdf, label="Theoretical CDF", color=COLORS["green"])

    # Plot distance between empirical and theoretical CDF at 0 if flag is set
    if args.show_0_arrow:
        plt.arrow(
            0,
            stats.norm.cdf(0),
            0,
            ecdf(0) - stats.norm.cdf(0),
            color=COLORS["red"],
            lw=2,
            head_width=0.05,
            head_length=0.05,
            length_includes_head=True,
        )

    plt.title("Empirical CDF vs Theoretical CDF")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2)

if args.keep_open:
    plt.show(block=True)
