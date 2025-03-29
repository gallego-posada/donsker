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

# By CLT,
#       sqrt(n) (ECDF_n(0) - CDF(0)) ~> N(0, Phi(0) * (1 - Phi(0))),
# where Phi(0) = P(N(0,1) <= 0). In other words,
#       ECDF_n(0) ~> N(Phi(0), Phi(0) * (1 - Phi(0)) / n)

sample_size = args.sample_size
NUM_TRIALS = 1_000


def simulate_ecdf_at_zero(n):
    data = sample_gaussian_data(n, loc=0, scale=1)
    ecdf = EmpiricalCDF(data)
    return ecdf(0)


fig = plt.figure(figsize=(8, 5))

for _ in range(args.num_frames):
    plt.pause(DEFAULT_PAUSE)
    plt.clf()
    plt.cla()

    # ---------------------------------------------------------------------------------
    # Draw random sample from N(0,1), calculate empirical CDF and check value at 0
    # ---------------------------------------------------------------------------------
    F0_samples = [simulate_ecdf_at_zero(n=sample_size) for _ in range(NUM_TRIALS)]

    plt.hist(
        np.array(F0_samples),
        bins=20,
        density=True,
        alpha=0.5,
        color=COLORS["blue"],
        label="Empirical",
    )

    phi_0 = stats.norm.cdf(0, loc=0, scale=1)
    sigma = np.sqrt(phi_0 * (1 - phi_0) / sample_size)
    x_linspace = phi_0 + np.linspace(-3 * sigma, 3 * sigma, 100)
    plt.plot(
        x_linspace,
        stats.norm.pdf(x_linspace, loc=phi_0, scale=sigma),
        color=COLORS["red"],
        label="Theoretical",
    )

    plt.axvline(phi_0, color=COLORS["black"], lw=1, linestyle="--")
    # plt.show()

if args.keep_open:
    plt.show(block=True)
