import numpy as np
import scipy.stats as stats

from utils import (
    create_default_parser,
)

parser = create_default_parser()
args = parser.parse_args()

sample_size = args.sample_size

DATA_GEN = {}
DATA_GEN["Exponential(1)"] = lambda: stats.expon.rvs(scale=1, size=sample_size)
DATA_GEN["Uniform(0,1)"] = lambda: stats.uniform.rvs(size=sample_size)
DATA_GEN["t(1)"] = lambda: stats.t.rvs(df=1, size=sample_size)
DATA_GEN["N(0.1,1)"] = lambda: stats.norm.rvs(loc=0.1, scale=1, size=sample_size)
DATA_GEN["t(3)"] = lambda: stats.t.rvs(df=10, size=sample_size)
DATA_GEN["t(100)"] = lambda: stats.t.rvs(df=10, size=sample_size)
DATA_GEN["N(0,1)"] = lambda: stats.norm.rvs(loc=0, scale=1, size=sample_size)


h0_CDF = stats.norm.cdf
alpha = 0.05

# Create confusion matrix for each distribution
CM = {}
for dist_name in DATA_GEN.keys():
    # CM format
    #            H0 Rejected    H0 Not Rejected
    #  H0 true     Type I          Correct
    #  H0 false    Correct        Type II
    CM[dist_name] = np.zeros((2, 2))

NUM_TRIALS = 200
for trial_id in range(NUM_TRIALS):
    for dist_name, data_gen in DATA_GEN.items():
        D_stat, p_val = stats.kstest(data_gen(), h0_CDF)
        is_H0_rejected = p_val < alpha

        if trial_id == 1:
            print(f"KS test for {dist_name}")
            print(f"   D-statistic = {D_stat:.4f}")
            print(f"   p-value = {p_val:.4f}")
            if is_H0_rejected:
                print("   🙅 Reject the null hypothesis")
            else:
                print("   🤷 Failed to reject the null hypothesis")

        row_number = 0 if dist_name == "N(0,1)" else 1
        col_number = 0 if is_H0_rejected else 1
        CM[dist_name][row_number, col_number] += 1

for dist_name, cm in CM.items():
    print(f"\nConfusion matrix for {dist_name} after {NUM_TRIALS} trials:")
    print(cm / NUM_TRIALS)
