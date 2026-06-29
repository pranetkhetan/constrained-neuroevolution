"""
Supplementary Figure S14 (fig:supp_power_curve) — Power curve for specificity test.
Output: fig_supp_power_curve.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from ._style import apply_pub_style, EVOLVED_COL, RANDOM_COL, FS_ANNOT, FIGSIZE, LW_SCALE, MARKER_SCALE


def generate(store, figures_dir: str) -> list[str]:
    apply_pub_style()
    d = store.phase3c()

    dz_obs = d['cohen_dz']
    mde_d = d['mde_d']
    mde_raw = d['mde_raw']
    req_n = d['required_n_80pct']

    def power_approx(d_z, n_val, alpha=0.05):
        z_a = norm.ppf(1 - alpha)
        return 1 - norm.cdf(z_a - d_z * np.sqrt(n_val))

    n_vals = np.arange(1, 35)
    power_obs = [power_approx(dz_obs, n) for n in n_vals]
    power_mde = [power_approx(mde_d, n) for n in n_vals]

    fig, ax = plt.subplots(figsize=FIGSIZE['s14'])

    ax.plot(n_vals, power_obs, color=EVOLVED_COL, lw=1.4 * LW_SCALE,
            label=f"Observed $d_z = {dz_obs:.2f}$")
    ax.plot(n_vals, power_mde, color=RANDOM_COL, lw=1.2 * LW_SCALE, ls='--',
            label=f"MDE $d_z = {mde_d:.2f}$ (80% threshold)")

    ax.axhline(0.80, color='0.5', lw=0.8 * LW_SCALE, ls=':')
    ax.axhline(0.95, color='0.7', lw=0.8 * LW_SCALE, ls=':')
    ax.text(34.2, 0.80, "80%", va='center', fontsize=FS_ANNOT, color='0.5')
    ax.text(34.2, 0.95, "95%", va='center', fontsize=FS_ANNOT, color='0.7')

    pw9 = power_approx(dz_obs, 9)
    ax.scatter([9], [pw9], color=EVOLVED_COL, zorder=5, s=30 * MARKER_SCALE)
    ax.annotate(f"n=9\n(power>{pw9:.2f})",
                xy=(9, pw9), xytext=(12, 0.87),
                arrowprops=dict(arrowstyle='->', lw=0.7 * LW_SCALE, color='0.4'),
                fontsize=FS_ANNOT, color=EVOLVED_COL)

    pw_req = power_approx(mde_d, req_n)
    ax.scatter([req_n], [pw_req], color=RANDOM_COL, zorder=5, s=30 * MARKER_SCALE)
    ax.annotate(f"n={req_n} for 80%\nat MDE",
                xy=(req_n, pw_req), xytext=(req_n + 4, 0.6),
                arrowprops=dict(arrowstyle='->', lw=0.7 * LW_SCALE, color='0.4'),
                fontsize=FS_ANNOT, color=RANDOM_COL)

    ax.set_xlabel("Number of mice (n)")
    ax.set_ylabel("Attained power (one-tailed \u03b1 = 0.05)")
    ax.set_title("Power curve \u2014 permutation specificity test\n"
                 f"(Cohen's $d_z = {dz_obs:.2f}$; MDE = {mde_raw:.2f} fitness units)")
    ax.set_xlim(1, 34)
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False)

    out = os.path.join(figures_dir, "fig_supp_power_curve.png")
    fig.savefig(out)
    plt.close(fig)
    return [out]
