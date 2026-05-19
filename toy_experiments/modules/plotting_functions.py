import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sbn
from tueplots import bundles

COLORS = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "gray":   "#7F7F7F",
    "black":  "#000000",
}
COLORS_NAMES = ["blue", "orange", "green", "red", "purple", "gray", "black"]
plt.rcParams.update({
    "figure.figsize": (3.3, 2.3),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "lines.linewidth": 1.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})


def plot_entropy_and_bounds_with_ood_bins_icml(
    entropy,
    lower_bound,
    upper_bound,
    aleatoric=None,
    is_ood=None,
    title=None,
    sort_by: str = "lower_bound",
    num_bins: int = 50,
    smooth: bool = True,
    window: int = 100,
    save_path: str = ".",
    save_plot: bool = False,
):
    """
    Plot predictive entropy with lower/upper bounds and overlay
    OOD concentration statistics computed in bins.

    ICML-ready:
    - single-column width
    - restrained color palette
    - no LaTeX dependency
    """

    # -------------------------
    # ICML + seaborn style
    # -------------------------
    params = bundles.icml2022()
    params.update({
        "text.usetex": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
    })
    plt.rcParams.update(params)

    sbn.set_context("paper", font_scale=0.9)
    sbn.set_style("white")

    # -------------------------
    # Input checks
    # -------------------------
    entropy = np.asarray(entropy)
    lower_bound = np.asarray(lower_bound)
    upper_bound = np.asarray(upper_bound)

    if aleatoric is None or is_ood is None:
        raise ValueError("aleatoric and is_ood must be provided")

    aleatoric = np.asarray(aleatoric)
    is_ood = np.asarray(is_ood).astype(bool)

    # -------------------------
    # Sorting
    # -------------------------
    if sort_by == "lower_bound":
        idx = np.argsort(lower_bound)
    elif sort_by == "entropy":
        idx = np.argsort(entropy)
    else:
        raise ValueError("sort_by must be 'lower_bound' or 'entropy'")

    entropy = entropy[idx]
    lower_bound = lower_bound[idx]
    upper_bound = upper_bound[idx]
    aleatoric = aleatoric[idx]
    is_ood = is_ood[idx]

    x = np.linspace(0, 1, len(entropy))

    # -------------------------
    # Optional smoothing
    # -------------------------
    epistemic = entropy - aleatoric

    if smooth and window > 1:
        kernel = np.ones(window) / window

        def smooth_arr(a):
            return np.convolve(a, kernel, mode="valid")

        entropy = smooth_arr(entropy)
        lower_bound = smooth_arr(lower_bound)
        upper_bound = smooth_arr(upper_bound)
        aleatoric = smooth_arr(aleatoric)
        epistemic = smooth_arr(epistemic)
        is_ood = is_ood[: len(entropy)]
        x = np.linspace(0, 1, len(entropy))

    # epistemic = entropy - aleatoric

    # -------------------------
    # Figure (ICML single column)
    # -------------------------
    ICML_COLUMN_WIDTH = 3.25
    ICML_TEXT_WIDTH = 6.75
    fig, ax1 = plt.subplots(
        figsize=(ICML_COLUMN_WIDTH, ICML_TEXT_WIDTH * 0.4)
    )
    Y_BASELINE = -0.10  # desired lower bound

    entropy_max = max(
        np.nanmax(upper_bound),
        np.nanmax(entropy),
    )

    ax1.set_ylim(Y_BASELINE, entropy_max)

    # ---- main curves ----
    ax1.plot(
        x, entropy,
        color=COLORS["blue"],
        lw=1.8,
        label="Predictive entropy",
        alpha=0.9,
    )
    ax1.plot(
        x, lower_bound,
        color=COLORS["gray"],
        lw=1.2,
        linestyle="--",
        # label="Lower bound",
    )
    ax1.plot(
        x, upper_bound,
        color=COLORS["gray"],
        lw=1.2,
        linestyle="--",
        # label="Upper bound",
    )

    ax1.set_xlabel("Samples sorted by bounds")
    ax1.set_ylabel("Entropy")

    # -------------------------
    # OOD statistics per bin
    # -------------------------
    bins = np.linspace(0, len(entropy), num_bins + 1).astype(int)
    bin_centers = np.linspace(0, 1, num_bins)

    ood_frac = np.zeros(num_bins)
    ood_epi = np.full(num_bins, np.nan)
    ood_ale = np.full(num_bins, np.nan)
    iid_epi = np.full(num_bins, np.nan)
    iid_ale = np.full(num_bins, np.nan)

    # Compute the total averaged values to
    epi = np.full(num_bins, np.nan)
    ale = np.full(num_bins, np.nan)

    for i in range(num_bins):
        start, end = bins[i], bins[i + 1]
        if end <= start:
            continue

        mask = is_ood[start:end]
        ood_frac[i] = mask.mean()

        if mask.any():
            ood_epi[i] = epistemic[start:end][mask].mean()
            ood_ale[i] = aleatoric[start:end][mask].mean()
            iid_epi[i] = epistemic[start:end][~mask].mean()
            iid_ale[i] = aleatoric[start:end][~mask].mean()
            epi[i] = epistemic[start:end].mean()
            ale[i] = aleatoric[start:end].mean()

    # -------------------------
    # Secondary axis: OOD fraction
    # -------------------------
    ax2 = ax1.twinx()

    # Compute relative zero position
    f = (0.0 - Y_BASELINE) / (entropy_max - Y_BASELINE)

    # Right axis: choose lower bound so that y=0 aligns physically
    ood_ymin = -f / (1.0 - f)
    ax2.set_ylim(ood_ymin, 1.0)
    # ax2.set_ylim(0, 1)

    ax2.plot(
        bin_centers,
        ood_frac,
        color=COLORS["red"],
        lw=1.6,
        label="OOD fraction",
    )
    ax2.set_ylabel("OOD fraction")

    # -------------------------
    # Epistemic gap (OOD)
    # -------------------------
    ax1.fill_between(
        bin_centers,
        ale,
        ale + epi,
        color=COLORS["blue"],
        alpha=0.15,
        label="Epistemic gap",
    )

    ax1.scatter(
        bin_centers,
        ood_epi,
        s=14,
        color=COLORS["red"],
        marker="o",
        zorder=3,
        label="Epistemic of OOD",
    )

    ax1.scatter(
        bin_centers,
        iid_epi,
        s=14,
        color=COLORS["black"],
        marker="x",
        linewidths=1.2,
        zorder=3,
        label="Epistemic of ID",
    )

    # -------------------------
    # Legend (compact, ICML-safe)
    # -------------------------
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    
    # Left axis legend (entropy-related)
    ax1.legend(
        loc="center left",
        bbox_to_anchor=(0.0, 0.65),
        frameon=False,
        fontsize=5,
        title_fontsize=7,
    )

    # Right axis legend (OOD-related)
    ax2.legend(
        loc="center right",
        bbox_to_anchor=(1.0, 0.8),
        frameon=False,
        fontsize=5,
        title_fontsize=7,
    )
    ax2.yaxis.label.set_color(COLORS["red"])
    ax2.tick_params(axis="y", colors=COLORS["red"])

    if title:
        ax1.set_title(title)

    plt.tight_layout()

    if save_plot is not False:
        plt.savefig(f"{save_path}/entropy_bounds_plot_{title}.pdf",
        bbox_inches="tight", dpi=300)
        # Save also as png
        plt.savefig(f"{save_path}/entropy_bounds_plot_{title}.png",
        bbox_inches="tight", dpi=300)




def plot_binned_averages_icml(
    x_y_pairs,
    num_bins,
    x_label: str,
    y_label: str,
    title: str = None,
    save_path: str = ".",
):
    """
    ICML-style binned average plot for multiple experiments.
    Plots multiple curves in the same figure.
    """

    # -------------------------
    # ICML + seaborn style (same as previous plot)
    # -------------------------
    params = bundles.icml2022()
    params.update({
        "text.usetex": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
    })
    plt.rcParams.update(params)

    sbn.set_context("paper", font_scale=0.9)
    sbn.set_style("white")

    # -------------------------
    # Binning
    # -------------------------
    ICML_COLUMN_WIDTH = 3.25
    ICML_TEXT_WIDTH = 6.75
    fig, ax = plt.subplots(
        figsize=(ICML_COLUMN_WIDTH, ICML_TEXT_WIDTH * 0.4)
    )

    # Loop through the different (x, y) pairs
    for c, (x_axis, y_axis, label) in enumerate(x_y_pairs):
        print(f"c: {c}")
        x_axis = np.asarray(x_axis)
        y_axis = np.asarray(y_axis)

        bins = np.linspace(x_axis.min(), x_axis.max(), num_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        avg_gap = np.full(num_bins, np.nan)
        std_gap = np.full(num_bins, np.nan)

        for i in range(num_bins):
            mask = (x_axis >= bins[i]) & (x_axis < bins[i + 1])
            if mask.any():
                avg_gap[i] = y_axis[mask].mean()
                std_gap[i] = y_axis[mask].std()

        # Plot each (x, y) pair
        ax.plot(
            bin_centers,
            avg_gap,
            lw=1.8,
            marker="o",
            markersize=3.5,
            label=label,
            color=COLORS[COLORS_NAMES[c]]
        )

        ax.fill_between(
            bin_centers,
            avg_gap - std_gap,
            avg_gap + std_gap,
            alpha=0.15,
            color=COLORS[COLORS_NAMES[c]]
        )

    # -------------------------
    # Labels & cosmetics
    # -------------------------
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if title:
        ax.set_title(title)

    ax.legend(
        frameon=False,
        fontsize=5,
        loc="best",
    )

    # Optional: subtle horizontal grid (same philosophy as previous)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.xaxis.grid(False)

    plt.tight_layout()

    plt.savefig(f"{save_path}/aleatoric_epistemic_relation_multiple_{title}.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{save_path}/aleatoric_epistemic_relation_multiple_{title}.png", bbox_inches="tight", dpi=300)



def plot_ensemble_histogram_icml(
    acc_per_model,
    ensemble_acc,
    acc_per_model_id,
    ensemble_acc_id,
    title=None,
    save_path: str = ".",
    ):
    """
    ICML-style bar plot comparing individual model accuracies
    with ensemble accuracy.
    """
    # -------------------------
    # ICML + seaborn style
    # -------------------------
    params = bundles.icml2022()
    params.update({
        "text.usetex": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
    })
    plt.rcParams.update(params)

    sbn.set_context("paper", font_scale=0.9)
    sbn.set_style("white")

    # -------------------------
    # Data
    # -------------------------
    accs = np.asarray(acc_per_model) * 100.0
    ensemble_acc = ensemble_acc * 100.0
    accs_id = np.asarray(acc_per_model_id) * 100.0
    ensemble_acc_id = ensemble_acc_id * 100.0

    x = np.arange(len(accs))
    x_id = np.arange(len(accs_id))

    # -------------------------
    # Figure (ICML single column)
    # -------------------------
    ICML_COLUMN_WIDTH = 3.25
    ICML_TEXT_WIDTH = 6.75
    fig, ax = plt.subplots(
        figsize=(ICML_COLUMN_WIDTH, ICML_TEXT_WIDTH * 0.4)
    )

    # -------------------------
    # Bars: individual models
    # -------------------------
    ax.bar(
        x_id,
        accs_id,
        color=COLORS["blue"],
        alpha=0.75,
        width=0.75,
        label="Individual models (ID)",
    )
    ax.bar(
        x,
        accs,
        color=COLORS["red"],
        alpha=0.75,
        width=0.75,
        label="Individual models (ID+OOD)",
    )

    # -------------------------
    # Ensemble accuracy line
    # -------------------------
    ax.axhline(
        y=ensemble_acc_id,
        linestyle="--",
        linewidth=1.8,
        color=COLORS["blue"],
        label=f"Ensemble accuracy (ID) ({ensemble_acc_id:.2f}%)",
    )
    ax.axhline(
        y=ensemble_acc,
        linestyle="--",
        linewidth=1.8,
        color=COLORS["red"],
        label=f"Ensemble accuracy (ID+OOD) ({ensemble_acc:.2f}%)",
    )

    # -------------------------
    # Labels & cosmetics
    # -------------------------
    ax.set_xlabel("Model index")
    ax.set_ylabel("Accuracy (%)")

    ax.set_xlim(-0.6, len(accs) - 0.4)

    # Optional: focus y-range for readability
    ymin = min(accs.min(), ensemble_acc) - 5.0
    ymax = max(accs_id.max(), ensemble_acc_id) + 1.0
    ax.set_ylim(ymin, ymax)

    ax.legend(
        frameon=False,
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.90),
    )

    # Subtle horizontal grid (same style as other plots)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.xaxis.grid(False)

    plt.tight_layout()

    plt.savefig(f"{save_path}/accuracies_histogram_{title}.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{save_path}/accuracies_histogram_{title}.png", bbox_inches="tight", dpi=300) 


# Plot coverage accuracy curves
def plot_risk_coverage(uncertainties: dict, accuracy, ax=None, label=None, title=None, save_path: str = ".", is_ood=None):
    """
    Plot risk-coverage curve:
    x-axis: 1 - coverage (fraction of discarded samples)
    y-axis: accuracy on remaining samples

    uncertainty: (N,) higher=more uncertain
    accuracy: (N,) binary {0,1}
    """
    print(f"Dictionay of uncertainties keys: {list(uncertainties.keys())}")
    # -------------------------
    # ICML + seaborn style
    # -------------------------
    params = bundles.icml2022()
    params.update({
        "text.usetex": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
    })
    plt.rcParams.update(params)
    
    sbn.set_context("paper", font_scale=0.9)
    sbn.set_style("white")
    
    # -------------------------
    # Figure (ICML single column)
    # -------------------------
    ICML_COLUMN_WIDTH = 3.25
    ICML_TEXT_WIDTH = 6.75
    fig, ax = plt.subplots(1, 1, figsize=(ICML_COLUMN_WIDTH, ICML_TEXT_WIDTH * 0.4))
    
    # Convert to numpy
    number_plots = len(uncertainties)
    accuracy = np.asarray(accuracy)

    if is_ood is not None:
        # Make accuracy zero for OOD samples
        is_ood = np.asarray(is_ood).astype(bool)
        accuracy = accuracy * (~is_ood).astype(int)

    for i, (labeled, uncertainty) in enumerate(uncertainties.items()):
        uncertainty = np.asarray(uncertainty)

        # Sort by uncertainty
        sorted_indices = np.argsort(-uncertainty)
        accuracy_sorted = accuracy[sorted_indices]

        N = len(accuracy_sorted)

        # Coverage decreases as we discard uncertainty samples
        coverage = np.arange(N, 0, -1) / N
        discarded = 1.0 - coverage

        # Cumulative accuracy on remaining samples
        cum_correct = np.cumsum(accuracy_sorted[::-1])[::-1]
        accuracy_remaining = cum_correct / np.arange(N, 0, -1)

        ax.plot(discarded, accuracy_remaining, label=labeled, color=COLORS[COLORS_NAMES[i]])

    # -------------------------
    # Labels & cosmetics
    # -------------------------
    ax.set_xlabel("1 - Coverage (fraction of discarded samples)")
    ax.set_ylabel("Accuracy on remaining samples")

    ax.set_xlim(0.0,1.0)

    # Optional: focus y-range for readability
    ax.set_ylim(accuracy.mean() - 0.05, 1.05)

    ax.legend(
        frameon=False,
        fontsize=7,
        loc="best",
        # bbox_to_anchor=(0.02, 0.90),
    )

    # Subtle horizontal grid (same style as other plots)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.xaxis.grid(False)
    # ax.grid(True)

    plt.tight_layout()

    plt.savefig(f"{save_path}/ROC_{title}.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{save_path}/ROC_{title}.png", bbox_inches="tight", dpi=300)