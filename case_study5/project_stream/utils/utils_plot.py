from collections.abc import Callable, Mapping, Sequence

import numpy as np
import matplotlib.pyplot as plt

from bayesflow.utils.dict_utils import compute_test_quantities
from bayesflow.utils.plot_utils import prepare_plot_data, add_titles_and_labels, prettify_subplots
from bayesflow.utils.ecdf import simultaneous_ecdf_bands
from bayesflow.utils.ecdf.ranks import fractional_ranks, distance_ranks


def calibration_ecdf(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    test_quantities: dict[str, Callable] = None,
    difference: bool = True,
    stacked: bool = False,
    rank_type: str | np.ndarray = "fractional",
    figsize: Sequence[float] = None,
    label_fontsize: int = 16,
    legend_fontsize: int = 14,
    legend_location: str = "lower right",
    title_fontsize: int = 18,
    tick_fontsize: int = 12,
    rank_ecdf_color: str | Sequence[str] | np.ndarray = "#132a70",
    fill_color: str = "grey",
    num_row: int = None,
    num_col: int = None,
    local_params: bool = False,
    title_local_params: str = "Stream",
    **kwargs,
) -> plt.Figure:
    # Optionally compute and prepend test quantities
    if test_quantities is not None:
        updated_data = compute_test_quantities(
            targets=targets,
            estimates=estimates,
            variable_keys=variable_keys,
            variable_names=variable_names,
            test_quantities=test_quantities,
        )
        variable_names = updated_data["variable_names"]
        variable_keys = updated_data["variable_keys"]
        estimates = updated_data["estimates"]
        targets = updated_data["targets"]

    # IMPORTANT: convert dict inputs to array format expected by rank functions
    plot_data = prepare_plot_data(
        estimates=estimates,
        targets=targets,
        variable_keys=variable_keys,
        variable_names=variable_names,
        num_col=num_col,
        num_row=num_row,
        figsize=figsize,
        stacked=stacked,
    )
    estimates = plot_data.pop("estimates")
    targets = plot_data.pop("targets")

    # Compute ranks
    if rank_type == "fractional":
        ranks = fractional_ranks(estimates, targets)
    elif rank_type == "distance":
        ranks = distance_ranks(estimates, targets, stacked=stacked, **kwargs.pop("ranks_kwargs", {}))
    else:
        raise ValueError(f"Unknown rank type: {rank_type}. Use 'fractional' or 'distance'.")

    n_vars = ranks.shape[-1]
    multi_color = False

    # Normalize colors: one color per variable
    if isinstance(rank_ecdf_color, np.ndarray):
        if rank_ecdf_color.ndim == 2:
            if rank_ecdf_color.shape[0] != n_vars:
                raise ValueError(
                    f"rank_ecdf_color has {rank_ecdf_color.shape[0]} colors, but there are {n_vars} variables."
                )
            colors = [tuple(c) for c in rank_ecdf_color]
            multi_color = True
        elif rank_ecdf_color.ndim == 1:
            colors = [tuple(rank_ecdf_color)] * n_vars
        else:
            raise ValueError("rank_ecdf_color ndarray must be 1D or 2D.")
    elif isinstance(rank_ecdf_color, Sequence) and not isinstance(rank_ecdf_color, (str, bytes)):
        if len(rank_ecdf_color) != n_vars:
            raise ValueError(f"rank_ecdf_color has {len(rank_ecdf_color)} colors, but there are {n_vars} variables.")
        colors = list(rank_ecdf_color)
        multi_color = True
    else:
        colors = [rank_ecdf_color] * n_vars

    # Plot ECDFs
    for j in range(n_vars):
        xx = np.repeat(np.sort(ranks[:, j]), 2)
        xx = np.pad(xx, (1, 1), constant_values=(0, 1))
        yy = np.linspace(0, 1, num=xx.shape[-1] // 2)
        yy = np.repeat(yy, 2)

        if difference:
            yy -= xx

        if stacked:
            if not isinstance(plot_data["axes"], np.ndarray):
                plot_data["axes"] = np.array([plot_data["axes"]])

            if multi_color:
                label = plot_data["variable_names"][j] if plot_data.get("variable_names") is not None else f"Var {j+1}"
            else:
                label = "Rank ECDFs" if j == 0 else None

            plot_data["axes"][0].plot(xx, yy, color=colors[j], alpha=0.95, label=label)
        else:
            plot_data["axes"].flat[j].plot(xx, yy, color=colors[j], alpha=0.95, label="Rank ECDF")

    # Uniform ECDF and bands
    alpha, z, L, U = simultaneous_ecdf_bands(estimates.shape[0], **kwargs.pop("ecdf_bands_kwargs", {}))

    if difference:
        L -= z
        U -= z
        ylab = "ECDF Difference"
    else:
        ylab = "ECDF"

    if not stacked:
        titles = plot_data["variable_names"]
    elif rank_type in ["distance", "random"]:
        titles = ["Joint ECDFs"]
        # titles = ""
    else:
        titles = ["Stacked ECDFs"]
        # titles = ""

    for i, (ax, title) in enumerate(zip(plot_data["axes"].flat, titles)):
        ax.fill_between(
            z, L, U, color=fill_color, alpha=0.2,
            # label=rf"{int((1 - alpha) * 100)}$\%$ Confidence Bands"
        )
        if local_params:
            ax.set_title(title_local_params, fontsize=title_fontsize)
        # if i == 0:
        #     ax.legend(fontsize=legend_fontsize, loc=legend_location)
        # AFTER
        # AFTER
        if i == 0:
            leg = ax.legend(
                fontsize=legend_fontsize,
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
                borderaxespad=0,
                frameon=False,          # removes the box
            )
            for line in leg.get_lines():
                line.set_linewidth(3)   # thicker color lines (adjust to taste)

        plot_data["fig"].tight_layout(rect=[0, 0, 0.82, 1])

    prettify_subplots(plot_data["axes"], num_subplots=plot_data["num_variables"], tick_fontsize=tick_fontsize)

    add_titles_and_labels(
        plot_data["axes"],
        plot_data["num_row"],
        plot_data["num_col"],
        xlabel=f"{rank_type.capitalize()} rank statistic",
        ylabel=ylab,
        label_fontsize=label_fontsize,
    )

    plot_data["fig"].tight_layout()
    return plot_data["fig"]