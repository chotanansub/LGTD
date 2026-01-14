"""
Visualization tools for decomposition results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Tuple
from LGTD.decomposition.lgtd import LGTDResult


def plot_decomposition(
    result: LGTDResult,
    ground_truth: Optional[Dict[str, np.ndarray]] = None,
    figsize: Tuple[int, int] = (14, 12),
    title: str = "Time Series Decomposition",
    show: bool = True,
    save_path: Optional[str] = None,
    model_name: str = "LGTD",
    init_point: int = 0
) -> plt.Figure:
    """
    Plot decomposition results with optional ground truth comparison.

    Args:
        result: LGTD decomposition result
        ground_truth: Optional dictionary with ground truth components
        figsize: Figure size (width, height)
        title: Plot title
        show: Whether to display the plot
        save_path: Path to save the figure (optional)
        model_name: Name of the model for legend labels (default: "LGTD")
        init_point: Index marking end of initialization period (highlighted in plot)

    Returns:
        Matplotlib figure object
    """
    colors = {
        'gt': '#2563eb',
        'est': '#dc2626',
        'original': '#000000'
    }

    n_plots = 4
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

    time = np.arange(len(result.y))

    # Add initialization region highlight to all subplots if init_point > 0
    if init_point > 0:
        for ax in axes:
            ax.axvspan(0, init_point, alpha=0.15, color='gray', zorder=0,
                      label='Init Period' if ax == axes[0] else '')

    # Original series
    axes[0].plot(time, result.y, label='Observed', color=colors['original'],
                linewidth=1.5, alpha=0.8)
    axes[0].set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    axes[0].set_title('(a) Original Time Series', fontsize=12, loc='left')
    axes[0].legend(loc='upper right', fontsize=10, framealpha=0.9)
    axes[0].grid(True, alpha=0.3, linestyle='--')

    # Trend
    if ground_truth is not None and 'trend' in ground_truth:
        axes[1].plot(time, ground_truth['trend'], label='Ground Truth',
                    color=colors['gt'], linewidth=2.5, alpha=0.8)
    axes[1].plot(time, result.trend, label=f'{model_name} Estimate',
                color=colors['est'], linewidth=2, alpha=0.7,
                linestyle='--' if ground_truth else '-')
    axes[1].set_ylabel('Trend', fontsize=12, fontweight='bold')
    axes[1].set_title('(b) Trend Component', fontsize=12, loc='left')
    axes[1].legend(loc='upper right', fontsize=10, framealpha=0.9)
    axes[1].grid(True, alpha=0.3, linestyle='--')

    # Seasonal
    if ground_truth is not None and 'seasonal' in ground_truth:
        axes[2].plot(time, ground_truth['seasonal'], label='Ground Truth',
                    color=colors['gt'], linewidth=2.5, alpha=0.8)
    axes[2].plot(time, result.seasonal, label=f'{model_name} Estimate',
                color=colors['est'], linewidth=2, alpha=0.7,
                linestyle='--' if ground_truth else '-')
    axes[2].set_ylabel('Seasonal', fontsize=12, fontweight='bold')
    axes[2].set_title('(c) Seasonal Component', fontsize=12, loc='left')
    axes[2].legend(loc='upper right', fontsize=10, framealpha=0.9)
    axes[2].grid(True, alpha=0.3, linestyle='--')

    # Residual
    if ground_truth is not None and 'residual' in ground_truth:
        axes[3].plot(time, ground_truth['residual'], label='Ground Truth',
                    color=colors['gt'], linewidth=2.5, alpha=0.8)
    axes[3].plot(time, result.residual, label=f'{model_name} Estimate',
                color=colors['est'], linewidth=2, alpha=0.7,
                linestyle='--' if ground_truth else '-')
    axes[3].set_ylabel('Residual', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Time', fontsize=12)
    axes[3].set_title('(d) Residual Component', fontsize=12, loc='left')
    axes[3].legend(loc='upper right', fontsize=10, framealpha=0.9)
    axes[3].grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig


def plot_comparison(
    ground_truth: Dict[str, np.ndarray],
    results_dict: Dict[str, Dict[str, np.ndarray]],
    component: str = 'trend',
    figsize: Tuple[int, int] = (14, 6),
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison of multiple decomposition methods for a single component.

    Args:
        ground_truth: Dictionary with ground truth components
        results_dict: Dictionary of {method_name: result_dict}
        component: Component to plot ('trend', 'seasonal', or 'residual')
        figsize: Figure size (width, height)
        title: Plot title (auto-generated if None)
        show: Whether to display the plot
        save_path: Path to save the figure (optional)

    Returns:
        Matplotlib figure object
    """
    if title is None:
        title = f"{component.capitalize()} Component Comparison"

    fig, ax = plt.subplots(figsize=figsize)

    time = np.arange(len(ground_truth[component]))

    # Plot ground truth
    ax.plot(time, ground_truth[component], label='Ground Truth',
           color='black', linewidth=2.5, alpha=0.9, linestyle='-')

    # Plot each method's result
    colors = plt.cm.Set2(np.linspace(0, 1, len(results_dict)))

    for idx, (method_name, result) in enumerate(results_dict.items()):
        if component in result:
            ax.plot(time, result[component], label=method_name,
                   color=colors[idx], linewidth=2, alpha=0.7, linestyle='--')

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(component.capitalize(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig


def plot_evaluation_bars(
    evaluation_df,
    metric: str = 'MSE',
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot bar chart comparison of evaluation metrics.

    Args:
        evaluation_df: Pandas DataFrame with evaluation results
        metric: Metric to plot ('MSE' or 'MAE')
        figsize: Figure size (width, height)
        title: Plot title (auto-generated if None)
        show: Whether to display the plot
        save_path: Path to save the figure (optional)

    Returns:
        Matplotlib figure object
    """
    if title is None:
        title = f'{metric.upper()} Comparison Across Methods'

    # Filter for specific metric
    metric_subset = evaluation_df[evaluation_df['metric'] == metric].copy()

    # Melt for seaborn
    components = ['trend', 'seasonal', 'residual']
    melted_data = metric_subset.melt(
        id_vars=['model', 'metric'],
        value_vars=components,
        var_name='component',
        value_name='error_value'
    )

    fig, ax = plt.subplots(figsize=figsize)

    barplot = sns.barplot(
        data=melted_data,
        x='model',
        y='error_value',
        hue='component',
        palette='Set2',
        ax=ax,
        alpha=0.85,
        edgecolor='black',
        linewidth=0.8
    )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Decomposition Method', fontsize=12)
    ax.set_ylabel(f'{metric.upper()} Error', fontsize=12)
    ax.tick_params(axis='x', rotation=0, labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    legend = ax.legend(
        title='Component',
        title_fontsize=11,
        fontsize=10,
        loc='upper right',
        frameon=True,
        fancybox=True,
        shadow=True
    )
    legend.get_title().set_fontweight('bold')

    # Add value labels on bars
    for container in barplot.containers:
        ax.bar_label(container, fmt='%.4f', fontsize=8, padding=3)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig
