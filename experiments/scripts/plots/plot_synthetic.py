#!/usr/bin/env python3
"""
Generate publication-quality plots for synthetic dataset decomposition results.

This script generates all plot variations:
- By-dataset comparisons (standard + zoom)
- By-model multi-dataset plots (standard + zoom)
- Colorful LGTD rainbow gradient versions
"""

from plot_components import (
    plot_dataset_comparison,
    plot_single_model_all_datasets,
    load_synthetic_result,
    load_ground_truth
)


# Configuration
DATASETS = [
    'synth1', 'synth2', 'synth3',
    'synth4', 'synth5', 'synth6',
    'synth7', 'synth8', 'synth9'
]

MODELS = [
    'LGTD',
    'STL',
    'OnlineSTL',
    'ASTD_Online',
    'FastRobustSTL'
]

ALL_MODELS = [
    'LGTD',
    'LGTD_Linear',
    'LGTD_LOWESS',
    'STL',
    'OnlineSTL',
    'OneShotSTL',
    'ASTD',
    'ASTD_Online',
    'FastRobustSTL',
    'STR'
]

RESULTS_DIR = 'experiments/results/decompositions/synthetic'
OUTPUT_DIR = 'experiments/results/figures/synthetic'


def main():
    """Generate all synthetic dataset plots."""
    print("=" * 70)
    print("Generating Synthetic Dataset Plots")
    print("=" * 70)
    print(f"Datasets: {len(DATASETS)}")
    print(f"Default Models: {', '.join(MODELS)}")
    print(f"All Models: {', '.join(ALL_MODELS)}")
    print()

    plot_types = [
        ('Standard', False, False),
        ('Zoomed', True, False),
        ('Colorful', False, True),
        ('Colorful Zoomed', True, True),
    ]

    for plot_name, zoom, colorful in plot_types:
        print(f"\n{plot_name} Plots:")
        print("-" * 70)

        # By-dataset plots (default models)
        print("  By-dataset comparisons (default models):")
        for dataset in DATASETS:
            plot_dataset_comparison(
                dataset_name=dataset,
                models=MODELS,
                results_dir=RESULTS_DIR,
                output_dir=OUTPUT_DIR,
                load_func=load_synthetic_result,
                gt_func=load_ground_truth,
                zoom=zoom,
                colorful=colorful
            )

        # By-dataset plots (all models)
        print("  By-dataset comparisons (all models):")
        for dataset in DATASETS:
            plot_dataset_comparison(
                dataset_name=dataset,
                models=ALL_MODELS,
                results_dir=RESULTS_DIR,
                output_dir=OUTPUT_DIR,
                load_func=load_synthetic_result,
                gt_func=load_ground_truth,
                zoom=zoom,
                colorful=colorful,
                plot_type='by_dataset_full'
            )

        # By-model plots
        print("  By-model multi-dataset:")
        for model in MODELS:
            plot_single_model_all_datasets(
                model_name=model,
                datasets=DATASETS,
                results_dir=RESULTS_DIR,
                output_dir=OUTPUT_DIR,
                load_func=load_synthetic_result,
                gt_func=load_ground_truth,
                zoom=zoom,
                colorful=colorful
            )

    print()
    print("=" * 70)
    print("Plot Generation Complete!")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}/")
    print()


if __name__ == '__main__':
    main()
