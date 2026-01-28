#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec

# --- ACADEMIC CONFIG (MAXIMUM LEGIBILITY) ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 20,                
    'axes.labelsize': 22,
    'axes.titlesize': 24,          
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.linewidth': 1.5,
    'grid.linewidth': 0.8,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05      
})

MODEL_COLORS = {
    'LGTD': '#e41a1c',
    'STL': '#377eb8',
    'STR': '#CCAC00',  
    'OnlineSTL': '#4daf4a',
    'OneShotSTL': '#66c2a5',
    'ASTD': '#984ea3',
    'ASTD_Online': '#ff7f00',
    'RobustSTL': '#a65628',
    'FastRobustSTL': '#f781bf'
}

MAX_LENGTH = 400
PROJECT_ROOT = Path(__file__).parent.parent.parent
GROUND_TRUTH_COLOR = "#000000" 

def normalize_name(name):
    return name.replace("_", "").lower()

def get_color_key(filename_stem):
    norm_stem = normalize_name(filename_stem)
    for key in MODEL_COLORS.keys():
        if normalize_name(key) == norm_stem:
            return key
    return None

def load_decomposition(decomp_dir: Path, dataset: str, model_filename: str):
    decomp_path = decomp_dir / dataset / f"{model_filename}.json"
    if not decomp_path.exists(): return None
    try:
        with open(decomp_path, 'r') as f:
            data = json.load(f)
        t, s, r = np.array(data.get('trend', [])), np.array(data.get('seasonal', [])), np.array(data.get('residual', []))
        if len(t) > MAX_LENGTH:
            idx = np.linspace(0, len(t)-1, MAX_LENGTH, dtype=int)
            t, s, r = t[idx], s[idx], r[idx]
        return t, s, r
    except: return None

def load_ground_truth(dataset_name: str):
    gt_path = PROJECT_ROOT.parent / 'data' / 'synthetic' / 'datasets' / 'all_synthetic_datasets.json'
    if not gt_path.exists():
        return None
    try:
        with open(gt_path, 'r') as f:
            data = json.load(f)
        for ds in data.get('datasets', []):
            if ds['name'] == dataset_name:
                d = ds['data']
                t, s, r = np.array(d.get('trend', [])), np.array(d.get('seasonal', [])), np.array(d.get('residual', []))
                if len(t) > MAX_LENGTH:
                    idx = np.linspace(0, len(t)-1, MAX_LENGTH, dtype=int)
                    t, s, r = t[idx], s[idx], r[idx]
                return t, s, r
        return None
    except:
        return None

def plot_combined_horizontal_barplot(ax, data, color, dataset_name="", ground_truth=None):
    """ REQUIRED FUNCTION NAME - Strictly preserved. """
    if ground_truth is not None and len(ground_truth) > 0:
        norm_gt = (ground_truth - np.mean(ground_truth)) / (np.std(ground_truth) + 1e-8)
        ax.plot(norm_gt, color=GROUND_TRUTH_COLOR, lw=2.5, alpha=0.4, zorder=1)

    if data is not None and len(data) > 0:
        norm_data = (data - np.mean(data)) / (np.std(data) + 1e-8)
        ax.plot(norm_data, color=color, lw=1.8, zorder=2)
        ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.3)
    
    ax.set_xticks([]); ax.set_yticks([])
    for s in ['left', 'top', 'right', 'bottom']:
        ax.spines[s].set_visible(False)

def process_source(source_dir: Path, source_name: str, dataset_filter=None):
    if not source_dir.exists():
        print(f"Skipping: {source_dir}")
        return

    all_datasets = sorted([d.name for d in source_dir.iterdir() if d.is_dir()])
    if dataset_filter:
        datasets = [ds for ds in all_datasets if ds in dataset_filter]
    else:
        datasets = all_datasets[:3]

    exclude_raw = {'LGTD_LOWESS', 'LGTD_Linear', 'ASTD'}
    exclude_norm = {normalize_name(m) for m in exclude_raw}
    is_synthetic = source_name == "synthetic"

    fig = plt.figure(figsize=(45.0, 15.0))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.88, wspace=0.04)
    subfigs = fig.subfigures(1, len(datasets), wspace=0.04)

    if len(datasets) == 1:
        subfigs = [subfigs]

    for d_idx, (ds_name, subfig) in enumerate(zip(datasets, subfigs)):
        ds_path = source_dir / ds_name
        json_files = [m.stem for m in ds_path.glob("*.json")]

        valid_files = []
        for jf in json_files:
            norm_jf = normalize_name(jf)
            color_key = get_color_key(jf)
            if norm_jf not in exclude_norm and color_key:
                valid_files.append((jf, color_key))

        valid_files.sort(key=lambda x: list(MODEL_COLORS.keys()).index(x[1]))
        gt_comps = load_ground_truth(ds_name) if is_synthetic else None

        subfig.suptitle(ds_name.replace("_", " ").upper(), fontweight='bold', fontsize=35)

        axes = subfig.subplots(len(valid_files), 3, gridspec_kw={'hspace': 0.1, 'wspace': 0.02})
        if len(valid_files) == 1:
            axes = axes.reshape(1, -1)

        for m_idx, (m_filename, color_key) in enumerate(valid_files):
            comps = load_decomposition(source_dir, ds_name, m_filename)
            color = MODEL_COLORS[color_key]

            if d_idx == 0:
                axes[m_idx, 0].annotate(m_filename, xy=(-0.15, 0.5), xycoords='axes fraction',
                                       ha='right', va='center', fontweight='bold', fontsize=28)

            for c_idx in range(3):
                ax = axes[m_idx, c_idx]
                data = comps[c_idx] if comps else None
                gt_data = gt_comps[c_idx] if gt_comps else None
                plot_combined_horizontal_barplot(ax, data, color, ground_truth=gt_data)

                if m_idx == 0:
                    ax.set_title(["Trend", "Seasonal", "Resid."][c_idx], fontsize=28, pad=10)

    suffix = "_".join(dataset_filter) if dataset_filter else "decomp_comparison"
    out_file = source_dir / f"{source_name}_{suffix}.png"
    plt.savefig(out_file, format='png', dpi=300)
    plt.close(fig)

def process_synthetic_grid(source_dir: Path, source_name: str):
    """Create a 3x3 grid plot for synth1-synth9 with MASSIVE text and REDUCED space."""
    if not source_dir.exists():
        print(f"Skipping: {source_dir}")
        return

    all_datasets = sorted([d.name for d in source_dir.iterdir() if d.is_dir()])
    synth_datasets = [ds for ds in all_datasets if ds.startswith('synth') and ds[5:].isdigit() and int(ds[5:]) <= 9][:9]

    exclude_raw = {'LGTD_LOWESS', 'LGTD_Linear', 'ASTD'}
    exclude_norm = {normalize_name(m) for m in exclude_raw}

    # Giant figure size to handle massive text without overlap
    fig = plt.figure(figsize=(65.0, 45.0))
    # hspace/wspace = 0.04 collapses the distance between the 9 datasets
    gs = GridSpec(3, 3, figure=fig, left=0.1, right=0.98, bottom=0.02, top=0.95, hspace=0.04, wspace=0.04)

    for grid_idx, ds_name in enumerate(synth_datasets):
        ds_path = source_dir / ds_name
        json_files = [m.stem for m in ds_path.glob("*.json")]

        valid_files = []
        for jf in json_files:
            norm_jf = normalize_name(jf)
            color_key = get_color_key(jf)
            if norm_jf not in exclude_norm and color_key:
                valid_files.append((jf, color_key))

        valid_files.sort(key=lambda x: list(MODEL_COLORS.keys()).index(x[1]))
        gt_comps = load_ground_truth(ds_name)

        grid_row = grid_idx // 3
        grid_col = grid_idx % 3

        subfig = fig.add_subfigure(gs[grid_row, grid_col])
        # REALLY HUGE dataset title
        subfig.suptitle(ds_name.replace("_", " ").upper(), fontweight='bold', fontsize=55, y=0.98)

        # hspace = 0.02 removes vertical gaps between model rows within a dataset
        axes = subfig.subplots(len(valid_files), 3, gridspec_kw={'hspace': 0.02, 'wspace': 0.01})
        if len(valid_files) == 1:
            axes = axes.reshape(1, -1)

        for m_idx, (m_filename, color_key) in enumerate(valid_files):
            comps = load_decomposition(source_dir, ds_name, m_filename)
            color = MODEL_COLORS[color_key]

            # HUGE model labels on the left edge
            if grid_col == 0:
                axes[m_idx, 0].annotate(m_filename, xy=(-0.08, 0.5), xycoords='axes fraction',
                                       ha='right', va='center', fontweight='bold', fontsize=42)

            for c_idx in range(3):
                ax = axes[m_idx, c_idx]
                data = comps[c_idx] if comps else None
                gt_data = gt_comps[c_idx] if gt_comps else None
                plot_combined_horizontal_barplot(ax, data, color, ground_truth=gt_data)

                # HUGE column titles
                if m_idx == 0:
                    ax.set_title(["Trend", "Seasonal", "Resid."][c_idx], fontsize=38, pad=20)

    out_file = source_dir / f"{source_name}_grid_3x3_massive.png"
    plt.savefig(out_file, format='png', dpi=300)
    plt.close(fig)
    print(f"✓ Saved {source_name} 3x3 grid with massive labels to: {out_file}")

def main():
    # Process regular comparisons
    process_source(PROJECT_ROOT / 'results' / 'decompositions' / 'real_world', "real_world")
    synthetic_dir = PROJECT_ROOT / 'results' / 'decompositions' / 'synthetic'
    process_source(synthetic_dir, "synthetic", dataset_filter=['synth5', 'synth6', 'synth8'])
    
    # Process the ultra-bold 3x3 grid
    process_synthetic_grid(synthetic_dir, "synthetic")

if __name__ == "__main__":
    main()