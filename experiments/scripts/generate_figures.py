#!/usr/bin/env python
"""
Script to generate all figures for the paper.

Generates:
- Figure 2: Synthetic dataset decomposition examples
- Figure 3: Method comparison on synthetic data
- Figure 4: Parameter sensitivity visualization
- Figure 5: Real-world dataset results
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Generate all figures for the paper."""
    print("="*70)
    print("GENERATING FIGURES FOR LGTD PAPER")
    print("="*70)
    print()

    plots_dir = project_root / "experiments" / "scripts" / "plots"

    # List of figure generation scripts
    figure_scripts = [
        ("plot_synthetic.py", "Synthetic dataset decomposition examples (Figures 2-4)"),
        ("plot_method_comparison.py", "Method comparison visualization (Figure 3)"),
        ("plot_realworld.py", "Real-world dataset results (Figure 5)"),
    ]

    for script_name, description in figure_scripts:
        script_path = plots_dir / script_name
        if script_path.exists():
            print(f"Generating {description}...")
            print(f"  Running: {script_path}")

            # Import and execute the script
            spec = __import__('importlib.util').util.spec_from_file_location(
                script_name.replace('.py', ''), script_path
            )
            module = __import__('importlib.util').util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, 'main'):
                module.main()

            print(f"  ✓ {description} generated successfully\n")
        else:
            print(f"  ⚠ Warning: {script_name} not found, skipping...\n")

    print("="*70)
    print("✓ All figures generated successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
