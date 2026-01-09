#!/usr/bin/env python
"""
Script to generate all tables for the paper.

Generates:
- Table 1: MAE comparison on synthetic datasets
- Table 2: Decomposition quality metrics
- Table 3: Parameter sensitivity analysis
- Table 4: Real-world dataset performance
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Generate all tables for the paper."""
    print("="*70)
    print("GENERATING TABLES FOR LGTD PAPER")
    print("="*70)
    print()

    tables_dir = project_root / "experiments" / "scripts" / "tables"

    # List of table generation scripts
    table_scripts = [
        ("generate_mae_tables.py", "MAE comparison tables (Table 1)"),
        ("generate_decomposition_tables.py", "Decomposition quality metrics (Table 2)"),
        ("generate_parameter_table.py", "Parameter sensitivity analysis (Table 3)"),
    ]

    for script_name, description in table_scripts:
        script_path = tables_dir / script_name
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
    print("✓ All tables generated successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
