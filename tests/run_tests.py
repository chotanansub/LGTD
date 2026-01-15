#!/usr/bin/env python
"""
Comprehensive test runner for LGTD with logging.

This script runs all tests and saves detailed logs to the log/ directory.
"""

import sys
import os
import subprocess
import datetime
from pathlib import Path
import json

# Ensure we're in the project root (parent of tests directory)
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

# Create log directory inside tests
LOG_DIR = PROJECT_ROOT / "tests" / "log"
LOG_DIR.mkdir(exist_ok=True)

# Generate timestamp for log files
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_command(cmd, log_file):
    """Run command and save output to log file."""
    print(f"Running: {' '.join(cmd)}")
    print(f"Logging to: {log_file}")

    with open(log_file, 'w') as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
        f.write("=" * 80 + "\n\n")

        try:
            # Set environment variable for coverage data file
            env = os.environ.copy()
            env['COVERAGE_FILE'] = str(PROJECT_ROOT / 'tests' / '.coverage')

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=PROJECT_ROOT,
                env=env
            )

            f.write(result.stdout)
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Exit code: {result.returncode}\n")

            return result.returncode == 0
        except Exception as e:
            f.write(f"\nERROR: {str(e)}\n")
            return False


def main():
    """Run comprehensive test suite."""
    print("=" * 80)
    print("LGTD Comprehensive Test Suite")
    print("=" * 80)
    print(f"Log directory: {LOG_DIR}")
    print(f"Timestamp: {TIMESTAMP}")
    print()

    results = {}

    # Test configurations
    test_configs = [
        {
            "name": "All Tests",
            "cmd": ["pytest", "tests/", "-v", "--tb=short"],
            "log": f"all_tests_{TIMESTAMP}.log"
        },
        {
            "name": "All Tests with Coverage",
            "cmd": ["pytest", "tests/", "-v", "--cov=src/lgtd", "--cov-report=term-missing", "--cov-report=html:tests/htmlcov", "-c", "tests/pytest.ini"],
            "log": f"coverage_{TIMESTAMP}.log"
        },
        {
            "name": "Module Tests",
            "cmd": ["pytest", "tests/test_modules/", "-v"],
            "log": f"module_tests_{TIMESTAMP}.log"
        },
        {
            "name": "Experiment Tests",
            "cmd": ["pytest", "tests/test_experiments/", "-v"],
            "log": f"experiment_tests_{TIMESTAMP}.log"
        },
        {
            "name": "lgtd Comprehensive Tests",
            "cmd": ["pytest", "tests/test_modules/test_lgtd_comprehensive.py", "-v"],
            "log": f"lgtd_comprehensive_{TIMESTAMP}.log"
        },
        {
            "name": "Metrics Comprehensive Tests",
            "cmd": ["pytest", "tests/test_modules/test_metrics_comprehensive.py", "-v"],
            "log": f"metrics_comprehensive_{TIMESTAMP}.log"
        },
        {
            "name": "Quick Sanity Check",
            "cmd": ["pytest", "tests/", "-x", "--tb=line"],
            "log": f"sanity_check_{TIMESTAMP}.log"
        },
    ]

    # Run each test configuration
    for i, config in enumerate(test_configs, 1):
        print(f"\n[{i}/{len(test_configs)}] {config['name']}")
        print("-" * 80)

        log_file = LOG_DIR / config['log']
        success = run_command(config['cmd'], log_file)

        results[config['name']] = {
            "success": success,
            "log_file": str(log_file),
            "command": " ".join(config['cmd'])
        }

        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"Status: {status}\n")

    # Save summary
    summary_file = LOG_DIR / f"test_summary_{TIMESTAMP}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "timestamp": TIMESTAMP,
            "results": results,
            "total_tests": len(test_configs),
            "passed": sum(1 for r in results.values() if r['success']),
            "failed": sum(1 for r in results.values() if not r['success'])
        }, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results.values() if r['success'])
    failed = sum(1 for r in results.values() if not r['success'])

    for name, result in results.items():
        status = "✓" if result['success'] else "✗"
        print(f"{status} {name}")

    print()
    print(f"Total: {len(test_configs)} test suites")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print()
    print(f"Summary saved to: {summary_file}")
    print(f"Coverage report: {PROJECT_ROOT / 'tests' / 'htmlcov' / 'index.html'}")
    print("=" * 80)

    # Exit with error if any tests failed
    if failed > 0:
        sys.exit(1)
    else:
        print("\n✓ All test suites passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
