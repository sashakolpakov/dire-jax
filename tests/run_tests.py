#!/usr/bin/env python
"""
Test runner for DiRe package unit tests.
Run this script from the root directory of the project.
"""

import sys
import os
import unittest

# Add the parent directory to sys.path so that the imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_all_tests():
    """Run all unit tests for the DiRe package."""
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('unit', pattern='test_*.py')
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    # Return non-zero exit code if tests failed
    sys.exit(0 if success else 1)
