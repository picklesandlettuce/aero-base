#!/usr/bin/env python3
"""
PyTest configuration and fixtures for CUDA test suite.

Provides common test utilities and GPU hardware validation.
"""

import pytest
import subprocess




@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU hardware is available."""
    try:
        result = subprocess.run(['nvidia-smi'],
                                capture_output=True, text=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


@pytest.fixture(scope="session")
def nvcc_available():
    """Check if CUDA compiler is available."""
    try:
        subprocess.run(['nvcc', '--version'],
                       capture_output=True, text=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False




