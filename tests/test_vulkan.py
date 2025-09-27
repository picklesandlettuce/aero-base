#!/usr/bin/env python3
"""
Tests Vulkan API availability and basic functionality.
Validates Vulkan drivers and GPU detection.
"""

import pytest
import subprocess


class TestVulkan:
    """Test Vulkan functionality."""

    def test_vulkaninfo_summary(self):
        """Test Vulkan installation and GPU detection using vulkaninfo --summary."""
        try:
            # Run vulkaninfo --summary to check Vulkan availability
            result = subprocess.run(['vulkaninfo', '--summary'],
                                    capture_output=True, text=True, check=True, timeout=30)

            # Check if output contains expected Vulkan information
            output = result.stdout.lower()

            # Basic checks for Vulkan functionality
            assert 'vulkan instance' in output or 'instance version' in output, \
                "Vulkan instance information not found in output"

            # Check for physical devices
            assert 'gpu' in output or 'device' in output or 'physical device' in output, \
                "No GPU/physical device information found in Vulkan output"

            print("Vulkan Summary Output:")
            print(result.stdout)

        except subprocess.CalledProcessError as e:
            pytest.fail(f"vulkaninfo command failed with return code {e.returncode}: {e.stderr}")
        except subprocess.TimeoutExpired:
            pytest.fail("vulkaninfo command timed out after 30 seconds")
        except FileNotFoundError:
            pytest.fail("vulkaninfo command not found - Vulkan tools not installed")

    def test_vulkan_version(self):
        """Test that vulkaninfo can report version information."""
        try:
            # Run vulkaninfo (it outputs version info by default)
            result = subprocess.run(['vulkaninfo'],
                                    capture_output=True, text=True, check=True, timeout=10)

            output = result.stdout.lower()
            assert 'vulkan instance version' in output, "Output should contain 'vulkan instance version'"

            # Extract version info from output
            version_lines = [line for line in result.stdout.split('\n') if 'vulkan instance version' in line.lower()]
            if version_lines:
                print(f"Vulkan version info: {version_lines[0].strip()}")
            else:
                print("Vulkan version extracted from output")

        except subprocess.CalledProcessError as e:
            pytest.fail(f"vulkaninfo failed with return code {e.returncode}: {e.stderr}")
        except subprocess.TimeoutExpired:
            pytest.fail("vulkaninfo timed out")
        except FileNotFoundError:
            pytest.skip("vulkaninfo command not found - skipping version test")