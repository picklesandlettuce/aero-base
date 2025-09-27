#!/usr/bin/env python3
"""
Tests OpenCV installation, CUDA support, and GPU image processing capabilities.
Validates OpenCV can perform image operations on GPU hardware.
"""

import pytest
import cv2
import numpy as np
import os


class TestOpenCVCuda:
    """Test OpenCV CUDA functionality."""

    @pytest.fixture(scope="class")
    def test_image_path(self):
        """Fixture providing path to test image."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "lena.jpg")

        if not os.path.exists(image_path):
            pytest.fail(f"Test image not found at {image_path}")

        return image_path

    @pytest.fixture(scope="class")
    def test_image_cpu(self, test_image_path):
        """Fixture providing loaded test image on CPU."""
        img_cpu = cv2.imread(test_image_path)
        if img_cpu is None:
            pytest.fail(f"Failed to load test image from {test_image_path}")

        print(f"Loaded test image: {img_cpu.shape} {img_cpu.dtype}")
        return img_cpu

    @pytest.fixture(scope="class")
    def test_image_gpu(self, test_image_cpu):
        """Fixture providing test image uploaded to GPU."""
        if not self._check_cuda_support():
            pytest.fail("OpenCV CUDA support not available")

        img_gpu = cv2.cuda_GpuMat()
        img_gpu.upload(test_image_cpu)
        return img_gpu

    def _check_cuda_support(self):
        """Helper method to check if OpenCV has CUDA support."""
        try:
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            return device_count > 0
        except Exception:
            return False

    def test_opencv_version(self):
        """Test OpenCV version and build information."""
        version = cv2.__version__
        assert version is not None and len(version) > 0, "OpenCV version should not be empty"

        print(f"OpenCV version: {version}")

        # Print build information for debugging
        build_info = cv2.getBuildInformation()
        print("OpenCV Build Information:")
        print(build_info)

    def test_opencv_cuda_support(self):
        """Test OpenCV CUDA support and device detection."""
        try:
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            assert device_count > 0, f"Expected CUDA devices, found {device_count}"
            print(f"OpenCV CUDA devices: {device_count}")
        except Exception as ex:
            pytest.fail(f"OpenCV was not built with CUDA support: {ex}")

    def test_image_loading(self, test_image_cpu):
        """Test basic image loading functionality."""
        assert test_image_cpu is not None, "Test image should load successfully"
        assert len(test_image_cpu.shape) == 3, f"Expected 3D image, got shape {test_image_cpu.shape}"
        assert test_image_cpu.dtype == np.uint8, f"Expected uint8 dtype, got {test_image_cpu.dtype}"

    def test_gpu_image_upload(self, test_image_cpu):
        """Test uploading image to GPU memory."""
        if not self._check_cuda_support():
            pytest.fail("OpenCV CUDA support not available")

        img_gpu = cv2.cuda_GpuMat()
        img_gpu.upload(test_image_cpu)

        # Verify GPU image dimensions match CPU image
        assert img_gpu.size() == (test_image_cpu.shape[1], test_image_cpu.shape[0]), \
            "GPU image dimensions should match CPU image"

    def test_gpu_resize_operation(self, test_image_gpu, test_image_cpu):
        """Test GPU image resize operation."""
        if not self._check_cuda_support():
            pytest.fail("OpenCV CUDA support not available")

        # Resize to half the original size
        new_height = int(test_image_cpu.shape[0] / 2)
        new_width = int(test_image_cpu.shape[1] / 2)

        img_resized_gpu = cv2.cuda.resize(test_image_gpu, (new_width, new_height))

        # Verify resized dimensions
        assert img_resized_gpu.size() == (new_width, new_height), \
            f"Resized image should be {new_width}x{new_height}"

    def test_gpu_color_conversion(self, test_image_gpu):
        """Test GPU color space conversion operations."""
        if not self._check_cuda_support():
            pytest.fail("OpenCV CUDA support not available")

        # Test BGR to LUV conversion
        luv_gpu = cv2.cuda.cvtColor(test_image_gpu, cv2.COLOR_BGR2LUV)
        luv_cpu = luv_gpu.download()

        assert luv_cpu is not None, "LUV conversion should succeed"
        assert luv_cpu.shape[2] == 3, "LUV image should have 3 channels"

        # Test BGR to HSV conversion
        hsv_gpu = cv2.cuda.cvtColor(test_image_gpu, cv2.COLOR_BGR2HSV)
        hsv_cpu = hsv_gpu.download()

        assert hsv_cpu is not None, "HSV conversion should succeed"
        assert hsv_cpu.shape[2] == 3, "HSV image should have 3 channels"

        # Test BGR to Grayscale conversion
        gray_gpu = cv2.cuda.cvtColor(test_image_gpu, cv2.COLOR_BGR2GRAY)
        gray_cpu = gray_gpu.download()

        assert gray_cpu is not None, "Grayscale conversion should succeed"
        assert len(gray_cpu.shape) == 2, "Grayscale image should have 2 dimensions"

    def test_gpu_clahe_operation(self, test_image_gpu):
        """Test GPU CLAHE (Contrast Limited Adaptive Histogram Equalization) operation."""
        if not self._check_cuda_support():
            pytest.fail("OpenCV CUDA support not available")

        # Convert to grayscale first
        gray_gpu = cv2.cuda.cvtColor(test_image_gpu, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE
        clahe = cv2.cuda.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        clahe_gpu = clahe.apply(gray_gpu, cv2.cuda_Stream.Null())
        clahe_cpu = clahe_gpu.download()

        assert clahe_cpu is not None, "CLAHE operation should succeed"
        assert len(clahe_cpu.shape) == 2, "CLAHE result should be grayscale"
        assert clahe_cpu.dtype == np.uint8, "CLAHE result should be uint8"

    def test_gpu_download_operation(self, test_image_gpu, test_image_cpu):
        """Test downloading image from GPU back to CPU."""
        if not self._check_cuda_support():
            pytest.fail("OpenCV CUDA support not available")

        downloaded_img = test_image_gpu.download()

        assert downloaded_img is not None, "Download operation should succeed"
        assert downloaded_img.shape == test_image_cpu.shape, \
            "Downloaded image shape should match original"
        assert downloaded_img.dtype == test_image_cpu.dtype, \
            "Downloaded image dtype should match original"

    def test_opencv_integration_workflow(self, test_image_cpu):
        """Test complete OpenCV GPU workflow integration."""
        if not self._check_cuda_support():
            pytest.fail("OpenCV CUDA support not available")

        # Complete workflow: upload -> resize -> color convert -> CLAHE -> download
        img_gpu = cv2.cuda_GpuMat()
        img_gpu.upload(test_image_cpu)

        # Resize
        resized_gpu = cv2.cuda.resize(img_gpu, (int(test_image_cpu.shape[1]/2), int(test_image_cpu.shape[0]/2)))

        # Convert to grayscale
        gray_gpu = cv2.cuda.cvtColor(resized_gpu, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE
        clahe = cv2.cuda.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        result_gpu = clahe.apply(gray_gpu, cv2.cuda_Stream.Null())

        # Download final result
        final_result = result_gpu.download()

        assert final_result is not None, "Complete workflow should succeed"
        expected_shape = (int(test_image_cpu.shape[0]/2), int(test_image_cpu.shape[1]/2))
        assert final_result.shape == expected_shape, \
            f"Final result shape should be {expected_shape}"

        print("OpenCV GPU workflow completed successfully")