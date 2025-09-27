#!/usr/bin/env python3
"""
Compiles a test CUDA kernel and ensures that GPU is available from cupy
"""

import pytest


class TestCupyCuda:
    """Test CuPy CUDA functionality."""

    def test_cupy_gpu_computation(self):
        """Test CuPy GPU array operations and CPU-GPU data transfer."""
        try:
            import cupy
            import numpy as np
        except ImportError:
            pytest.fail("CuPy not available")

        if not cupy.cuda.is_available():
            pytest.fail("CuPy CUDA support not available")

        device_count = cupy.cuda.runtime.getDeviceCount()
        assert device_count > 0, f"Expected GPU devices, found {device_count}"

        # Test GPU array operations
        a = cupy.random.randn(1000, 1000)
        b = cupy.random.randn(1000, 1000)
        c = cupy.dot(a, b)

        assert hasattr(c, 'device'), "Result is not a GPU array"

        # Test CPU-GPU data transfer
        cpu_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        gpu_array = cupy.asarray(cpu_array)
        gpu_result = gpu_array * 2 + 1
        cpu_result = cupy.asnumpy(gpu_result)

        expected = cpu_array * 2 + 1
        assert np.allclose(cpu_result, expected), "CPU-GPU transfer verification failed"

    def test_cupy_custom_kernels(self):
        """Test CuPy custom CUDA kernel compilation and execution."""
        try:
            import cupy
        except ImportError:
            pytest.fail("CuPy not available")

        if not cupy.cuda.is_available():
            pytest.fail("CUDA not available")

        # Define simple custom kernel
        kernel_code = '''
        extern "C" __global__
        void square_kernel(float* x, float* y, int n) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid < n) {
                y[tid] = x[tid] * x[tid];
            }
        }
        '''

        try:
            kernel = cupy.RawKernel(kernel_code, 'square_kernel')
        except Exception as e:
            pytest.fail(f"Custom kernel compilation failed: {e}")

        # Test kernel execution
        n = 1000
        x = cupy.random.randn(n, dtype=cupy.float32)
        y = cupy.zeros(n, dtype=cupy.float32)

        block_size = 256
        grid_size = (n + block_size - 1) // block_size
        kernel((grid_size,), (block_size,), (x, y, n))

        # Verify results
        expected = x * x
        assert cupy.allclose(y, expected), "Custom kernel execution verification failed"