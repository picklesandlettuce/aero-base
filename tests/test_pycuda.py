#!/usr/bin/env python3
"""
Tests PyCUDA GPU computation, memory management, and custom kernel compilation.
Validates PyCUDA can perform GPU operations and compile CUDA kernels.
"""

import pytest


class TestPycuda:
    """Test PyCUDA functionality."""

    def test_pycuda_device_initialization(self):
        """Test PyCUDA device initialization and context management."""
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            pytest.fail("PyCUDA not available")

        # Test device count and basic properties
        cuda.init()
        device_count = cuda.Device.count()
        assert device_count > 0, f"Expected GPU devices, found {device_count}"

        # Test device properties
        device = cuda.Device(0)
        name = device.name()
        compute_capability = device.compute_capability()
        total_memory = device.total_memory()

        assert len(name) > 0, "Device name should not be empty"
        assert compute_capability[0] >= 3, f"Compute capability too low: {compute_capability}"
        assert total_memory > 0, f"Total memory should be positive: {total_memory}"

        print(f"PyCUDA Device: {name}")
        print(f"Compute Capability: {compute_capability}")
        print(f"Total Memory: {total_memory // (1024**2)} MB")

    def test_pycuda_memory_operations(self):
        """Test PyCUDA GPU memory allocation and data transfer."""
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            import pycuda.gpuarray as gpuarray
            import numpy as np
        except ImportError:
            pytest.fail("PyCUDA or NumPy not available")

        # Test basic GPU array operations
        a_cpu = np.random.randn(4, 4).astype(np.float32)
        b_cpu = np.random.randn(4, 4).astype(np.float32)

        # Transfer to GPU
        a_gpu = gpuarray.to_gpu(a_cpu)
        b_gpu = gpuarray.to_gpu(b_cpu)

        # Perform GPU operations
        c_gpu = a_gpu + b_gpu
        c_cpu = c_gpu.get()

        # Verify results
        expected = a_cpu + b_cpu
        assert np.allclose(c_cpu, expected), "GPU array addition verification failed"

        # Test memory management
        assert a_gpu.nbytes > 0, "GPU array should have allocated memory"
        assert c_gpu.shape == (4, 4), f"Result shape incorrect: {c_gpu.shape}"

    def test_pycuda_custom_kernel(self):
        """Test PyCUDA custom CUDA kernel compilation and execution."""
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            import pycuda.gpuarray as gpuarray
            import pycuda.compiler as compiler
            import numpy as np
        except ImportError:
            pytest.fail("PyCUDA or NumPy not available")

        # Define a simple vector addition kernel
        kernel_code = """
        __global__ void vector_add(float *a, float *b, float *c, int n)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
        """

        try:
            mod = compiler.SourceModule(kernel_code)
            vector_add = mod.get_function("vector_add")
        except Exception as e:
            pytest.fail(f"Custom kernel compilation failed: {e}")

        # Test kernel execution
        n = 1024
        a_cpu = np.random.randn(n).astype(np.float32)
        b_cpu = np.random.randn(n).astype(np.float32)

        # Allocate GPU memory
        a_gpu = gpuarray.to_gpu(a_cpu)
        b_gpu = gpuarray.to_gpu(b_cpu)
        c_gpu = gpuarray.empty_like(a_gpu)

        # Execute kernel
        block_size = 256
        grid_size = (n + block_size - 1) // block_size
        vector_add(
            a_gpu, b_gpu, c_gpu,
            np.int32(n),
            block=(block_size, 1, 1),
            grid=(grid_size, 1)
        )

        # Verify results
        c_cpu = c_gpu.get()
        expected = a_cpu + b_cpu
        assert np.allclose(c_cpu, expected), "Custom kernel execution verification failed"

    def test_pycuda_matrix_multiplication(self):
        """Test PyCUDA matrix multiplication using custom kernel."""
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            import pycuda.gpuarray as gpuarray
            import pycuda.compiler as compiler
            import numpy as np
        except ImportError:
            pytest.fail("PyCUDA or NumPy not available")

        # Simple matrix multiplication kernel
        kernel_code = """
        __global__ void matrix_mult(float *a, float *b, float *c, int n)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            if (row < n && col < n) {
                float sum = 0.0f;
                for (int k = 0; k < n; k++) {
                    sum += a[row * n + k] * b[k * n + col];
                }
                c[row * n + col] = sum;
            }
        }
        """

        try:
            mod = compiler.SourceModule(kernel_code)
            matrix_mult = mod.get_function("matrix_mult")
        except Exception as e:
            pytest.fail(f"Matrix multiplication kernel compilation failed: {e}")

        # Test with small matrices
        n = 16
        a_cpu = np.random.randn(n, n).astype(np.float32)
        b_cpu = np.random.randn(n, n).astype(np.float32)

        # Allocate GPU memory
        a_gpu = gpuarray.to_gpu(a_cpu)
        b_gpu = gpuarray.to_gpu(b_cpu)
        c_gpu = gpuarray.empty((n, n), dtype=np.float32)

        # Execute kernel
        block_size = 16
        grid_size = (n + block_size - 1) // block_size
        matrix_mult(
            a_gpu, b_gpu, c_gpu,
            np.int32(n),
            block=(block_size, block_size, 1),
            grid=(grid_size, grid_size)
        )

        # Verify results
        c_cpu = c_gpu.get()
        expected = np.dot(a_cpu, b_cpu)
        assert np.allclose(c_cpu, expected, rtol=1e-5), "Matrix multiplication verification failed"

    def test_pycuda_reduction_operation(self):
        """Test PyCUDA reduction operation (sum) using custom kernel."""
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            import pycuda.gpuarray as gpuarray
            import pycuda.reduction as reduction
            import numpy as np
        except ImportError:
            pytest.fail("PyCUDA or NumPy not available")

        # Test using PyCUDA's built-in reduction
        try:
            sum_kernel = reduction.ReductionKernel(
                np.float32,
                neutral="0",
                reduce_expr="a+b",
                map_expr="x[i]",
                arguments="float *x"
            )
        except Exception as e:
            pytest.fail(f"Reduction kernel creation failed: {e}")

        # Test reduction operation
        n = 10000
        a_cpu = np.random.randn(n).astype(np.float32)
        a_gpu = gpuarray.to_gpu(a_cpu)

        # Perform reduction on GPU
        gpu_sum = sum_kernel(a_gpu).get()
        cpu_sum = np.sum(a_cpu)

        assert np.allclose(gpu_sum, cpu_sum, rtol=1e-5), "Reduction operation verification failed"
        print(f"GPU sum: {gpu_sum}, CPU sum: {cpu_sum}")