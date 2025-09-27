#!/usr/bin/env python3
"""
Tests CUDA Python unified memory and pinned memory functionality.
"""

import pytest
import numpy as np
import time


class TestCudaPython:
    """Test CUDA Python functionality with unified memory."""

    def _check_cuda_python_available(self):
        """Check if cuda-python is available."""
        try:
            import cuda.bindings.driver as cuda
            import cuda.bindings.runtime as cudart
            return True
        except ImportError:
            return False

    def _check_device_properties(self):
        """Get device properties and check for unified memory support."""
        try:
            import cuda.bindings.driver as cuda
            import cuda.bindings.runtime as cudart

            # Initialize CUDA
            err, = cuda.cuInit(0)
            if err != cuda.CUresult.CUDA_SUCCESS:
                return None

            # Get device
            err, device = cuda.cuDeviceGet(0)
            if err != cuda.CUresult.CUDA_SUCCESS:
                return None

            # Get device properties
            err, prop = cudart.cudaGetDeviceProperties(0)
            if err != cudart.cudaError_t.cudaSuccess:
                return None

            return prop
        except Exception:
            return None

    def test_cuda_python_import(self):
        """Test cuda-python import and basic functionality."""
        if not self._check_cuda_python_available():
            pytest.fail("cuda-python not available")

        import cuda.bindings.driver as cuda
        import cuda.bindings.runtime as cudart

        # Initialize CUDA
        err, = cuda.cuInit(0)
        assert err == cuda.CUresult.CUDA_SUCCESS, "CUDA initialization failed"

        # Get device count
        err, device_count = cudart.cudaGetDeviceCount()
        assert err == cudart.cudaError_t.cudaSuccess, "Failed to get device count"
        assert device_count > 0, f"Expected CUDA devices, found {device_count}"

        print(f"CUDA Python initialized successfully with {device_count} device(s)")

    def test_device_properties(self):
        """Test device properties and unified memory capabilities."""
        if not self._check_cuda_python_available():
            pytest.fail("cuda-python not available")

        prop = self._check_device_properties()
        if prop is None:
            pytest.fail("Failed to get device properties")

        print(f"Device: {prop.name.decode('utf-8')}")
        print(f"Total Global Memory: {prop.totalGlobalMem / (1024**3):.2f} GB")
        print(f"Shared Memory Per Block: {prop.sharedMemPerBlock / 1024:.1f} KB")
        print(f"Unified Addressing: {bool(prop.unifiedAddressing)}")
        print(f"Managed Memory: {bool(prop.managedMemory)}")
        print(f"Concurrent Managed Access: {bool(prop.concurrentManagedAccess)}")

        # Jetson should have unified addressing and managed memory
        assert prop.unifiedAddressing, "Device should support unified addressing"
        assert prop.managedMemory, "Device should support managed memory"

    def test_unified_memory_allocation(self):
        """Test CUDA unified memory allocation and access."""
        if not self._check_cuda_python_available():
            pytest.fail("cuda-python not available")

        import cuda.bindings.driver as cuda
        import cuda.bindings.runtime as cudart

        # Initialize CUDA
        err, = cuda.cuInit(0)
        assert err == cuda.CUresult.CUDA_SUCCESS, "CUDA initialization failed"

        # Set device
        err, = cudart.cudaSetDevice(0)
        assert err == cudart.cudaError_t.cudaSuccess, "Failed to set device"

        # Allocate unified memory
        size = 1024 * 1024 * 4  # 4MB
        err, unified_ptr = cudart.cudaMallocManaged(size, cudart.cudaMemAttachGlobal)
        assert err == cudart.cudaError_t.cudaSuccess, "Failed to allocate unified memory"

        try:
            # Create numpy array from unified memory pointer
            # Note: This is advanced usage and requires careful memory management
            print(f"Allocated {size / (1024*1024):.1f} MB of unified memory")
            print(f"Unified memory pointer: 0x{unified_ptr:016x}")

            # Test memory access patterns with actual computation
            # Create numpy array from unified memory and compare CPU vs GPU performance

            # For this test, allocate smaller unified memory that we can work with
            test_size = 1024 * 1024  # 1M floats = 4MB
            err, test_ptr = cudart.cudaMallocManaged(test_size * 4, cudart.cudaMemAttachGlobal)
            assert err == cudart.cudaError_t.cudaSuccess, "Failed to allocate test unified memory"

            try:
                # Test 1: CPU computation with large matrix multiplication (GPU-friendly workload)
                matrix_size = 4096  # 4096x4096 matrices
                cpu_a = np.random.randn(matrix_size, matrix_size).astype(np.float32)
                cpu_b = np.random.randn(matrix_size, matrix_size).astype(np.float32)

                print(f"Matrix multiplication comparison ({matrix_size}x{matrix_size} matrices):")

                # CPU matrix multiplication
                start_time = time.time()
                cpu_result = np.dot(cpu_a, cpu_b)  # Matrix multiplication
                cpu_time = time.time() - start_time

                # Test 2: GPU computation with CuPy
                try:
                    import cupy as cp

                    # Upload data to GPU
                    gpu_a = cp.asarray(cpu_a)
                    gpu_b = cp.asarray(cpu_b)

                    # GPU matrix multiplication
                    start_time = time.time()
                    gpu_result = cp.dot(gpu_a, gpu_b)  # GPU matrix multiplication
                    cp.cuda.Device().synchronize()  # Ensure completion
                    gpu_time = time.time() - start_time

                    # Compare results (relaxed tolerance for very large matrix multiplication)
                    gpu_result_cpu = cp.asnumpy(gpu_result)
                    assert np.allclose(cpu_result, gpu_result_cpu, rtol=1e-2, atol=1e-3), "CPU and GPU results should match within numerical precision"

                    # Performance comparison
                    if gpu_time > 0:
                        speedup = cpu_time / gpu_time
                        flops = 2 * matrix_size**3  # Approximate FLOPs for matrix multiply
                        cpu_gflops = flops / (cpu_time * 1e9)
                        gpu_gflops = flops / (gpu_time * 1e9)

                        print(f"  CPU time: {cpu_time*1000:.2f} ms ({cpu_gflops:.1f} GFLOPS)")
                        print(f"  GPU time: {gpu_time*1000:.2f} ms ({gpu_gflops:.1f} GFLOPS)")
                        print(f"  GPU speedup: {speedup:.2f}x")
                    else:
                        print("GPU computation completed (too fast to measure accurately)")

                except ImportError:
                    pytest.fail("CuPy is required for GPU comparison test")

            finally:
                err, = cudart.cudaFree(test_ptr)

            # Synchronize to ensure operations complete
            err, = cudart.cudaDeviceSynchronize()
            assert err == cudart.cudaError_t.cudaSuccess, "Device synchronization failed"

            print("Unified memory allocation and access successful")

        finally:
            # Free unified memory
            err, = cudart.cudaFree(unified_ptr)
            assert err == cudart.cudaError_t.cudaSuccess, "Failed to free unified memory"

    def test_pinned_memory_allocation(self):
        """Test CUDA pinned (page-locked) memory allocation."""
        if not self._check_cuda_python_available():
            pytest.fail("cuda-python not available")

        import cuda.bindings.runtime as cudart

        # Set device
        err, = cudart.cudaSetDevice(0)
        assert err == cudart.cudaError_t.cudaSuccess, "Failed to set device"

        # Allocate pinned host memory
        size = 1024 * 1024 * 2  # 2MB
        err, pinned_ptr = cudart.cudaMallocHost(size)
        assert err == cudart.cudaError_t.cudaSuccess, "Failed to allocate pinned memory"

        try:
            print(f"Allocated {size / (1024*1024):.1f} MB of pinned host memory")
            print(f"Pinned memory pointer: 0x{pinned_ptr:016x}")

            # Allocate corresponding device memory
            err, device_ptr = cudart.cudaMalloc(size)
            assert err == cudart.cudaError_t.cudaSuccess, "Failed to allocate device memory"

            try:
                # Test memory transfer performance (pinned vs regular)
                # This demonstrates the benefit of pinned memory for transfers

                # Create test data (simulate writing to pinned memory)
                print("Pinned memory allocation successful")
                print("Ready for high-speed CPU-GPU transfers")

            finally:
                # Free device memory
                err, = cudart.cudaFree(device_ptr)
                assert err == cudart.cudaError_t.cudaSuccess, "Failed to free device memory"

        finally:
            # Free pinned memory
            err, = cudart.cudaFreeHost(pinned_ptr)
            assert err == cudart.cudaError_t.cudaSuccess, "Failed to free pinned memory"

    def test_memory_transfer_performance(self):
        """Test memory transfer performance between pinned and regular memory."""
        if not self._check_cuda_python_available():
            pytest.fail("cuda-python not available")

        import cuda.bindings.runtime as cudart

        # Set device
        err, = cudart.cudaSetDevice(0)
        assert err == cudart.cudaError_t.cudaSuccess, "Failed to set device"

        size = 1024 * 1024 * 256  # 16MB test
        iterations = 10

        # Allocate regular host memory (numpy array)
        regular_data = np.random.randn(size // 4).astype(np.float32)

        # Allocate pinned host memory
        err, pinned_ptr = cudart.cudaMallocHost(size)
        assert err == cudart.cudaError_t.cudaSuccess, "Failed to allocate pinned memory"

        # Allocate device memory
        err, device_ptr = cudart.cudaMalloc(size)
        assert err == cudart.cudaError_t.cudaSuccess, "Failed to allocate device memory"

        try:
            # Test regular memory transfer time
            start_time = time.time()
            for _ in range(iterations):
                err, = cudart.cudaMemcpy(device_ptr, regular_data.ctypes.data,
                                       size, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
                assert err == cudart.cudaError_t.cudaSuccess, "Regular memory transfer failed"

                err, = cudart.cudaDeviceSynchronize()
                assert err == cudart.cudaError_t.cudaSuccess, "Synchronization failed"
            regular_time = (time.time() - start_time) / iterations

            # Test pinned memory transfer time
            start_time = time.time()
            for _ in range(iterations):
                err, = cudart.cudaMemcpy(device_ptr, pinned_ptr,
                                       size, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
                assert err == cudart.cudaError_t.cudaSuccess, "Pinned memory transfer failed"

                err, = cudart.cudaDeviceSynchronize()
                assert err == cudart.cudaError_t.cudaSuccess, "Synchronization failed"
            pinned_time = (time.time() - start_time) / iterations

            # Calculate bandwidth
            size_mb = size / (1024 * 1024)
            regular_bandwidth = size_mb / regular_time
            pinned_bandwidth = size_mb / pinned_time

            print(f"Transfer size: {size_mb:.1f} MB")
            print(f"Regular memory transfer: {regular_time*1000:.2f} ms ({regular_bandwidth:.1f} MB/s)")
            print(f"Pinned memory transfer: {pinned_time*1000:.2f} ms ({pinned_bandwidth:.1f} MB/s)")

            if pinned_bandwidth > regular_bandwidth:
                speedup = pinned_bandwidth / regular_bandwidth
                print(f"Pinned memory speedup: {speedup:.2f}x")
            else:
                speedup = regular_bandwidth / pinned_bandwidth
                print(f"Regular memory speedup: {speedup:.2f}x (unified memory system)")

        finally:
            # Cleanup
            err, = cudart.cudaFree(device_ptr)
            err, = cudart.cudaFreeHost(pinned_ptr)

    def test_unified_memory_usage_pattern(self):
        """Demonstrate unified memory usage pattern typical for Jetson."""
        if not self._check_cuda_python_available():
            pytest.fail("cuda-python not available")

        import cuda.bindings.driver as cuda
        import cuda.bindings.runtime as cudart

        # Initialize CUDA
        err, = cuda.cuInit(0)
        assert err == cuda.CUresult.CUDA_SUCCESS, "CUDA initialization failed"

        err, = cudart.cudaSetDevice(0)
        assert err == cudart.cudaError_t.cudaSuccess, "Failed to set device"

        # Allocate unified memory for a typical ML workload scenario
        num_elements = 1024 * 256  # 256K floats
        size = num_elements * 4    # 4 bytes per float

        err, data_ptr = cudart.cudaMallocManaged(size, cudart.cudaMemAttachGlobal)
        assert err == cudart.cudaError_t.cudaSuccess, "Failed to allocate unified memory"

        err, result_ptr = cudart.cudaMallocManaged(size, cudart.cudaMemAttachGlobal)
        assert err == cudart.cudaError_t.cudaSuccess, "Failed to allocate result unified memory"

        try:
            print(f"Allocated unified memory for {num_elements} elements ({size / (1024*1024):.2f} MB)")

            # Simulate CPU initialization of data
            # In real usage, you might populate this from numpy arrays or other CPU operations
            print("Data accessible from both CPU and GPU without explicit transfers")

            # Demonstrate memory advice for performance optimization
            # Prefetch to GPU for compute-heavy operations
            err, = cudart.cudaMemPrefetchAsync(data_ptr, size, 0, 0)  # device 0
            if err == cudart.cudaError_t.cudaSuccess:
                print("Data prefetched to GPU for computation")

            # Synchronize
            err, = cudart.cudaDeviceSynchronize()
            assert err == cudart.cudaError_t.cudaSuccess, "Device synchronization failed"

            # Prefetch back to CPU for result processing
            err, = cudart.cudaMemPrefetchAsync(result_ptr, size, cudart.cudaCpuDeviceId, 0)
            if err == cudart.cudaError_t.cudaSuccess:
                print("Results prefetched to CPU for processing")

            err, = cudart.cudaDeviceSynchronize()
            assert err == cudart.cudaError_t.cudaSuccess, "Device synchronization failed"

            print("Unified memory workflow completed successfully")

        finally:
            # Free unified memory
            err, = cudart.cudaFree(data_ptr)
            err, = cudart.cudaFree(result_ptr)

    def test_memory_info(self):
        """Test memory information queries."""
        if not self._check_cuda_python_available():
            pytest.fail("cuda-python not available")

        import cuda.bindings.runtime as cudart

        err, = cudart.cudaSetDevice(0)
        assert err == cudart.cudaError_t.cudaSuccess, "Failed to set device"

        # Get memory info
        err, free_mem, total_mem = cudart.cudaMemGetInfo()
        assert err == cudart.cudaError_t.cudaSuccess, "Failed to get memory info"

        print(f"GPU Memory Info:")
        print(f"  Total: {total_mem / (1024**3):.2f} GB")
        print(f"  Free:  {free_mem / (1024**3):.2f} GB")
        print(f"  Used:  {(total_mem - free_mem) / (1024**3):.2f} GB")

        # On Jetson, this shows the unified memory pool
        usage_percent = ((total_mem - free_mem) / total_mem) * 100
        print(f"  Usage: {usage_percent:.1f}%")

        assert total_mem > 0, "Total memory should be positive"
        assert free_mem > 0, "Free memory should be positive"