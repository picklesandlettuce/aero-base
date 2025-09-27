#!/usr/bin/env python3
"""
Tests CUDA compiler availability and basic GPU runtime functionality
"""

import os
import subprocess
import tempfile
import pytest


class TestCudaToolkit:
    """Test CUDA toolkit installation and functionality on Jetson."""

    def test_nvcc_compiler_available(self):
        """Verify nvcc compiler is available and reports correct version."""
        errors = []

        try:
            result = subprocess.run(['nvcc', '--version'],
                                    capture_output=True, text=True, check=True)
            if 'release' not in result.stdout:
                errors.append("nvcc version output malformed")
            else:
                # Extract CUDA version for Jetson validation
                version_line = [line for line in result.stdout.split('\n') if 'release' in line][0]
                print(f"NVCC version: {version_line}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            errors.append(f"nvcc compiler not found or non-functional: {e}")

        if errors:
            pytest.fail(f"NVCC issues: {'; '.join(errors)}")

    def test_nvidia_smi_jetson_info(self):
        """Test nvidia-smi and verify Jetson GPU detection."""
        errors = []

        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)

            # Check for Jetson-specific GPU names
            jetson_gpus = ['Orin', 'Xavier', 'Tegra']
            found_jetson_gpu = any(gpu in result.stdout for gpu in jetson_gpus)

            if not found_jetson_gpu:
                errors.append(f"No Jetson GPU detected in nvidia-smi output")
            else:
                print("Jetson GPU detected in nvidia-smi")

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            errors.append(f"nvidia-smi failed: {e}")

        # Test GPU topology for multi-GPU Jetson systems
        try:
            topo_result = subprocess.run(['nvidia-smi', 'topo', '-m'],
                                       capture_output=True, text=True, timeout=10)
            if topo_result.returncode == 0:
                print("GPU topology information available")
        except Exception as e:
            errors.append(f"GPU topology check failed: {e}")

        if errors:
            pytest.fail(f"nvidia-smi issues: {'; '.join(errors)}")

    def test_cuda_runtime_jetson_compilation(self):
        """Test CUDA runtime with Jetson-optimized kernel compilation."""
        errors = []

        cuda_code = '''
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void jetson_test_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simple computation suitable for Jetson
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

int main() {
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);

    if (error != cudaSuccess) {
        printf("CUDA Error: %s\\n", cudaGetErrorString(error));
        return 1;
    }

    if (device_count == 0) {
        printf("No CUDA devices found\\n");
        return 1;
    }

    // Get device properties for Jetson validation
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\\n", prop.name);
    printf("Compute Capability: %d.%d\\n", prop.major, prop.minor);
    printf("Memory: %zu MB\\n", prop.totalGlobalMem / (1024*1024));

    // Test basic memory allocation
    const int N = 1024;
    size_t size = N * sizeof(float);
    float *d_data, *h_data;

    h_data = (float*)malloc(size);
    for(int i = 0; i < N; i++) h_data[i] = i;

    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Launch kernel with Jetson-appropriate grid size
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    jetson_test_kernel<<<gridSize, blockSize>>>(d_data, N);

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Verify first few results
    bool success = true;
    for(int i = 0; i < 10; i++) {
        float expected = i * 2.0f + 1.0f;
        if(abs(h_data[i] - expected) > 0.001f) {
            success = false;
            break;
        }
    }

    free(h_data);

    if(success) {
        printf("CUDA Jetson runtime test passed\\n");
        return 0;
    } else {
        printf("CUDA computation verification failed\\n");
        return 1;
    }
}
'''

        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = os.path.join(tmpdir, "test_jetson_cuda.cu")
            executable = os.path.join(tmpdir, "test_jetson_cuda")

            with open(source_file, 'w') as f:
                f.write(cuda_code)

            try:
                # Compile with Jetson-appropriate flags
                compile_result = subprocess.run([
                    'nvcc', '-o', executable, source_file,
                    '-arch=sm_87',  # Jetson Orin compute capability
                    '--ptxas-options=-v'
                ], capture_output=True, text=True, check=True)

                # Run the test
                run_result = subprocess.run([executable],
                                          capture_output=True, text=True, check=True)

                if "CUDA Jetson runtime test passed" not in run_result.stdout:
                    errors.append(f"CUDA runtime test did not pass: {run_result.stdout}")
                else:
                    print("CUDA Jetson runtime test completed successfully")

            except subprocess.CalledProcessError as e:
                errors.append(f"CUDA compilation failed: {e.stderr}")

        if errors:
            pytest.fail(f"CUDA runtime issues: {'; '.join(errors)}")

    def test_jetson_cuda_libraries_present(self):
        """Verify essential CUDA libraries for Jetson are present."""
        errors = []

        # Jetson-specific library paths
        jetson_lib_paths = [
            '/usr/local/cuda/lib64',
            '/usr/lib/aarch64-linux-gnu',  # ARM64 specific
            '/usr/local/cuda/targets/aarch64-linux/lib'
        ]

        required_libs = [
            'libcudart.so',
            'libcublas.so',
            'libcufft.so',
            'libcurand.so',
            'libcusparse.so'
        ]

        found_libs = []
        lib_locations = {}

        for lib_path in jetson_lib_paths:
            if os.path.exists(lib_path):
                for lib in required_libs:
                    lib_patterns = [lib, lib + '.*']  # Handle versioned libs
                    for pattern in lib_patterns:
                        full_path = os.path.join(lib_path, pattern.replace('.*', ''))
                        if os.path.exists(full_path) or any(
                            f.startswith(lib.replace('.so', '')) and '.so' in f
                            for f in os.listdir(lib_path) if f.startswith(lib.replace('.so', ''))
                        ):
                            if lib not in found_libs:
                                found_libs.append(lib)
                                lib_locations[lib] = lib_path
                            break

        missing_libs = [lib for lib in required_libs if lib not in found_libs]

        if missing_libs:
            errors.append(f"Missing critical CUDA libraries: {missing_libs}")

        if len(found_libs) < len(required_libs) // 2:
            errors.append(f"Too few CUDA libraries found. Found: {found_libs}")
        else:
            print(f"Found CUDA libraries: {found_libs}")
            for lib, path in lib_locations.items():
                print(f"  {lib} -> {path}")

        if errors:
            pytest.fail(f"CUDA library issues: {'; '.join(errors)}")

    def test_nvcc_tool_available(self):
        """Verify nvcc compiler tool is available and functional."""
        errors = []

        try:
            # Check if nvcc is in PATH
            result = subprocess.run(['which', 'nvcc'], capture_output=True, text=True)
            if result.returncode != 0:
                errors.append("nvcc not found in PATH")
            else:
                nvcc_path = result.stdout.strip()
                print(f"Found nvcc at: {nvcc_path}")

                # Test nvcc functionality
                version_result = subprocess.run(['nvcc', '--version'],
                                              capture_output=True, text=True, timeout=10)
                if version_result.returncode == 0:
                    print("nvcc is functional")
                else:
                    errors.append("nvcc found but not functional")

        except Exception as e:
            errors.append(f"Error checking nvcc: {e}")

        if errors:
            pytest.fail(f"NVCC tool issues: {'; '.join(errors)}")