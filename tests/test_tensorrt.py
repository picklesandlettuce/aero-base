#!/usr/bin/env python3
"""
Tests TensorRT GPU inference engine creation and tool availability.
Validates TensorRT can build and execute inference engines on GPU hardware.
"""

import subprocess
import sys
import tempfile
import pytest
import os
import signal


class TestTensorRtCuda:
    """Test TensorRT CUDA functionality."""

    def test_tensorrt_import(self):
        """Test that TensorRT can be imported without crashing."""
        try:
            import tensorrt as trt
            assert hasattr(trt, '__version__'), "TensorRT version not available"
            print(f"TensorRT version: {trt.__version__}")
        except ImportError:
            pytest.fail("TensorRT Python module not available")

    def test_cuda_driver_available(self):
        """Test CUDA driver availability before TensorRT tests."""
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.fail("CUDA not available through PyTorch")

            device_count = torch.cuda.device_count()
            assert device_count > 0, f"Expected GPU devices, found {device_count}"
            print(f"CUDA devices: {device_count}")
        except ImportError:
            pytest.fail("PyTorch is required for CUDA tests")

    def test_tensorrt_logger_creation(self):
        """Test TensorRT logger creation in isolation."""
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        assert logger is not None, "Failed to create TensorRT logger"

    def test_tensorrt_runtime_creation(self):
        """Test TensorRT runtime creation."""
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        assert runtime is not None, "Failed to create TensorRT runtime"

    def test_tensorrt_builder_creation(self):
        """Test TensorRT builder creation."""
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        assert builder is not None, "Failed to create TensorRT builder"

    def test_tensorrt_gpu_inference_engine(self):
        """Test TensorRT GPU inference engine creation and execution."""
        python_test = '''
import sys
import numpy as np
try:
    import tensorrt as trt
    import torch

    # Check GPU availability first
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        sys.exit(2)

    # Create logger
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # Create network with explicit batch dimension
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # Create config for GPU
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)  # 1MB workspace

    # Build a simple identity network
    input_tensor = network.add_input(name="input", dtype=trt.float32, shape=(1, 3, 224, 224))
    identity = network.add_identity(input_tensor)
    network.mark_output(identity.get_output(0))

    # Build engine
    serialized_engine = builder.build_serialized_network(network, config)
    if not serialized_engine:
        print("ERROR: Failed to build engine")
        sys.exit(1)

    # Create runtime and deserialize
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if not engine:
        print("ERROR: Failed to deserialize engine")
        sys.exit(1)

    # Create execution context
    context = engine.create_execution_context()
    if not context:
        print("ERROR: Failed to create execution context")
        sys.exit(1)

    print(f"SUCCESS: TensorRT GPU engine created with {engine.num_io_tensors} tensors")
    print(f"Input shape: {engine.get_tensor_shape('input')}")
    print(f"Output shape: {engine.get_tensor_shape(engine.get_tensor_name(1))}")

except ImportError as e:
    print(f"SKIP: Import failed - {e}")
    sys.exit(2)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''

        try:
            result = subprocess.run([sys.executable, '-c', python_test],
                                    capture_output=True, text=True, timeout=120)

            if result.returncode == 2:
                pytest.skip("TensorRT GPU test requirements not available")
            elif result.returncode != 0:
                pytest.fail(f"TensorRT GPU engine creation failed: {result.stdout}\nstderr: {result.stderr}")

            assert "SUCCESS" in result.stdout, "TensorRT GPU engine creation did not complete"
            assert "tensors" in result.stdout, "Engine tensor information not found"

        except subprocess.TimeoutExpired:
            pytest.fail("TensorRT GPU engine creation timed out")

    def test_tensorrt_tools_available(self):
        """Test TensorRT command-line tools availability."""
        tools = ['trtexec']
        found_tools = []

        for tool in tools:
            try:
                result = subprocess.run(['which', tool], capture_output=True, text=True)
                if result.returncode == 0:
                    # Verify tool functionality
                    version_result = subprocess.run([tool, '--help'],
                                                    capture_output=True, text=True, timeout=10)
                    if 'TensorRT' in version_result.stdout or 'TensorRT' in version_result.stderr:
                        found_tools.append(tool)
            except Exception:
                continue

        assert len(found_tools) > 0, "No TensorRT tools found"
        assert 'trtexec' in found_tools, "trtexec tool is required but not found"

    def test_tensorrt_gpu_inference_execution(self):
        """Test actual GPU inference execution with TensorRT."""
        python_test = '''
import sys
import numpy as np
try:
    import tensorrt as trt
    import torch
    import pycuda.driver as cuda
    import pycuda.autoinit

    # Check GPU availability
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        sys.exit(2)

    print("Creating TensorRT engine for GPU inference...")

    # Create logger
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # Create network with explicit batch
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # Create config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 24)  # 16MB workspace

    # Build simple addition network: output = input + 1.0
    input_tensor = network.add_input(name="input", dtype=trt.float32, shape=(1, 10))
    constant = network.add_constant(shape=(1, 10), weights=np.ones((1, 10), dtype=np.float32))
    add_layer = network.add_elementwise(input_tensor, constant.get_output(0), trt.ElementWiseOperation.SUM)
    network.mark_output(add_layer.get_output(0))

    # Build engine
    print("Building TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    if not serialized_engine:
        print("ERROR: Failed to build engine")
        sys.exit(1)

    # Create runtime and deserialize
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()

    # Allocate GPU memory
    input_size = 1 * 10 * 4  # batch_size * elements * sizeof(float32)
    output_size = input_size

    # Allocate GPU memory using PyCUDA
    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)

    # Prepare test data
    input_data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32)
    expected_output = input_data + 1.0

    # Copy input to GPU
    cuda.memcpy_htod(d_input, input_data)

    # Execute inference
    print("Executing inference on GPU...")
    context.set_tensor_address("input", int(d_input))
    context.set_tensor_address(engine.get_tensor_name(1), int(d_output))
    context.execute_async_v3(0)

    # Copy output back to CPU
    output_data = np.empty((1, 10), dtype=np.float32)
    cuda.memcpy_dtoh(output_data, d_output)

    # Verify results
    if np.allclose(output_data, expected_output, rtol=1e-5):
        print(f"SUCCESS: GPU inference executed correctly")
        print(f"Input:    {input_data[0]}")
        print(f"Output:   {output_data[0]}")
        print(f"Expected: {expected_output[0]}")
    else:
        print(f"ERROR: Inference results incorrect")
        print(f"Expected: {expected_output[0]}")
        print(f"Got:      {output_data[0]}")
        sys.exit(1)

except ImportError as e:
    print(f"SKIP: Import failed - {e}")
    sys.exit(2)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''

        try:
            result = subprocess.run([sys.executable, '-c', python_test],
                                    capture_output=True, text=True, timeout=180)

            if result.returncode == 2:
                pytest.skip("TensorRT GPU execution test requirements not available")
            elif result.returncode != 0:
                pytest.fail(f"TensorRT GPU execution failed: {result.stdout}\nstderr: {result.stderr}")

            assert "SUCCESS" in result.stdout, "TensorRT GPU execution did not complete successfully"
            assert "Input:" in result.stdout and "Output:" in result.stdout, "Inference results not shown"

        except subprocess.TimeoutExpired:
            pytest.fail("TensorRT GPU execution timed out")