#!/usr/bin/env python3
"""
Tests ONNX Runtime GPU inference capabilities and provider availability.
Validates ONNX Runtime can perform GPU-accelerated inference.
"""

import pytest
import numpy as np


class TestOnnxRuntime:
    """Test ONNX Runtime functionality."""

    def test_onnxruntime_import(self):
        """Test ONNX Runtime import and basic functionality."""
        import onnxruntime as ort

        print(f"ONNX Runtime version: {ort.__version__}")
        print(f"Available providers: {ort.get_available_providers()}")

    def test_onnxruntime_gpu_provider(self):
        """Test ONNX Runtime GPU provider availability."""
        import onnxruntime as ort

        available_providers = ort.get_available_providers()

        # Check for CUDA provider
        if 'CUDAExecutionProvider' not in available_providers:
            pytest.fail(f"CUDA execution provider not available. Available: {available_providers}")

        print(f"CUDA provider available: CUDAExecutionProvider")

    def test_onnxruntime_tensorrt_provider(self):
        """Test ONNX Runtime TensorRT provider availability."""
        import onnxruntime as ort

        available_providers = ort.get_available_providers()

        if 'TensorrtExecutionProvider' not in available_providers:
            pytest.fail(f"TensorRT execution provider not available. Available: {available_providers}")

        print("TensorRT execution provider available")

    def test_onnxruntime_simple_inference(self):
        """Test simple GPU inference with ONNX Runtime."""
        import onnxruntime as ort

        # Check CUDA provider availability
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' not in available_providers:
            pytest.fail("CUDA execution provider not available")

        # Create simple model programmatically (addition operation)
        import onnx
        from onnx import helper, TensorProto

        # Define model: output = input1 + input2
        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [1, 10])
        input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, [1, 10])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10])

        add_node = helper.make_node('Add', ['input1', 'input2'], ['output'])
        graph = helper.make_graph([add_node], 'simple_add', [input1, input2], [output])
        model = helper.make_model(graph)

        # Create inference session with CUDA provider
        session = ort.InferenceSession(
            model.SerializeToString(),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        # Verify session is using CUDA
        providers = session.get_providers()
        assert 'CUDAExecutionProvider' in providers, f"CUDA provider not active: {providers}"

        # Test inference
        input_data1 = np.random.randn(1, 10).astype(np.float32)
        input_data2 = np.random.randn(1, 10).astype(np.float32)
        expected_output = input_data1 + input_data2

        result = session.run(['output'], {
            'input1': input_data1,
            'input2': input_data2
        })

        assert np.allclose(result[0], expected_output, rtol=1e-5), "GPU inference results incorrect"

        print(f"GPU inference successful - Input shapes: {input_data1.shape}, {input_data2.shape}")
        print(f"Output shape: {result[0].shape}")

    def test_onnxruntime_performance_comparison(self):
        """Test CPU vs GPU performance comparison with optimized settings."""
        import onnxruntime as ort
        import time

        # Check CUDA provider availability
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' not in available_providers:
            pytest.fail("CUDA execution provider not available")

        # Create larger model for meaningful performance test
        import onnx
        from onnx import helper, TensorProto

        # Massive workload for maximum GPU advantage
        batch_size = 256
        input_dim = 4096
        hidden_dim = 4096
        output_dim = 2048

        # Multi-layer model: input -> hidden -> output
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [batch_size, input_dim])
        weights1_tensor = helper.make_tensor_value_info('weights1', TensorProto.FLOAT, [input_dim, hidden_dim])
        weights2_tensor = helper.make_tensor_value_info('weights2', TensorProto.FLOAT, [hidden_dim, output_dim])
        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [batch_size, output_dim])

        # Layer 1: input @ weights1
        matmul1 = helper.make_node('MatMul', ['input', 'weights1'], ['hidden'])
        # ReLU activation
        relu1 = helper.make_node('Relu', ['hidden'], ['hidden_relu'])
        # Layer 2: hidden @ weights2
        matmul2 = helper.make_node('MatMul', ['hidden_relu', 'weights2'], ['output'])

        graph = helper.make_graph(
            [matmul1, relu1, matmul2],
            'mlp_test',
            [input_tensor, weights1_tensor, weights2_tensor],
            [output_tensor]
        )
        model = helper.make_model(graph)

        # Optimized session options
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # CPU session with optimizations
        cpu_session = ort.InferenceSession(
            model.SerializeToString(),
            sess_options=so,
            providers=['CPUExecutionProvider']
        )

        # GPU session with CUDA optimizations
        providers = [
            ("CUDAExecutionProvider", {
                "cudnn_conv_use_max_workspace": "1",
                "do_copy_in_default_stream": "1",
            }),
            "CPUExecutionProvider"
        ]

        gpu_session = ort.InferenceSession(
            model.SerializeToString(),
            sess_options=so,
            providers=providers
        )

        # Test data - seeded for reproducibility
        np.random.seed(0)
        input_data = np.random.randn(batch_size, input_dim).astype(np.float32)
        weights1_data = np.random.randn(input_dim, hidden_dim).astype(np.float32)
        weights2_data = np.random.randn(hidden_dim, output_dim).astype(np.float32)

        feed_dict = {
            'input': input_data,
            'weights1': weights1_data,
            'weights2': weights2_data
        }

        # Warmup runs to eliminate startup overhead
        print("Warming up sessions...")
        for _ in range(10):
            cpu_session.run(['output'], feed_dict)
            gpu_session.run(['output'], feed_dict)

        # CPU timing
        iterations = 50
        print(f"Running {iterations} iterations for CPU...")
        start_time = time.time()
        for _ in range(iterations):
            cpu_result = cpu_session.run(['output'], feed_dict)
        cpu_time = (time.time() - start_time) / iterations

        # GPU timing
        print(f"Running {iterations} iterations for GPU...")
        start_time = time.time()
        for _ in range(iterations):
            gpu_result = gpu_session.run(['output'], feed_dict)
        gpu_time = (time.time() - start_time) / iterations

        # Check numerical differences between CPU and GPU (both FP32)
        max_abs_diff = np.max(np.abs(cpu_result[0] - gpu_result[0]))
        max_rel_diff = np.max(np.abs((cpu_result[0] - gpu_result[0]) / (cpu_result[0] + 1e-8)))

        print(f"Max absolute difference: {max_abs_diff:.6f}")
        print(f"Max relative difference: {max_rel_diff:.6f}")

        # Verify results match (relaxed tolerance for large workloads with TF32/FP32 differences)
        assert np.allclose(cpu_result[0], gpu_result[0], rtol=1e-2, atol=15.0), "CPU and GPU results should match within expected precision"
        # Calculate FLOPS for context
        flops_per_inference = 2 * (batch_size * input_dim * hidden_dim + batch_size * hidden_dim * output_dim)
        cpu_gflops = (flops_per_inference / (cpu_time * 1e9))
        gpu_gflops = (flops_per_inference / (gpu_time * 1e9))

        print(f"Workload: {batch_size}×{input_dim} → {hidden_dim} → {output_dim} ({flops_per_inference/1e6:.1f}M FLOPs)")
        print(f"CPU inference time: {cpu_time*1000:.2f} ms ({cpu_gflops:.1f} GFLOPS)")
        print(f"GPU inference time: {gpu_time*1000:.2f} ms ({gpu_gflops:.1f} GFLOPS)")

        if gpu_time < cpu_time:
            speedup = cpu_time / gpu_time
            print(f"GPU speedup: {speedup:.2f}x")
        else:
            slowdown = gpu_time / cpu_time
            print(f"GPU slower by {slowdown:.2f}x (optimization opportunity or insufficient workload)")


    def test_onnxruntime_tensorrt_fp16_performance(self):
        """Test TensorRT FP16 optimized performance for maximum speed."""
        import onnxruntime as ort
        import time

        # Check TensorRT provider availability
        available_providers = ort.get_available_providers()
        if 'TensorrtExecutionProvider' not in available_providers:
            pytest.fail("TensorRT execution provider not available")

        # Create large model for TensorRT optimization
        import onnx
        from onnx import helper, TensorProto

        # Massive workload optimized for TensorRT Tensor Cores
        batch_size = 256
        input_dim = 4096
        hidden_dim = 4096
        output_dim = 2048

        # Multi-layer model: input -> hidden -> output
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [batch_size, input_dim])
        weights1_tensor = helper.make_tensor_value_info('weights1', TensorProto.FLOAT, [input_dim, hidden_dim])
        weights2_tensor = helper.make_tensor_value_info('weights2', TensorProto.FLOAT, [hidden_dim, output_dim])
        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [batch_size, output_dim])

        # Layer 1: input @ weights1
        matmul1 = helper.make_node('MatMul', ['input', 'weights1'], ['hidden'])
        # ReLU activation
        relu1 = helper.make_node('Relu', ['hidden'], ['hidden_relu'])
        # Layer 2: hidden @ weights2
        matmul2 = helper.make_node('MatMul', ['hidden_relu', 'weights2'], ['output'])

        graph = helper.make_graph(
            [matmul1, relu1, matmul2],
            'trt_fp16_test',
            [input_tensor, weights1_tensor, weights2_tensor],
            [output_tensor]
        )
        model = helper.make_model(graph)

        # Session options with full optimization
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # CPU baseline session
        cpu_session = ort.InferenceSession(
            model.SerializeToString(),
            sess_options=so,
            providers=['CPUExecutionProvider']
        )

        # TensorRT FP16 optimized session for maximum Orin performance
        trt_providers = [
            ("TensorrtExecutionProvider", {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "/tmp/ort_trt_cache",
                "trt_timing_cache_enable": True,
                "trt_max_workspace_size": 2147483648,  # 2GB workspace
            }),
            ("CUDAExecutionProvider", {
                "do_copy_in_default_stream": "1"
            }),
            "CPUExecutionProvider",
        ]

        trt_session = ort.InferenceSession(
            model.SerializeToString(),
            sess_options=so,
            providers=trt_providers
        )

        # Verify TensorRT is actually being used
        active_providers = trt_session.get_providers()
        if 'TensorrtExecutionProvider' not in active_providers:
            pytest.fail("TensorRT provider not active in session")

        print(f"Active providers: {active_providers}")

        # Test data
        np.random.seed(42)
        input_data = np.random.randn(batch_size, input_dim).astype(np.float32)
        weights1_data = np.random.randn(input_dim, hidden_dim).astype(np.float32)
        weights2_data = np.random.randn(hidden_dim, output_dim).astype(np.float32)

        feed_dict = {
            'input': input_data,
            'weights1': weights1_data,
            'weights2': weights2_data
        }

        # Extended warmup for TensorRT engine building
        print("Building TensorRT engine and warming up (this may take time on first run)...")
        for i in range(20):
            if i == 0:
                print("Building TensorRT engine...")
            trt_session.run(['output'], feed_dict)
            if i == 10:
                print("Engine built, continuing warmup...")

        print("Warming up CPU session...")
        for _ in range(10):
            cpu_session.run(['output'], feed_dict)

        # Timing comparison
        iterations = 100

        # CPU timing
        print(f"Running {iterations} iterations for CPU...")
        start_time = time.time()
        for _ in range(iterations):
            cpu_result = cpu_session.run(['output'], feed_dict)
        cpu_time = (time.time() - start_time) / iterations

        # TensorRT timing
        print(f"Running {iterations} iterations for TensorRT FP16...")
        start_time = time.time()
        for _ in range(iterations):
            trt_result = trt_session.run(['output'], feed_dict)
        trt_time = (time.time() - start_time) / iterations

        # Calculate FLOPS
        flops_per_inference = 2 * (batch_size * input_dim * hidden_dim + batch_size * hidden_dim * output_dim)
        cpu_gflops = (flops_per_inference / (cpu_time * 1e9))
        trt_gflops = (flops_per_inference / (trt_time * 1e9))

        print(f"Workload: {batch_size}×{input_dim} → {hidden_dim} → {output_dim} ({flops_per_inference/1e9:.1f}B FLOPs)")
        print(f"CPU inference time: {cpu_time*1000:.2f} ms ({cpu_gflops:.1f} GFLOPS)")
        print(f"TensorRT FP16 inference time: {trt_time*1000:.2f} ms ({trt_gflops:.1f} GFLOPS)")

        # Performance validation
        if trt_time < cpu_time:
            speedup = cpu_time / trt_time
            print(f"TensorRT FP16 speedup: {speedup:.2f}x")
            assert speedup > 1.5, f"Expected significant speedup, got {speedup:.2f}x"
        else:
            slowdown = trt_time / cpu_time
            print(f"TensorRT slower by {slowdown:.2f}x (unexpected)")

        # Verify TensorRT results are reasonable (very relaxed for FP16 vs FP32)
        max_abs_diff = np.max(np.abs(cpu_result[0] - trt_result[0]))
        max_rel_diff = np.max(np.abs((cpu_result[0] - trt_result[0]) / (cpu_result[0] + 1e-6)))

        print(f"Max absolute difference (FP32 vs FP16): {max_abs_diff:.3f}")
        print(f"Max relative difference (FP32 vs FP16): {max_rel_diff:.3f}")

        # Very relaxed tolerance for FP32 vs FP16 comparison
        assert np.allclose(cpu_result[0], trt_result[0], rtol=5e-2, atol=10.0), "TensorRT FP16 results should be within expected FP16 precision"

        print("TensorRT FP16 optimization test completed successfully")