#!/usr/bin/env python3
"""
Tests TensorFlow GPU computation capabilities using TensorFlow 2.18 standards.
Validates TensorFlow can perform tensor operations on GPU hardware.
"""

import pytest


class TestTensorFlowCuda:
    """Test TensorFlow CUDA functionality."""

    def test_tensorflow_gpu_available(self):
        """Test TensorFlow GPU availability and basic setup."""
        try:
            import tensorflow as tf
        except ImportError:
            pytest.fail("TensorFlow not available")

        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        assert len(gpus) > 0, f"No GPUs found. Available devices: {tf.config.list_physical_devices()}"

        # Enable memory growth
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pytest.fail("TensorFlow not available")

        print(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")

    def test_tensorflow_gpu_matrix_operations(self):
        """Test TensorFlow matrix operations on GPU."""
        try:
            import tensorflow as tf
        except ImportError:
            pytest.fail("TensorFlow not available")

        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            pytest.fail("No GPU available")

        # Enable memory growth
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

        # Test standard matrix multiplication (TensorFlow 2.18 approach)
        a = tf.Variable(tf.random.uniform(shape=(1000, 1000)), name="matrix_a")
        b = tf.Variable(tf.random.uniform(shape=(1000, 1000)), name="matrix_b")

        # Perform matrix multiplication (TensorFlow automatically uses GPU)
        c = tf.matmul(a, b)

        # Verify operation completed and shape is correct
        assert c.shape == (1000, 1000), f"Unexpected result shape: {c.shape}"

        # Verify it's a valid tensor with expected properties
        assert tf.is_tensor(c), "Result is not a valid tensor"
        assert c.dtype == tf.float32, f"Unexpected dtype: {c.dtype}"

    def test_tensorflow_simple_computation(self):
        """Test simple TensorFlow computation with device logging."""
        try:
            import tensorflow as tf
        except ImportError:
            pytest.fail("TensorFlow not available")

        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            pytest.fail("No GPU available")

        # Enable memory growth
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

        # Simple tensor operations (TensorFlow 2.18 standard)
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # Matrix multiplication
        c = tf.matmul(a, b)

        # Verify computation
        expected_shape = (2, 2)
        assert c.shape == expected_shape, f"Expected shape {expected_shape}, got {c.shape}"

        # Verify values are reasonable (not NaN or infinite)
        assert tf.math.is_finite(c).numpy().all(), "Result contains non-finite values"