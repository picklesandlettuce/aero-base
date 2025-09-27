#!/usr/bin/env python3
"""
Tests PyTorch GPU computation capabilities and cuDNN integration.
Validates PyTorch can perform tensor operations and neural network
computations on GPU hardware.
"""

import pytest


class TestPyTorchCuda:
    """Test PyTorch CUDA functionality."""

    def test_pytorch_gpu_computation(self):
        """Test PyTorch GPU tensor operations and autograd."""
        try:
            import torch
        except ImportError:
            pytest.fail("PyTorch not available")

        if not torch.cuda.is_available():
            pytest.fail("PyTorch CUDA support not available")

        device_count = torch.cuda.device_count()
        assert device_count > 0, f"Expected GPU devices, found {device_count}"

        # Test basic GPU operations
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')
        c = torch.matmul(a, b)

        assert c.device.type == 'cuda', "Matrix multiplication not executed on GPU"

        # Test autograd on GPU
        x = torch.randn(100, 100, device='cuda', requires_grad=True)
        y = torch.randn(100, 100, device='cuda', requires_grad=True)
        z = torch.matmul(x, y)
        loss = z.sum()
        loss.backward()

        assert x.grad is not None, "Autograd failed to compute gradients"
        assert x.grad.device.type == 'cuda', "Gradients not computed on GPU"

    def test_pytorch_cudnn_operations(self):
        """Test PyTorch cuDNN integration for neural network operations."""
        try:
            import torch
        except ImportError:
            pytest.fail("PyTorch not available")

        if not torch.cuda.is_available():
            pytest.fail("CUDA not available")

        if not torch.backends.cudnn.enabled:
            pytest.fail("cuDNN not enabled in PyTorch")

        # Test convolution operation (requires cuDNN)
        x = torch.randn(2, 3, 64, 64, device='cuda')
        conv = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
        y = conv(x)

        assert y.device.type == 'cuda', "Convolution not executed on GPU"
        assert y.shape == (2, 16, 64, 64), f"Unexpected convolution output shape: {y.shape}"

        # Test LSTM operation (requires cuDNN)
        lstm = torch.nn.LSTM(32, 64, batch_first=True).cuda()
        lstm_input = torch.randn(2, 10, 32, device='cuda')
        lstm_output, _ = lstm(lstm_input)

        assert lstm_output.device.type == 'cuda', "LSTM not executed on GPU"
        assert lstm_output.shape == (2, 10, 64), f"Unexpected LSTM output shape: {lstm_output.shape}"

    def test_torch_extensions_availability(self):
        """Test availability of torch extensions and their basic functionality."""
        extensions_tested = []

        # Test torchvision
        try:
            import torchvision
            import torch
            extensions_tested.append(f"torchvision {torchvision.__version__}")

            # Test a simple torchvision transform on GPU
            if torch.cuda.is_available():
                tensor = torch.randn(3, 224, 224, device='cuda')
                transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
                normalized = transform(tensor)
                assert normalized.device.type == 'cuda', "TorchVision transform not on GPU"
        except ImportError:
            pytest.fail("torchvision is required")

        # Test torchaudio
        import torchaudio
        extensions_tested.append(f"torchaudio {torchaudio.__version__}")

        # Test torch-tensorrt
        import torch_tensorrt
        extensions_tested.append(f"torch_tensorrt {torch_tensorrt.__version__}")

        # Test torchao
        import torchao
        extensions_tested.append(f"torchao {torchao.__version__}")

        # Test torchcodec
        import torchcodec
        extensions_tested.append(f"torchcodec {torchcodec.__version__}")

        print(f"Available torch extensions: {', '.join(extensions_tested)}")
        assert len(extensions_tested) > 0, "No torch extensions found"
